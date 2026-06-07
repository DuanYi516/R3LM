"""Stage-II reason-conditioned regression training for R3LM."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from r3lm.hf_assets import ensure_data_dir
from r3lm.input_assembly import INPUT_MODES, build_model_frame


def _default_prompt_prefix(cell_line: str) -> str:
    return (
        f"You are a computational regulatory genomics expert specializing in "
        f"{cell_line} cell line biology. Your task is to analyze the provided "
        f"DNA sequence features to predict the gene expression level (0-3).\n"
        f"Provide a step-by-step reasoning chain explaining the biological mechanism.\n\n"
    )


@dataclass
class RegressionConfig:
    cell_line: str = "K562"
    input_mode: str = "generated_cot"
    text_model_name_or_path: str = "Qwen/Qwen3-4B-Instruct-2507"
    cache_dir: Optional[str] = None
    local_files_only: bool = False
    max_length_text: int = 2048

    lr: float = 1e-5
    weight_decay: float = 0.01
    max_epochs: int = 10
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    max_grad_norm: float = 200.0

    finetune_mode: str = "lora"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    num_workers: int = 8
    devices: Any = "auto"
    strategy: str = "ddp"

    save_top_k: int = 5
    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 500

    prompt_prefix: str = field(default="")
    prompt_suffix: str = "\n"
    output_dir: str = ""
    ckpt_dir: Optional[str] = None
    save_every_n_epochs: int = 1
    save_every_n_steps: Optional[int] = None
    wandb_project: str = "r3lm"

    def __post_init__(self) -> None:
        if not self.prompt_prefix:
            self.prompt_prefix = _default_prompt_prefix(self.cell_line)
        if not self.output_dir:
            model_tag = self.text_model_name_or_path.split("/")[-1]
            self.output_dir = (
                f"./outputs/{self.cell_line}/"
                f"{self.finetune_mode}-{self.input_mode}-{model_tag}"
            )


def _num_devices(devices: Any) -> int:
    if devices == "auto":
        return torch.cuda.device_count()
    if isinstance(devices, int):
        return devices
    if isinstance(devices, (list, tuple)):
        return len(devices)
    return 1


def _get_target_modules_text(model: nn.Module) -> List[str]:
    target_modules: List[str] = []
    seen = set()
    for _name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            token = _name.split(".")[-1]
            if token != "lm_head" and token not in seen:
                target_modules.append(token)
                seen.add(token)
    for pattern in ["q_proj", "k_proj", "v_proj", "out_proj", "query", "key", "value"]:
        if pattern not in seen:
            target_modules.append(pattern)
    return list(dict.fromkeys(target_modules))


class SequenceRegressionDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        seq_col: str = "model_input",
        y_col: str = "expression_score",
    ):
        self.seqs = df[seq_col].astype(str).tolist()
        self.y = df[y_col].astype(np.float32).to_numpy()

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> Tuple[str, float]:
        return self.seqs[idx], float(self.y[idx])


def _make_collate_fn(tokenizer: AutoTokenizer, cfg: RegressionConfig):
    def collate(batch: List[Tuple[str, float]]):
        seqs, ys = zip(*batch)
        texts = [cfg.prompt_prefix + s + cfg.prompt_suffix for s in seqs]
        toks = tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=cfg.max_length_text,
            return_attention_mask=True,
        )
        padded = tokenizer.pad(
            toks,
            padding=True,
            max_length=cfg.max_length_text,
            return_tensors="pt",
        )
        y = torch.tensor(ys, dtype=torch.float32).unsqueeze(-1)
        return padded, y

    return collate


class ReasonConditionedRegressor(pl.LightningModule):
    def __init__(self, cfg: RegressionConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.text_model_name_or_path,
            cache_dir=cfg.cache_dir,
            trust_remote_code=True,
            local_files_only=cfg.local_files_only,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.text_model = AutoModel.from_pretrained(
            cfg.text_model_name_or_path,
            cache_dir=cfg.cache_dir,
            trust_remote_code=True,
            local_files_only=cfg.local_files_only,
            torch_dtype=torch.bfloat16,
        )

        if cfg.finetune_mode == "freeze":
            for param in self.text_model.parameters():
                param.requires_grad = False
        elif cfg.use_lora:
            lora_cfg = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=_get_target_modules_text(self.text_model),
                init_lora_weights="gaussian",
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.text_model = prepare_model_for_kbit_training(self.text_model)
            self.text_model = get_peft_model(self.text_model, lora_cfg)

        if hasattr(self.text_model, "lm_head"):
            for param in self.text_model.lm_head.parameters():
                param.requires_grad = False
        elif (
            hasattr(self.text_model, "base_model")
            and hasattr(self.text_model.base_model, "model")
            and hasattr(self.text_model.base_model.model, "lm_head")
        ):
            for param in self.text_model.base_model.model.lm_head.parameters():
                param.requires_grad = False

        hidden = self.text_model.config.hidden_size
        self.reg_head = nn.Linear(hidden, 1)
        self.reg_norm = nn.LayerNorm(hidden, eps=1e-5)
        self.loss_fn = nn.MSELoss()
        self.val_preds: List[torch.Tensor] = []
        self.val_targets: List[torch.Tensor] = []

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.text_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_dict=True,
        )
        hs = out.last_hidden_state
        mask = batch["attention_mask"]
        last_idx = mask.sum(dim=1) - 1
        pooled = hs[torch.arange(hs.size(0), device=hs.device), last_idx]
        return self.reg_head(self.reg_norm(pooled))

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = {k: v.to(self.device) for k, v in x.items()}
        y = y.to(self.device)
        pred = self.forward(x)
        loss = self.loss_fn(pred, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.to(self.device)
        x = {k: v.to(self.device) for k, v in x.items()}
        pred = self.forward(x)
        loss = self.loss_fn(pred, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.val_preds.append(pred.detach().float())
        self.val_targets.append(y.detach().float())
        return {"val_loss": loss}

    def on_validation_epoch_start(self) -> None:
        self.val_preds = []
        self.val_targets = []

    def on_validation_epoch_end(self) -> None:
        if not self.val_preds:
            return

        preds = torch.cat(self.val_preds, dim=0).view(-1)
        targets = torch.cat(self.val_targets, dim=0).view(-1)
        mse = torch.mean((preds - targets) ** 2)
        vx = preds - preds.mean()
        vy = targets - targets.mean()
        pearson_rho = (vx * vy).sum() / (
            torch.sqrt((vx**2).sum()) * torch.sqrt((vy**2).sum()) + 1e-8
        )
        r2 = 1 - (mse / (targets.var() + 1e-8))

        self.log("val_mse", mse, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_pearson_rho", pearson_rho, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_r2", r2, prog_bar=True, on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _y = batch
        x = {k: v.to(self.device) for k, v in x.items()}
        return self.forward(x).detach().cpu()

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.cfg.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def _default_train_json(cell_line: str) -> str:
    return f"./data/{cell_line}/82k+1k-with-reason-sharegpt.jsonl"


def _default_test_json(cell_line: str) -> str:
    return f"./data/{cell_line}/100-as-test-sharegpt-with+reason.jsonl"


def _maybe_limit_frame(df: pd.DataFrame, max_samples: Optional[int]) -> pd.DataFrame:
    if max_samples is None or max_samples <= 0:
        return df
    return df.head(max_samples).reset_index(drop=True)


def _resolve_output_dir(args: argparse.Namespace, model_name: str) -> str:
    if args.output_dir:
        return args.output_dir
    return (
        f"./outputs/{args.cell_line}/"
        f"{args.finetune_mode}-{args.input_mode}-{args.run_name}-{model_name}"
    )


def _build_dataloader(
    dataset: Dataset,
    cfg: RegressionConfig,
    collate_fn,
    batch_size: int,
    shuffle: bool,
    sampler=None,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train R3LM Stage-II regressor.")
    parser.add_argument("--cell_line", type=str, default="K562", choices=["K562", "HepG2", "SKNSH"])
    parser.add_argument("--finetune_mode", type=str, default="lora", choices=["lora", "full", "freeze"])
    parser.add_argument(
        "--input_mode",
        type=str,
        default="generated_cot",
        choices=sorted(INPUT_MODES),
        help=(
            "oracle_cot: RCC + gold rationale (upper-bound eval); "
            "generated_cot: RCC + self-generated rationale (deployment); "
            "rcc_only: RCC without rationale"
        ),
    )
    parser.add_argument("--train_json", type=str, default=None)
    parser.add_argument("--val_json", type=str, default=None)
    parser.add_argument("--test_json", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="82k+1k")
    parser.add_argument(
        "--text_model_name_or_path",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
    )
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--train_split_ratio", type=float, default=0.9)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--predict_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="disabled",
        choices=["disabled", "online"],
    )
    parser.add_argument("--wandb_project", type=str, default="r3lm")
    parser.add_argument("--skip_test_after_train", action="store_true")
    return parser.parse_args()


def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = parse_args()

    if args.wandb_mode == "online" and not os.environ.get("WANDB_API_KEY"):
        raise EnvironmentError(
            "WANDB_API_KEY is not set. Export it before enabling --wandb_mode online."
        )

    ensure_data_dir()
    torch.set_float32_matmul_precision("medium")
    model_tag = args.text_model_name_or_path.split("/")[-1]
    output_dir = _resolve_output_dir(args, model_tag)
    cfg = RegressionConfig(
        cell_line=args.cell_line,
        input_mode=args.input_mode,
        finetune_mode=args.finetune_mode,
        use_lora=args.finetune_mode == "lora",
        text_model_name_or_path=args.text_model_name_or_path,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        output_dir=output_dir,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        num_workers=args.num_workers,
        wandb_project=args.wandb_project,
    )
    os.makedirs(cfg.output_dir, exist_ok=True)

    ckpt_dir = cfg.ckpt_dir or os.path.join(cfg.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"[Checkpoint] directory: {os.path.abspath(ckpt_dir)}")

    ckpt_cb_kw = dict(
        dirpath=ckpt_dir,
        filename="{epoch:02d}-{step}-{val_pearson_rho:.4f}",
        monitor="val_pearson_rho",
        mode="max",
        save_top_k=cfg.save_top_k,
        every_n_epochs=cfg.save_every_n_epochs,
        save_last=True,
        verbose=True,
    )
    if cfg.save_every_n_steps is not None:
        ckpt_cb_kw["every_n_train_steps"] = cfg.save_every_n_steps
    ckpt_cb = ModelCheckpoint(**ckpt_cb_kw)

    train_json = args.train_json or _default_train_json(args.cell_line)
    test_json = args.test_json or _default_test_json(args.cell_line)

    if args.val_json:
        train_raw = pd.read_json(train_json, lines=True)
        val_raw = pd.read_json(args.val_json, lines=True)
    else:
        df_all = pd.read_json(train_json, lines=True)
        df_all = df_all.sample(frac=1, random_state=args.split_seed).reset_index(drop=True)
        n_train = int(len(df_all) * args.train_split_ratio)
        train_raw = df_all.iloc[:n_train].reset_index(drop=True)
        val_raw = df_all.iloc[n_train:].reset_index(drop=True)

    train = build_model_frame(train_raw, input_mode=args.input_mode)
    val = build_model_frame(val_raw, input_mode=args.input_mode)
    train = _maybe_limit_frame(train, args.max_train_samples)
    val = _maybe_limit_frame(val, args.max_val_samples)

    if len(train) > 0:
        train.loc[: min(1, len(train) - 1), ["user_transformed", "model_input"]].to_json(
            os.path.join(cfg.output_dir, "train_input_preview.json"),
            orient="records",
            force_ascii=False,
            indent=2,
        )

    train_ds = SequenceRegressionDataset(train)
    val_ds = SequenceRegressionDataset(val)
    model = ReasonConditionedRegressor(cfg)
    collate_fn = _make_collate_fn(model.tokenizer, cfg)

    sampler = None
    shuffle = True
    n_dev = _num_devices(cfg.devices)
    if "label" in train.columns and n_dev == 1:
        vc = train["label"].value_counts()
        sample_weights = train["label"].map(lambda x: 1.0 / vc[x]).astype(np.float32).to_numpy()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False
    elif "label" in train.columns and n_dev > 1:
        print("[Warn] WeightedRandomSampler is disabled under DDP; using shuffle=True.")

    train_dl = _build_dataloader(
        train_ds, cfg, collate_fn, cfg.batch_size, shuffle=shuffle, sampler=sampler
    )
    val_dl = _build_dataloader(val_ds, cfg, collate_fn, args.eval_batch_size, shuffle=False)

    logger = False
    if args.wandb_mode == "online":
        logger = WandbLogger(project=cfg.wandb_project, name=os.path.basename(cfg.output_dir))

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="gpu",
        devices=cfg.devices,
        strategy=cfg.strategy if n_dev > 1 else "auto",
        precision="bf16",
        accumulate_grad_batches=cfg.gradient_accumulation_steps,
        gradient_clip_val=cfg.max_grad_norm,
        gradient_clip_algorithm="norm",
        logger=logger,
        callbacks=[ckpt_cb],
        log_every_n_steps=cfg.logging_steps,
        val_check_interval=cfg.eval_steps,
    )

    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    best_ckpt = (
        trainer.checkpoint_callback.best_model_path
        or trainer.checkpoint_callback.last_model_path
    )
    print("Best checkpoint:", best_ckpt)

    if args.skip_test_after_train:
        return

    test_raw = pd.read_json(test_json, lines=True)
    test = build_model_frame(test_raw, input_mode=args.input_mode)
    test = _maybe_limit_frame(test, args.max_test_samples)
    test_ds = SequenceRegressionDataset(test)
    test_dl = _build_dataloader(test_ds, cfg, collate_fn, args.predict_batch_size, shuffle=False)

    preds = torch.cat(
        trainer.predict(
            model=model,
            dataloaders=test_dl,
            ckpt_path=best_ckpt,
            weights_only=False,
        ),
        dim=0,
    ).float().cpu().numpy()
    test["predicted_expression_score"] = preds.squeeze(-1)

    rho, p_value = pearsonr(test["expression_score"], test["predicted_expression_score"])
    rmse = np.sqrt(
        np.mean((test["expression_score"] - test["predicted_expression_score"]) ** 2)
    )
    print(f"RMSE: {rmse:.4f}")
    print(f"Pearson rho: {rho:.4f} (p={p_value:.2e})")

    pred_path = os.path.join(cfg.output_dir, f"{args.input_mode}_test_predictions.csv")
    test.to_csv(pred_path, index=False)
    print(f"Saved predictions to {pred_path}")


if __name__ == "__main__":
    main()

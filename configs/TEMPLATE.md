# Custom ChatML Template: `chatml_with_n`

Stage-I training uses a ChatML variant that appends a newline before the end-of-sequence token in assistant turns. Register it in your LLaMA-Factory installation before running Stage-I SFT.

## Registration

Add the following block to `src/llamafactory/data/template.py` in your LLaMA-Factory checkout (near the standard `chatml` template). Use the same special tokens as the Qwen3 ChatML template (`IM_START`, `IM_END` below):

```python
IM_START = "<|im_start|>"
IM_END = ""  # Qwen3 end-of-message token

register_template(
    name="chatml_with_n",
    format_user=StringFormatter(
        slots=[f"{IM_START}user\n{{{{content}}}}{IM_END}\n{IM_START}assistant\n"]
    ),
    format_assistant=StringFormatter(slots=["{{content}}\n" + IM_END + "\n"]),
    format_system=StringFormatter(
        slots=[f"{IM_START}system\n{{{{content}}}}{IM_END}\n"]
    ),
    format_observation=StringFormatter(
        slots=[f"{IM_START}tool\n{{{{content}}}}{IM_END}\n{IM_START}assistant\n"]
    ),
    stop_words=[IM_END, IM_START],
    replace_eos=True,
    replace_jinja_template=True,
)
```

## Data directory setup

Copy `configs/dataset_info.json` into the LLaMA-Factory data directory and point `LLAMAFACTORY_DATA_DIR` to the R3LM `data/` folder:

```bash
export LLAMAFACTORY_DATA_DIR=/path/to/R3LM/data
cp configs/dataset_info.json "$LLAMAFACTORY_DATA_DIR/dataset_info.json"
```

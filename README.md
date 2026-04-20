# HiEdit

Official repository for the paper "HiEdit: Lifelong Model Editing with HierarchicalReinforcement Learning".

## Quick Start

First, create a virtual environment and install the necessary dependencies.

```
conda create -n hiedit python==3.10
conda activate hiedit
```

Second, download the LLMs, `meta-llama/Meta-Llama-3-8B-Instruct` and `google/gemma-2-9b`, into the directories `hugging_cache/llama-3` and `hugging_cache/gemma-2`, respectively.

Third, execute the following examples for lifelong model editing using the "2000*1" configuration.

```
#### llama-3,zsre,mend-style ####
CUDA_VISIBLE_DEVICES=0 python main.py \
    dataset=zsre \
    model=llama-3 \
    editor=hiedit \
    num_seq=2000 \
    num_pre=500 \
    glue_step=0 \
    editor.n_layers=6 \
    editor.rank=1920


#### llama-3,zsre,malmen-style ####
CUDA_VISIBLE_DEVICES=0 python main.py \
    dataset=zsre \
    model=llama-3 \
    editor=hieditmm \
    num_seq=2000 \
    num_pre=500 \
    glue_step=0 \
    editor.n_layers=6 \
    editor.rank=1024
```

The parameters including:

- `dataset`: the dataset used for lifelong model editing.
- `model`: the LLMs used for lifelong model editing.
- `editor`: the method used for lifelong model editing. `hiedit` and `hieditmm` denote the MEND-style and MALMEN-style implementations, respectively.
- `num_seq`: the total length of the editing sequence.
- `num_pre`: the number of previously edited instances for evaluation.
- `editor.n_layers`: the number of editing layers per edit.

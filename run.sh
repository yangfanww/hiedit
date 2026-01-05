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
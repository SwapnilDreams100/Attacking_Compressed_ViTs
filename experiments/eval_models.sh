# SOURCE_MODELS='small_quant'

# DATASET='imagenet'

# for sm in $SOURCE_MODELS;
# do
#     python eval_model.py \
#         --source-model $sm \
#         --dataset $DATASET \
#         --batch-size 50 \
#         --device 'cpu' \
#         --subfolder $DATASET \
#         --postfix _${sm}
# done

SOURCE_MODELS='dvit_0.7 
dvit_0.5 
dvit_0.6 
deit_tiny 
deit_small 
deit_base 
distill_tiny 
distill_small 
distill_base 
mini_tiny 
mini_small 
mini_base'
for sm in $SOURCE_MODELS;
do
    python eval_model.py \
        --source-model $sm \
        --dataset 'imagenet' \
        --batch-size 64 \
        --device 'cuda' \
        --subfolder 'imagenet' \
        --postfix _${sm}
done


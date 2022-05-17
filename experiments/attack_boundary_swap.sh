SOURCE_MODELS='deit_small 
               quant'

for sm in $SOURCE_MODELS; do
    CUDA_VISIBLE_DEVICES=-1 python attack_fb_swap.py \
        --source-model ${sm} \
        --target-model quant deit_small  \
        --dataset imagenet \
        --batch-size 64 \
        --attack-variant SpatialAttack \
        --subfolder spatial_quant_swap \
        --postfix _${sm}
done

for sm in $SOURCE_MODELS; do
    CUDA_VISIBLE_DEVICES=-1 python attack_fb_swap.py \
        --source-model ${sm} \
        --target-model quant deit_small  \
        --dataset imagenet \
        --batch-size 64 \
        --attack-variant SaltAndPepperNoiseAttack \
        --subfolder sp_quant_swap \
        --postfix _${sm}
done

for sm in $SOURCE_MODELS; do
    CUDA_VISIBLE_DEVICES=-1 python attack_fb_swap.py \
        --source-model ${sm} \
        --target-model quant deit_small  \
        --dataset imagenet \
        --batch-size 64 \
        --attack-variant LinearSearchBlendedUniformNoiseAttack \
        --subfolder linsearch_quant_swap \
        --postfix _${sm}
done
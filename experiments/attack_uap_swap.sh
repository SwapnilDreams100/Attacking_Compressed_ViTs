SOURCE_MODELS='deit_small'

for sm in $SOURCE_MODELS; do 
    python attack_uap.py \
        --source-model $sm \
        --target-model deit_small distill_small mini_small dvit_0.7 dvit_0.6 dvit_0.5 \
        --dataset imagenet \
        --device 'cuda' \
        --batch-size 64 \
        --attack-iterations 1000 \
        --attack-loss-fn ce-untargeted \
        --attack-lr 0.005 \
        --attack-epsilon 10/255 \
        --subfolder 'uap_res/10_255' \
        --postfix _$sm
done

SOURCE_MODELS='deit_tiny'

for sm in $SOURCE_MODELS; do 
    python attack_uap.py \
        --source-model $sm \
        --target-model deit_tiny distill_tiny mini_tiny \
        --dataset imagenet \
        --device 'cuda' \
        --batch-size 64 \
        --attack-iterations 1000 \
        --attack-loss-fn ce-untargeted \
        --attack-lr 0.005 \
        --attack-epsilon 10/255 \
        --subfolder 'uap_res/10_255' \
        --postfix _$sm
done

SOURCE_MODELS='deit_base'

for sm in $SOURCE_MODELS; do 
    python attack_uap.py \
        --source-model $sm \
        --target-model deit_base distill_base mini_base \
        --dataset imagenet \
        --device 'cuda' \
        --batch-size 64 \
        --attack-iterations 1000 \
        --attack-loss-fn ce-untargeted \
        --attack-lr 0.005 \
        --attack-epsilon 10/255 \
        --subfolder 'uap_res/10_255' \
        --postfix _$sm
done

SOURCE_MODELS='mini_tiny'

for sm in $SOURCE_MODELS; do 
    python attack_uap.py \
        --source-model $sm \
        --target-model deit_tiny mini_tiny \
        --dataset imagenet \
        --device 'cuda' \
        --batch-size 64 \
        --attack-iterations 1000 \
        --attack-loss-fn ce-untargeted \
        --attack-lr 0.005 \
        --attack-epsilon 10/255 \
        --subfolder 'uap_res/10_255' \
        --postfix _$sm
done

SOURCE_MODELS='mini_small'

for sm in $SOURCE_MODELS; do 
    python attack_uap.py \
        --source-model $sm \
        --target-model deit_small mini_small \
        --dataset imagenet \
        --device 'cuda' \
        --batch-size 64 \
        --attack-iterations 1000 \
        --attack-loss-fn ce-untargeted \
        --attack-lr 0.005 \
        --attack-epsilon 10/255 \
        --subfolder 'uap_res/10_255' \
        --postfix _$sm
done

SOURCE_MODELS='mini_base'

for sm in $SOURCE_MODELS; do 
    python attack_uap.py \
        --source-model $sm \
        --target-model deit_base mini_base \
        --dataset imagenet \
        --device 'cuda' \
        --batch-size 64 \
        --attack-iterations 1000 \
        --attack-loss-fn ce-untargeted \
        --attack-lr 0.005 \
        --attack-epsilon 10/255 \
        --subfolder 'uap_res/10_255' \
        --postfix _$sm
done

SOURCE_MODELS='distill_tiny'

for sm in $SOURCE_MODELS; do 
    python attack_uap.py \
        --source-model $sm \
        --target-model deit_tiny distill_tiny \
        --dataset imagenet \
        --device 'cuda' \
        --batch-size 64 \
        --attack-iterations 1000 \
        --attack-loss-fn ce-untargeted \
        --attack-lr 0.005 \
        --attack-epsilon 10/255 \
        --subfolder 'uap_res/10_255' \
        --postfix _$sm
done

SOURCE_MODELS='distill_small'

for sm in $SOURCE_MODELS; do 
    python attack_uap.py \
        --source-model $sm \
        --target-model deit_small distill_small \
        --dataset imagenet \
        --device 'cuda' \
        --batch-size 64 \
        --attack-iterations 1000 \
        --attack-loss-fn ce-untargeted \
        --attack-lr 0.005 \
        --attack-epsilon 10/255 \
        --subfolder 'uap_res/10_255' \
        --postfix _$sm
done

SOURCE_MODELS='distill_base'

for sm in $SOURCE_MODELS; do 
    python attack_uap.py \
        --source-model $sm \
        --target-model deit_base distill_base \
        --dataset imagenet \
        --device 'cuda' \
        --batch-size 64 \
        --attack-iterations 1000 \
        --attack-loss-fn ce-untargeted \
        --attack-lr 0.005 \
        --attack-epsilon 10/255 \
        --subfolder 'uap_res/10_255' \
        --postfix _$sm
done

# SOURCE_MODELS='small_quant 
# small_deit'

# for sm in $SOURCE_MODELS; do 
#     python attack_uap.py \
#         --source-model $sm \
#         --target-model small_deit small_quant \
#         --dataset imagenet \
#         --device 'cpu' \
#         --batch-size 64 \
#         --attack-iterations 1000 \
#         --attack-loss-fn ce-untargeted \
#         --attack-lr 0.005 \
#         --attack-epsilon 10/255 \
#         --subfolder 'uap_res/10_255' \
#         --postfix _$sm
# done
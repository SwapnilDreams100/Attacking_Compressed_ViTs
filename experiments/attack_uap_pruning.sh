SOURCE_MODELS='dvit_0.7'

for sm in $SOURCE_MODELS; do 
    python attack_uap.py \
        --source-model $sm \
        --target-model deit_small dvit_0.7 \
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

SOURCE_MODELS='dvit_0.6'

for sm in $SOURCE_MODELS; do 
    python attack_uap.py \
        --source-model $sm \
        --target-model deit_small dvit_0.6 \
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

SOURCE_MODELS='dvit_0.5'

for sm in $SOURCE_MODELS; do 
    python attack_uap.py \
        --source-model $sm \
        --target-model deit_small dvit_0.5 \
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
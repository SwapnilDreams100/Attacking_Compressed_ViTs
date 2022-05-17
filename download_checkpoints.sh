mkdir -p ./checkpoints

wget https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-B_16-224.npz -O ./checkpoints/ViT-B_16-224.npz
wget https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth -O ./checkpoints/deit_tiny_patch16_224.pth
wget https://github.com/DominickZhang/MiniViT-model-zoo/releases/download/v1.0.0/mini_deit_tiny_patch16_224.pth -O ./checkpoints/mini_deit_tiny_patch16_224.pth
# wget https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-B_16-224.npz -O ./checkpoints/ViT-B_16-224.npz
# wget https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-L_16-224.npz -O ./checkpoints/ViT-L_16-224.npz
# wget https://storage.googleapis.com/vit_models/augreg/B_16-i1k-300ep-lr_0.001-aug_medium2-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz -O ./checkpoints/ViT-augreg_B_16-i1k-300ep-lr_0.001-aug_medium2-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res-224.npz
# wget https://storage.googleapis.com/vit_models/augreg/L_16-i1k-300ep-lr_0.001-aug_medium2-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz -O ./checkpoints/ViT-augreg_L_16-i1k-300ep-lr_0.001-aug_medium2-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res-224.npz
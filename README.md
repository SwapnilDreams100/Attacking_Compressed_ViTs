# Compressing Adversarial Attacks

## Downloading Models:

Please run `download_checkpoint.sh` to download the models and use `environment.yml` to setup the environment using conda

Download models from drive link `https://drive.google.com/drive/folders/1ZJNrhe6IW82tl4VD0X84Scs4ocETL8-1?usp=sharing` 
and store them in `/dvit/pretrained` and `checkpoints` respectively

## Running Experiments 

`final_runner.ipynb`: is all you need to run all the experiments
Description of its commands:

`!bash ./experiments/eval_models.sh` - evaluate all the models using `eval_models.py`

`!bash ./experiments/attack_uap_swap.sh` - create attacks on base and distilled models using `attack_uap.py`

`!bash ./experiments/attack_uap_pruning.sh` - create attacks on pruned models using `attack_uap.py`

`!bash ./experiments/attack_boundary_swap.sh` - create black box attacks on quantized using `attack_fb_swap.py`

`utils.py` - loads all the models

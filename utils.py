import os, csv, logging
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from functools import partial
import timm
        
def load_ground_truth(csv_filename):
    image_id_list = []
    label_ori_list = []
    label_tar_list = []

    with open(csv_filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            image_id_list.append( row['ImageId'] )
            label_ori_list.append( int(row['TrueLabel'])-1 )
            label_tar_list.append( int(row['TargetClass'])-1 )

    return image_id_list,label_ori_list,label_tar_list

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.model = model
    
    def forward(self, x):
        lo = self.model.forward(x)
        if isinstance(lo, (tuple, list)):
           lo = lo[0]
        return lo

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()

        self.mean = torch.Tensor(mean).reshape(1,3,1,1)
        self.std = torch.Tensor(std).reshape(1,3,1,1)

    def forward(self, x):
        return (x - self.mean.type_as(x)) / self.std.type_as(x)

class Unnormalize(nn.Module):
    def __init__(self, mean, std):
        super(Unnormalize, self).__init__()

        self.mean = torch.Tensor(mean).reshape(1,3,1,1)
        self.std = torch.Tensor(std).reshape(1,3,1,1)

    def forward(self, x):
        return (x * self.std.type_as(x)) + self.mean.type_as(x)

def get_logger(path, filename='log.txt'):
    logger = logging.getLogger('logbuch')
    logger.setLevel(level=logging.DEBUG)
    
    # Stream handler
    sh = logging.StreamHandler()
    sh.setLevel(level=logging.DEBUG)
    sh_formatter = logging.Formatter('%(message)s')
    sh.setFormatter(sh_formatter)
    
    # File handler
    fh = logging.FileHandler(os.path.join(path, filename))
    fh.setLevel(level=logging.DEBUG)
    fh_formatter = logging.Formatter('%(message)s')
    fh.setFormatter(fh_formatter)

    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger

def get_timestamp():
    ISOTIMEFORMAT='%Y%m%d_%H%M%S_%f'
    timestamp = '{}'.format(datetime.utcnow().strftime( ISOTIMEFORMAT)[:-3])
    return timestamp

def one_hot(class_labels, num_classes):
    class_labels = class_labels.cpu()
    return torch.zeros(len(class_labels), num_classes).scatter_(1, class_labels.unsqueeze(1), 1.).cuda()

def get_model(model_name):
    if model_name.startswith('deit'):
        if 'tiny' in model_name:
            model = torch.load('./checkpoints/deit_tiny_class')
            model.load_state_dict(torch.load('./checkpoints/deit_tiny_patch16_224.pth')['model'])
        elif 'small' in model_name:
            model = torch.load('./checkpoints/deit_small_class')
            model.load_state_dict(torch.load('./checkpoints/deit_small_patch16_224.pth')['model'])
        elif 'base' in model_name:
            model = torch.load('./checkpoints/deit_base_class')
            model.load_state_dict(torch.load('./checkpoints/deit_base_patch16_224.pth')['model'])
    
    elif model_name.startswith('distill'):
        from deit.models import DistilledVisionTransformer

        if 'tiny' in model_name:
            model = DistilledVisionTransformer(
                        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6))
            model.load_state_dict(torch.load('./checkpoints/deit_tiny_distilled_patch16_224.pth')['model'])
        elif 'small' in model_name:
            model = DistilledVisionTransformer(
                        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6))
            model.load_state_dict(torch.load('./checkpoints/deit_small_distilled_patch16_224.pth')['model'])
        elif 'base' in model_name:
            model = DistilledVisionTransformer(
                        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6))
            model.load_state_dict(torch.load('./checkpoints/deit_base_distilled_patch16_224.pth')['model'])
            
    elif model_name.startswith('mini'):
        import mini_deit.mini_deit_models

        if 'tiny' in model_name:    
            model = timm.create_model( 'mini_deit_tiny_patch16_224', pretrained=False, num_classes=1000, drop_rate=0.0, drop_path_rate=0.1, drop_block_rate=None )
            model.load_state_dict(torch.load('./checkpoints/mini_deit_tiny_patch16_224.pth')['model'])
        elif 'small' in model_name:
            model = timm.create_model( 'mini_deit_small_patch16_224', pretrained=False, num_classes=1000, drop_rate=0.0, drop_path_rate=0.1, drop_block_rate=None )
            model.load_state_dict(torch.load('./checkpoints/mini_deit_small_patch16_224.pth')['model'])
        elif 'base' in model_name:
            model = timm.create_model( 'mini_deit_base_patch16_224', pretrained=False, num_classes=1000, drop_rate=0.0, drop_path_rate=0.1, drop_block_rate=None )
            model.load_state_dict(torch.load('./checkpoints/mini_deit_base_patch16_224.pth')['model'])
      
    elif model_name=='quant':
        model = torch.load('./checkpoints/deit_small_class')
        model.load_state_dict(torch.load('./checkpoints/deit_small_patch16_224.pth')['model'])
        model.eval()
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
     
    elif model_name.startswith('dvit'):
        from dvit.vit_extern import VisionTransformerDiffPruning

        if '0.7' in model_name:
            base_rate = 0.7
            KEEP_RATE = [base_rate, base_rate ** 2, base_rate ** 3]
            PRUNING_LOC = [3,6,9]
            CKPT_PATH = 'dvit/pretrained/dynamic-vit_384_r0.7.pth'
            model = VisionTransformerDiffPruning(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE)
            checkpoint = torch.load(CKPT_PATH, map_location='cpu')['model']
            model.load_state_dict(checkpoint)
        elif '0.6' in model_name:
            base_rate = 0.6
            KEEP_RATE = [base_rate, base_rate ** 2, base_rate ** 3]
            PRUNING_LOC = [3,6,9]
            CKPT_PATH = 'dvit/pretrained/dynamic-vit_384_r0.6.pth'
            model = VisionTransformerDiffPruning(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE)
            checkpoint = torch.load(CKPT_PATH, map_location='cpu')['model']
            model.load_state_dict(checkpoint)
        elif '0.5' in model_name:
            base_rate = 0.5
            KEEP_RATE = [base_rate, base_rate ** 2, base_rate ** 3]
            PRUNING_LOC = [3,6,9]
            CKPT_PATH = 'dvit/pretrained/dynamic-vit_384_r0.5.pth'
            model = VisionTransformerDiffPruning(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE)
            checkpoint = torch.load(CKPT_PATH, map_location='cpu')['model']
            model.load_state_dict(checkpoint)
    model.eval()
    return model
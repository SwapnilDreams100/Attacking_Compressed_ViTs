import argparse, os, sys
import shutil
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.utils
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import foolbox as fb
import eagerpy as ep
from utils import Normalize, Unnormalize, get_timestamp, load_ground_truth, get_model
from utils import get_logger as get_my_logger
from config import IMAGENET_PATH, NEURIPS_DATA_PATH, NEURIPS_CSV_PATH

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Adversarial Attack with Foolbox')
# Dataset / Model parameters
parser.add_argument('--source-model', type=str, default='resnet101', help='Source model')
parser.add_argument('--target-model', nargs="+", default=['resnet101'], help='Target model')         
parser.add_argument('--dataset', default='imagenet', type=str, help='Used Dataset')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size')

# Attack
parser.add_argument('--attack-variant', default='cw_l2', type=str, help='Attack variant')

# # CW Attack
# parser.add_argument('--cw-binary-search-steps', default=9, type=int, help='Binary search steps')
# parser.add_argument('--cw-steps', default=10000, type=int, help='Steps')
# parser.add_argument('--cw-stepsize', default=0.01, type=float, help='Step size')
# parser.add_argument('--cw-confidence', default=0., type=float, help='Confidence')
# parser.add_argument('--cw-initial-constant', default=0.001, type=float, help='Initial constant')
# parser.add_argument('--cw-abort-early', type=eval, default=True, choices=[True,False], help='Abort early')

# # DeepFool Attack
# parser.add_argument('--deepfool-steps', default=50, type=int, help='Binary search steps')
# parser.add_argument('--deepfool-candidates', default=10, type=int, help='Candidates')
# parser.add_argument('--deepfool-overshoot', default=0.02, type=float, help='Overshoot')
# parser.add_argument('--deepfool-loss', default='logits', type=str, help='Loss')

# # Boundary Attack 
# parser.add_argument('--boundary-steps', default=50, type=int, help='Boundary attack steps')
# parser.add_argument('--boundary-spherical-step', default=0.01, type=float, help='Boundary attack steps')
# parser.add_argument('--boundary-source-step', default=0.01, type=float, help='Boundary attack steps')
# parser.add_argument('--boundary-source-step-convergence', default=1e-07, type=float, help='Boundary attack steps')
# parser.add_argument('--boundary-step-adaptation', default=1.5, type=float, help='Boundary attack steps')

# Misc
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--workers', type=int, default=8, help='Dataloader workers')
parser.add_argument('--subfolder', default='', type=str, help='Subfolder to store the results in')
parser.add_argument('--postfix', type=str, default='', help='Postfix to append to results folder')

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.model = model
    
    def forward(self, x):
        lo = self.model.forward(x)
        if isinstance(lo, (tuple, list)):
            lo = lo[0]
        return lo

def _parse_args():
    args = parser.parse_args()
    return args

def main():
    args = _parse_args()

    args.distributed = False

    torch.manual_seed(args.seed)

    result_path = os.path.join('./output', 'attack', args.subfolder, get_timestamp() + args.postfix)
    os.makedirs(result_path)

    # Saving this file
    shutil.copy(sys.argv[0], os.path.join(result_path, sys.argv[0]))
    _logger = get_my_logger(result_path)

    state = {k: v for k, v in args._get_kwargs()}
    for key, value in state.items():
        _logger.info('{} : {}'.format(key, value))
    
    if args.dataset == 'imagenet':
        num_classes = 1000
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transform_eval = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                ])

        dir_eval = os.path.join(IMAGENET_PATH, 'val')
        data_eval = torchvision.datasets.ImageFolder(root=dir_eval, transform=transform_eval)

        loader_eval = torch.utils.data.DataLoader(data_eval,
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    num_workers=args.workers,
                                                    pin_memory=True)
    
    source_model = ModelWrapper(get_model(args.source_model)).eval()
    source_model.cpu()
    
    num_target_models = len(args.target_model)
    target_model = []
    for tm in args.target_model:
        model=get_model(tm)
        model.cpu()
        target_model.append(model) 

    norm = Normalize(mean=mean, std=std)
    norm_vit = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    unnorm = Unnormalize(mean=mean, std=std)
    
    # Store results
    acc_untargeted=np.zeros((num_target_models, 0))
    acc_targeted=np.zeros((num_target_models, 0))
    norm_l2 = np.zeros((0))
    norm_linf = np.zeros((0))


    ################# ATTACK ######################
    preprocessing = dict(mean=mean, std=std, axis=-3)
    
    fmodel = fb.PyTorchModel(source_model, bounds=(0, 1), preprocessing=preprocessing)
    if args.attack_variant == 'SpatialAttack':
        attack = fb.attacks.SpatialAttack(grid_search = False, random_steps = 1000)
        epsilons = [None]
    elif args.attack_variant == 'GaussianBlur':
        attack = fb.attacks.GaussianBlurAttack(steps = 100)
        epsilons = [None]
    elif args.attack_variant == 'L2ContrastReductionAttack':
        attack = fb.attacks.L2ContrastReductionAttack()
        epsilons = [0.5]
    elif args.attack_variant == 'L2AdditiveGaussianNoiseAttack':  
        attack = fb.attacks.L2AdditiveGaussianNoiseAttack()
        epsilons = [0.5]
    elif args.attack_variant == 'SaltAndPepperNoiseAttack':  
        attack = fb.attacks.SaltAndPepperNoiseAttack(steps= 10)
        epsilons = [None]
    elif args.attack_variant == 'LinearSearchBlendedUniformNoiseAttack':  
        attack = fb.attacks.LinearSearchBlendedUniformNoiseAttack(steps= 10)
        epsilons = [None]
    else:
        raise ValueError
    
    for batch_idx, (x, lbl) in enumerate(loader_eval):
        if batch_idx>20:
            break
        print("Attack: {}/{}".format(batch_idx+1, len(loader_eval)))

        if args.dataset == 'neurips':
            y_gt = lbl[0]
            y_tar = lbl[1]
        elif args.dataset == 'imagenet':
            y_gt = lbl 
            rnd = torch.randint(1, num_classes,(len(lbl),))
            y_tar = (y_gt+rnd) % num_classes
        
        x_unnorm = unnorm(x.cpu())
        y_gt = y_gt.cpu()
        y_tar = y_tar.cpu()
        images, labels = ep.astensors(x_unnorm, y_gt)
        
        ####### Attack #######
        _, clipped_advs, success = attack(fmodel, images, labels, epsilons=epsilons)
        ##################
        # Back to native format eagerpy -> tensor with .raw
#         print(len(clipped_advs))
        if type(clipped_advs) == list:
            x_adv = clipped_advs[0].raw
        else:
            x_adv = clipped_advs.raw

        # Store stats
        corr_cl = np.zeros((num_target_models, x.size(0)), dtype=np.bool)
        corr_tar = np.zeros((num_target_models, x.size(0)), dtype=np.bool)
        for tm_idx, (tm, tm_name) in enumerate(zip(target_model, args.target_model)):
#             if 'ViT' in tm_name:
            lo = tm(norm_vit(x_adv))
#             elif 'mixer_' in tm_name:
#                 lo = tm(norm_vit(x_adv))            
#             else:
#                 lo = tm(norm(x_adv))
            if isinstance(lo, (tuple, list)):
                lo = lo[0]
            pred = torch.argmax(lo, dim=-1)
#             print(pred, y_gt)
            # Get the number of correctly classified samples
            corr_cl[tm_idx] = (pred == y_gt).cpu().numpy()
#             print(corr_cl[tm_idx] )
            # Get the number of correctly targeted samples
            corr_tar[tm_idx] = (pred == y_tar).cpu().numpy()
        # Concat
        acc_untargeted = np.concatenate((acc_untargeted, corr_cl), axis=1)
        acc_targeted = np.concatenate((acc_targeted, corr_tar), axis=1)

        delta = x_adv - x_unnorm
        # Calc l2 norm of delta
        l2 = torch.norm(delta.reshape(delta.size(0), -1), p=2, dim=1).detach().cpu().numpy()
        norm_l2 = np.concatenate((norm_l2, l2))
        # Calc linf norm of delta
        linf = torch.norm(delta.reshape(delta.size(0), -1), p=np.inf, dim=1).detach().cpu().numpy()
        norm_linf = np.concatenate((norm_linf, linf))

        # Results
        _logger.info('\n-- Untargeted ASR --')
        for tm_idx, tm in enumerate(args.target_model):
            _logger.info('{} -> {}'.format(args.source_model, tm))
            _logger.info('{}'.format(1. - np.mean(acc_untargeted[tm_idx])))

        _logger.info('\n-- Targeted Accuracy --')
        for tm_idx, tm in enumerate(args.target_model):
            _logger.info('{} -> {}'.format(args.source_model, tm))
            _logger.info('{}'.format(np.mean(acc_targeted[tm_idx])))
        
        _logger.info('\n-- L2 norm --')
        _logger.info('{}'.format(np.mean(norm_l2)))

        _logger.info('\n-- Linf norm --')
        _logger.info('{}'.format(np.mean(norm_linf)))

    # Results
    _logger.info('\n-- Final Evaluation --')
    _logger.info('\n-- Untargeted ASR --')
    for tm_idx, tm in enumerate(args.target_model):
        _logger.info('{} -> {}'.format(args.source_model, tm))
        _logger.info('{}'.format(1. - np.mean(acc_untargeted[tm_idx])))

    _logger.info('\n-- Targeted Accuracy --')
    for tm_idx, tm in enumerate(args.target_model):
        _logger.info('{} -> {}'.format(args.source_model, tm))
        _logger.info('{}'.format(np.mean(acc_targeted[tm_idx])))
    
    _logger.info('\n-- L2 norm --')
    _logger.info('{}'.format(np.mean(norm_l2)))

    _logger.info('\n-- Linf norm --')
    _logger.info('{}'.format(np.mean(norm_linf)))

    model_string=''
    untargeted_string=''
    targeted_string=''
    untar_tar_string=''
    for tm_idx, tm in enumerate(args.target_model):     
        model_string += tm + ' '
        untargeted_string += '{} '.format(1. - np.mean(acc_untargeted[tm_idx]))
        targeted_string += '{} '.format(np.mean(acc_targeted[tm_idx]))
        untar_tar_string += '{}/{} '.format(1. - np.mean(acc_untargeted[tm_idx]), np.mean(acc_targeted[tm_idx]))

    _logger.info('{}'.format(model_string))
    _logger.info('-- Untargeted ASR --')
    _logger.info(untargeted_string)
    _logger.info('-- Targeted Accuracy --')
    _logger.info(targeted_string)
    _logger.info('-- Untargeted ASR / Targeted Accuracy --')
    _logger.info(untar_tar_string)

if __name__ == '__main__':
    main()

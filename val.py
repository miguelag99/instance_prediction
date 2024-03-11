import os
import socket
import time
import argparse
import numpy as np
import random

import lightning as L
import torch
import yaml

from prediction.data.prepare_loader import prepare_dataloaders
from prediction.configs import baseline_cfg
from prediction.config import namespace_to_dict
from prediction.trainer import TrainingModule

def main(args):
    
    # Load training config
    if args.config == 'baseline':
        cfg = baseline_cfg

    hparams = namespace_to_dict(cfg)
        
    # Set random seed for reproducibility
    seed = 42
    L.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    _ , valloader = prepare_dataloaders(cfg)
    
    ckpt_cfg = cfg.PRETRAINED
    # assert os.isdir
    assert ckpt_cfg.LOAD_WEIGHTS and ckpt_cfg.PATH != '',\
        'No weights to load for evaluation'
    assert os.path.exists(ckpt_cfg.PATH),\
        f'Path {ckpt_cfg.PATH} does not exist'
    assert os.path.exists(os.path.join(ckpt_cfg.PATH,ckpt_cfg.CKPT)),\
        f'Checkpoint {ckpt_cfg.CKPT} does not exist'
    
    save_dir = ckpt_cfg.PATH
            
        
    l_module = TrainingModule.load_from_checkpoint(os.path.join(ckpt_cfg.PATH,ckpt_cfg.CKPT))
    l_module.eval()

    trainer = L.Trainer(
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICES,
        precision=cfg.PRECISION,
        sync_batchnorm=True,
        gradient_clip_val=cfg.GRAD_NORM_CLIP,
        max_epochs=cfg.EPOCHS,
        log_every_n_steps=cfg.LOGGING_INTERVAL,
    )
    
    metrics = trainer.validate(l_module, valloader)
    
    # Save dict to yaml
    with open(os.path.join(save_dir,'val_metrics.yaml'), 'w') as file:
        documents = yaml.dump(metrics[0], file)
        
    # Free memory
    del l_module
    del trainer
    del valloader
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Create parser with one argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='baseline')
    args = parser.parse_args()

    main(args)
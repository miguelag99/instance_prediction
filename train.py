import os
import socket
import time
import argparse
import numpy as np
import random

import lightning as L
import torch

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary, LearningRateMonitor
# from lightning.pytorch.strategies import DDPStrategy

from prediction.data.prepare_loader import prepare_dataloaders
from prediction.configs import baseline_cfg
from prediction.config import namespace_to_dict
from prediction.trainer import TrainingModule

import gc 
gc.collect()

def main(args):

    # Load training config
    if args.config == 'baseline':
        cfg = baseline_cfg

    hparams = namespace_to_dict(cfg)
    save_dir = os.path.join(
        cfg.LOG_DIR, time.strftime('%d%B%Yat%H:%M:%S%Z') + '_' + socket.gethostname() + '_' + cfg.TAG
    ) 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(os.path.join(save_dir, 'checkpoints'))

    # Set random seed for reproducibility
    seed = 42
    L.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    trainloader, valloader = prepare_dataloaders(cfg)

    l_module = TrainingModule(hparams, cfg)

    if cfg.PRETRAINED.RESUME_TRAINING:
        # TODO: Load training state from checkpoint
        pass

    if cfg.PRETRAINED.LOAD_WEIGHTS:
        # Load single-image instance segmentation model.
        pretrained_model_weights = torch.load(
            cfg.PRETRAINED.PATH , map_location='cpu'
        )['state_dict']

        l_module.load_state_dict(pretrained_model_weights, strict=False)
        print(f'Loaded single-image model weights from {cfg.PRETRAINED.PATH}')
  

    wdb_logger = WandbLogger(project=cfg.WANDB_PROJECT,save_dir=save_dir,
                             log_model=False, name=cfg.TAG)
    chkpt_callback = ModelCheckpoint(dirpath=os.path.join(save_dir,'checkpoints'),
                                     monitor='vpq',
                                     save_top_k=3,
                                     mode='max',
                                     filename='model-{epoch}-{vpq:.4f}')

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = L.Trainer(
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICES,
        precision=cfg.PRECISION,
        sync_batchnorm=True,
        gradient_clip_val=cfg.GRAD_NORM_CLIP,
        max_epochs=cfg.EPOCHS,
        logger=wdb_logger,
        log_every_n_steps=cfg.LOGGING_INTERVAL,
        callbacks=[chkpt_callback, lr_monitor],
        profiler='simple',
    )

    trainer.fit(l_module, trainloader, valloader)

    # Free memory
    del l_module
    del trainer
    del trainloader
    del valloader
    torch.cuda.empty_cache()



if __name__ == "__main__":
    # Create parser with one argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='baseline')
    args = parser.parse_args()



    main(args)
import torch

from nuscenes.nuscenes import NuScenes
from prediction.data.nuscenes_dataset import NuscenesDataset
from prediction.data.powerbev_loader import FuturePredictionDataset

def prepare_dataloaders(cfg, return_dataset=False):
    """
    Prepare the dataloader of PowerBEV.
    """
    version = cfg.DATASET.VERSION
    train_on_training_data = True

    """ 
    if cfg.DATASET.NAME == 'nuscenes':
        # 28130 train and 6019 val
        # dataroot = os.path.join(cfg.DATASET.DATAROOT, version)
        dataroot = cfg.DATASET.DATAROOT
        nusc = NuScenes(version=cfg.DATASET.VERSION, dataroot=dataroot, verbose=True)
    train_data = FuturePredictionDataset(nusc, train_on_training_data, cfg)
    val_data = FuturePredictionDataset(nusc, False, cfg)
     """ 

    # TODO: pass nuscenes object to each loader instead of creating it inside? 
    train_data = NuscenesDataset(cfg, mode = 'train', return_orig_images=False)
    val_data = NuscenesDataset(cfg, mode = 'val', return_orig_images=False)

    if cfg.DATASET.VERSION == 'mini':
        train_data.indices = train_data.indices[:10]
        val_data.indices = val_data.indices[:10]
     
    nworkers = cfg.N_WORKERS
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=cfg.BATCHSIZE, shuffle=True,
        num_workers=nworkers, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=cfg.BATCHSIZE, shuffle=False,
        num_workers=nworkers, pin_memory=True, drop_last=True)

    if return_dataset:
        return train_loader, val_loader, train_data, val_data
    else:
        return train_loader, val_loader
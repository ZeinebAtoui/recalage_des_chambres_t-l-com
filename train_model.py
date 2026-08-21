import os
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.utils import register_all_modules

# 0. Enregistrement des modules MMSegmentation
register_all_modules()

# --- 1. CONFIGURATION DES CLASSES (Ordre labelmap.txt) ---
classes = (
    'background', 'chambre_A1', 'chambre_A2', 'chambre_A3', 'chambre_A4', 'chambre_A5',
    'chambre_B1', 'chambre_B2', 'chambre_C1', 'chambre_D2', 'chambre_K1C', 'chambre_K2C',
    'chambre_K3C', 'chambre_L0T', 'chambre_L1C', 'chambre_L1T', 'chambre_L2C', 'chambre_L2T',
    'chambre_L3C', 'chambre_L3T', 'chambre_L4C', 'chambre_L4T', 'chambre_L5T', 'chambre_L6T',
    'chambre_M1C', 'chambre_M1T', 'chambre_M2C', 'chambre_M2T', 'chambre_M3T', 'chambre_OHN',
    'chambre_P1C', 'chambre_P1T', 'chambre_P2T', 'chambre_REG', 'hard_negative', 'obstacle'
)

# --- 2. CHARGEMENT DE LA CONFIGURATION DE BASE ---
# On utilise la config compatible avec le modèle v1.0
cfg = Config.fromfile('mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb2-80k_cityscapes-512x1024.py')

# --- 3. ADAPTATION DU MODÈLE (36 CLASSES) ---
cfg.model.decode_head.num_classes = len(classes)
cfg.model.auxiliary_head.num_classes = len(classes)

# Forcer la normalisation classique pour CPU (évite les erreurs SyncBN)
cfg.norm_cfg = dict(type='BN', requires_grad=True)
cfg.model.backbone.norm_cfg = cfg.norm_cfg
cfg.model.decode_head.norm_cfg = cfg.norm_cfg
cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg

# --- 4. CONFIGURATION DES DONNÉES ---
cfg.data_root = 'dataset_telecom/'

# Train Dataloader
cfg.train_dataloader.dataset.type = 'BaseSegDataset'
cfg.train_dataloader.dataset.data_root = cfg.data_root
cfg.train_dataloader.dataset.metainfo = dict(classes=classes)
cfg.train_dataloader.dataset.data_prefix = dict(img_path='train/images', seg_map_path='train/masks')
cfg.train_dataloader.num_workers = 0
cfg.train_dataloader.persistent_workers = False
cfg.train_dataloader.batch_size = 1

# Val Dataloader
cfg.val_dataloader.dataset.type = 'BaseSegDataset'
cfg.val_dataloader.dataset.data_root = cfg.data_root
cfg.val_dataloader.dataset.metainfo = dict(classes=classes)
cfg.val_dataloader.dataset.data_prefix = dict(img_path='test/images', seg_map_path='test/masks')
cfg.val_dataloader.num_workers = 0
cfg.val_dataloader.persistent_workers = False

cfg.test_dataloader = cfg.val_dataloader

# --- 5. RÉGLAGES RUNNER (Optimisé CPU) ---
cfg.device = 'cpu'
cfg.train_cfg = dict(type='IterBasedTrainLoop', max_iters=1000, val_interval=500)
cfg.default_hooks.checkpoint.interval = 500
cfg.default_hooks.logger.interval = 10
cfg.work_dir = './work_dirs/telecom_model'

# --- 6. NOUVELLE URL DE TRANSFER LEARNING (VÉRIFIÉE) ---
cfg.load_from = 'https://download.openmmlab.com/mmsegmentation/v1.0/deeplabv3plus/deeplabv3plus_r50-d8_4xb2-80k_cityscapes-512x1024/deeplabv3plus_r50-d8_4xb2-80k_cityscapes-512x1024_20221208_213106-bc30f7a0.pth'

# --- 7. LANCEMENT ---
if __name__ == '__main__':
    print(" Lancement du moteur d'entraînement sur CPU...")
    print(f"Modèle pré-entraîné : {cfg.load_from.split('/')[-1]}")
    runner = Runner.from_cfg(cfg)
    runner.train()
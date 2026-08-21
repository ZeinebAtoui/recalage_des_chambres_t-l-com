

_base_ = [
     './mmsegmentation/configs/_base_/models/deeplabv3plus_r50-d8.py',
    './mmsegmentation/configs/_base_/default_runtime.py'
]


# ==================================================
# Classes
# ==================================================

classes = (
    'background',
    'hard_negative',
    'chambre_telecom'
)


metainfo = dict(
    classes=classes
)



load_from =None 



# ==================================================
# Data Preprocessor
# ==================================================

crop_size = (512,512)


data_preprocessor = dict(

    type='SegDataPreProcessor',

    size=crop_size,

    mean=[123.675,116.28,103.53],

    std=[58.395,57.12,57.375],

    bgr_to_rgb=True,

    seg_pad_val=255
)



# ==================================================
# Normalisation
# ==================================================

norm_cfg = dict(

    type='SyncBN',

    requires_grad=True
)



# ==================================================
# Model
# ==================================================

model = dict(

    data_preprocessor=data_preprocessor,

    backbone=dict(

        norm_cfg=norm_cfg,

        norm_eval=False

    ),

    decode_head=dict(

        num_classes=3,

        norm_cfg=norm_cfg,

        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
           
            loss_weight=1.0
        )

    ),

    auxiliary_head=dict(

        num_classes=3,

        norm_cfg=norm_cfg,

        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            
            loss_weight=0.4
        )

    )

)


    



# ==================================================
# Dataset
# ==================================================

dataset_type = 'BaseSegDataset'


data_root = (
    '/kaggle/working/dataset_crops_v4'
)



# ==================================================
# Pipelines
# ==================================================

train_pipeline = [

    dict(
        type='LoadImageFromFile'
    ),


    dict(
        type='LoadAnnotations'
    ),


    dict(
        type='RandomResize',

        scale=(512,512),

        ratio_range=(0.5,2.0),

        keep_ratio=True
    ),


    dict(
        type='RandomCrop',

        crop_size=crop_size,

        cat_max_ratio=0.75
    ),


    dict(
        type='RandomFlip',

        prob=0.5
    ),


    dict(
        type='PackSegInputs'
    )

]



test_pipeline = [

    dict(
        type='LoadImageFromFile'
    ),


    dict(
        type='Resize',

        scale=(512,512),

        keep_ratio=True
    ),


    dict(
        type='LoadAnnotations'
    ),


    dict(
        type='PackSegInputs'
    )

]



# ==================================================
# Dataloader TRAIN
# ==================================================

train_dataloader = dict(

    batch_size=2,

    num_workers=2,

    drop_last=True,


    dataset=dict(

        type=dataset_type,


        data_root=data_root,


        metainfo=metainfo,


        data_prefix=dict(

            img_path='train/images',

            seg_map_path='train/masks'

        ),


        pipeline=train_pipeline

    )

)



# ==================================================
# Dataloader TEST
# ==================================================

val_dataloader = dict(

    batch_size=1,

    num_workers=2,


    dataset=dict(

        type=dataset_type,


        data_root=data_root,


        metainfo=metainfo,


        data_prefix=dict(

            img_path='test/images',

            seg_map_path='test/masks'

        ),


        pipeline=test_pipeline

    )

)



test_dataloader = val_dataloader



# ==================================================
# Evaluation
# ==================================================

val_evaluator = dict(

    type='IoUMetric',

    iou_metrics=['mIoU']

)



test_evaluator = val_evaluator



# ==================================================
# Optimizer
# ==================================================

optim_wrapper = dict(

    optimizer=dict(

        type='SGD',

        lr=0.01,

        momentum=0.9,

        weight_decay=0.0005

    )

)



# ==================================================
# Training
# ==================================================

train_cfg = dict(

    type='IterBasedTrainLoop',

    max_iters=20000,

    val_interval=1000

)



val_cfg = dict(

    type='ValLoop'

)



test_cfg = dict(

    type='TestLoop'

)



# ==================================================
# Learning rate
# ==================================================

param_scheduler = [

    dict(

        type='PolyLR',

        eta_min=0,

        power=0.9,

        begin=0,

        end=20000,

        by_epoch=False

    )

]



# ==================================================
# Checkpoints
# ==================================================

default_hooks = dict(

    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=1000,
        save_best='mIoU',
        rule='greater',
        max_keep_ckpts=3
    )

)

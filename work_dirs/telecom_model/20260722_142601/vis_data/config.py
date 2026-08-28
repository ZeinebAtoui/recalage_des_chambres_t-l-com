crop_size = (
    512,
    1024,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        512,
        1024,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
data_root = 'dataset_telecom/'
dataset_type = 'CityscapesDataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=200, type='CheckpointHook'),
    logger=dict(interval=10, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
device = 'cpu'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
load_from = 'https://download.openmmlab.com/mmsegmentation/v1.0/deeplabv3plus/deeplabv3plus_r50-d8_4xb2-80k_cityscapes-512x1024/deeplabv3plus_r50-d8_4xb2-80k_cityscapes-512x1024_20221208_213106-bc30f7a0.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=1024,
        in_index=2,
        loss_decode=dict(
            loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='BN'),
        num_classes=36,
        num_convs=1,
        type='FCNHead'),
    backbone=dict(
        contract_dilation=True,
        depth=50,
        dilations=(
            1,
            1,
            2,
            4,
        ),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=False,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        strides=(
            1,
            2,
            1,
            1,
        ),
        style='pytorch',
        type='ResNetV1c'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            512,
            1024,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        c1_channels=48,
        c1_in_channels=256,
        channels=512,
        dilations=(
            1,
            12,
            24,
            36,
        ),
        dropout_ratio=0.1,
        in_channels=2048,
        in_index=3,
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='BN'),
        num_classes=36,
        type='DepthwiseSeparableASPPHead'),
    pretrained='open-mmlab://resnet50_v1c',
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='BN')
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005),
    type='OptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=80000,
        eta_min=0.0001,
        power=0.9,
        type='PolyLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='test/images', seg_map_path='test/masks'),
        data_root='dataset_telecom/',
        metainfo=dict(
            classes=(
                'background',
                'chambre_A1',
                'chambre_A2',
                'chambre_A3',
                'chambre_A4',
                'chambre_A5',
                'chambre_B1',
                'chambre_B2',
                'chambre_C1',
                'chambre_D2',
                'chambre_K1C',
                'chambre_K2C',
                'chambre_K3C',
                'chambre_L0T',
                'chambre_L1C',
                'chambre_L1T',
                'chambre_L2C',
                'chambre_L2T',
                'chambre_L3C',
                'chambre_L3T',
                'chambre_L4C',
                'chambre_L4T',
                'chambre_L5T',
                'chambre_L6T',
                'chambre_M1C',
                'chambre_M1T',
                'chambre_M2C',
                'chambre_M2T',
                'chambre_M3T',
                'chambre_OHN',
                'chambre_P1C',
                'chambre_P1T',
                'chambre_P2T',
                'chambre_REG',
                'hard_negative',
                'obstacle',
            )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2048,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='BaseSegDataset'),
    num_workers=0,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        2048,
        1024,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=1000, type='IterBasedTrainLoop', val_interval=100)
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='train/images', seg_map_path='train/masks'),
        data_root='dataset_telecom/',
        metainfo=dict(
            classes=(
                'background',
                'chambre_A1',
                'chambre_A2',
                'chambre_A3',
                'chambre_A4',
                'chambre_A5',
                'chambre_B1',
                'chambre_B2',
                'chambre_C1',
                'chambre_D2',
                'chambre_K1C',
                'chambre_K2C',
                'chambre_K3C',
                'chambre_L0T',
                'chambre_L1C',
                'chambre_L1T',
                'chambre_L2C',
                'chambre_L2T',
                'chambre_L3C',
                'chambre_L3T',
                'chambre_L4C',
                'chambre_L4T',
                'chambre_L5T',
                'chambre_L6T',
                'chambre_M1C',
                'chambre_M1T',
                'chambre_M2C',
                'chambre_M2T',
                'chambre_M3T',
                'chambre_OHN',
                'chambre_P1C',
                'chambre_P1T',
                'chambre_P2T',
                'chambre_REG',
                'hard_negative',
                'obstacle',
            )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    2048,
                    1024,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.75, crop_size=(
                    512,
                    1024,
                ), type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='BaseSegDataset'),
    num_workers=0,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            2048,
            1024,
        ),
        type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=(
        512,
        1024,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.75, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='test/images', seg_map_path='test/masks'),
        data_root='dataset_telecom/',
        metainfo=dict(
            classes=(
                'background',
                'chambre_A1',
                'chambre_A2',
                'chambre_A3',
                'chambre_A4',
                'chambre_A5',
                'chambre_B1',
                'chambre_B2',
                'chambre_C1',
                'chambre_D2',
                'chambre_K1C',
                'chambre_K2C',
                'chambre_K3C',
                'chambre_L0T',
                'chambre_L1C',
                'chambre_L1T',
                'chambre_L2C',
                'chambre_L2T',
                'chambre_L3C',
                'chambre_L3T',
                'chambre_L4C',
                'chambre_L4T',
                'chambre_L5T',
                'chambre_L6T',
                'chambre_M1C',
                'chambre_M1T',
                'chambre_M2C',
                'chambre_M2T',
                'chambre_M3T',
                'chambre_OHN',
                'chambre_P1C',
                'chambre_P1T',
                'chambre_P2T',
                'chambre_REG',
                'hard_negative',
                'obstacle',
            )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2048,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='BaseSegDataset'),
    num_workers=0,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/telecom_model'

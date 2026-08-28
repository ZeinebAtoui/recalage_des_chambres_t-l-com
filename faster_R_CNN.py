import torch
import torch.nn as nn

from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter

from torchvision.ops import FeaturePyramidNetwork
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_PATH = "./faster_rcnn_epoch_9.pth"

NUM_CLASSES = 2

# Mapping du modèle
# 0 = background
# 1 = chambre_telecom

CLASS_NAMES = {
    0: "background",
    1: "chambre_telecom",
}


# ============================================================
# FPN : CONV + BATCH NORMALIZATION
# ============================================================

class ConvBN(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size):

        padding = kernel_size // 2

        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )


# ============================================================
# BACKBONE RESNET-50 + FPN
# ============================================================

print("\n[1/6] Construction du ResNet-50 + FPN...")


resnet = resnet50(
    weights=None
)


# Sorties du ResNet utilisées par le FPN
return_layers = {
    "layer1": "0",
    "layer2": "1",
    "layer3": "2",
    "layer4": "3",
}


body = IntermediateLayerGetter(
    resnet,
    return_layers=return_layers
)


# FPN
fpn = FeaturePyramidNetwork(
    in_channels_list=[
        256,
        512,
        1024,
        2048
    ],
    out_channels=256
)


# ------------------------------------------------------------
# Remplacement des blocs FPN
# par Conv2d + BatchNorm2d
# ------------------------------------------------------------

fpn.inner_blocks = nn.ModuleList([

    ConvBN(
        256,
        256,
        1
    ),

    ConvBN(
        512,
        256,
        1
    ),

    ConvBN(
        1024,
        256,
        1
    ),

    ConvBN(
        2048,
        256,
        1
    )
])


fpn.layer_blocks = nn.ModuleList([

    ConvBN(
        256,
        256,
        3
    ),

    ConvBN(
        256,
        256,
        3
    ),

    ConvBN(
        256,
        256,
        3
    ),

    ConvBN(
        256,
        256,
        3
    )
])


# ============================================================
# BACKBONE COMPLET
# ============================================================

class CustomBackbone(nn.Module):

    def __init__(self, body, fpn):

        super().__init__()

        self.body = body
        self.fpn = fpn

        self.out_channels = 256


    def forward(self, x):

        x = self.body(x)

        x = self.fpn(x)

        return x


backbone = CustomBackbone(
    body,
    fpn
)


print("OK")
print("ResNet-50 + FPN")
print("FPN channels : 256")


# ============================================================
# ANCHOR GENERATOR
# ============================================================

print("\n[2/6] Construction du AnchorGenerator...")


anchor_generator = AnchorGenerator(

    sizes=(
        (32,),
        (64,),
        (128,),
        (256,),
    ),

    aspect_ratios=(
        (0.5, 1.0, 2.0),
        (0.5, 1.0, 2.0),
        (0.5, 1.0, 2.0),
        (0.5, 1.0, 2.0),
    )
)


print("OK")
print("Anchors par position : 3")


# ============================================================
# RPN HEAD PERSONNALISE
# ============================================================

print("\n[3/6] Construction du RPN Head...")


class CustomRPNHead(nn.Module):

    def __init__(
        self,
        in_channels,
        num_anchors
    ):

        super().__init__()


        # Deux convolutions
        self.conv = nn.Sequential(

            nn.Sequential(

                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    padding=1
                ),

                nn.ReLU()
            ),


            nn.Sequential(

                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    padding=1
                ),

                nn.ReLU()
            )
        )


        # Classification RPN

        self.cls_logits = nn.Conv2d(
            in_channels,
            num_anchors,
            kernel_size=1
        )


        # Régression RPN

        self.bbox_pred = nn.Conv2d(
            in_channels,
            num_anchors * 4,
            kernel_size=1
        )


    def forward(self, x):

        logits = []

        bbox_reg = []


        for feature in x:

            t = self.conv(feature)


            logits.append(
                self.cls_logits(t)
            )


            bbox_reg.append(
                self.bbox_pred(t)
            )


        return logits, bbox_reg


rpn_head = CustomRPNHead(
    in_channels=256,
    num_anchors=3
)


print("OK")
print("RPN convolutions : 2")
print("RPN anchors      : 3")


# ============================================================
# ROI ALIGN
# ============================================================

print("\n[4/6] Construction du ROIAlign...")


roi_pooler = MultiScaleRoIAlign(

    featmap_names=[
        "0",
        "1",
        "2",
        "3"
    ],

    output_size=7,

    sampling_ratio=2
)


print("OK")
print("ROI output : 7 x 7")


# ============================================================
# ROI BOX HEAD PERSONNALISE
# ============================================================

print("\n[5/6] Construction du ROI Box Head...")


class CustomBoxHead(nn.Sequential):

    def __init__(self):

        super().__init__(

            # ------------------------------------------------
            # CONVOLUTION 1
            # ------------------------------------------------

            nn.Sequential(

                nn.Conv2d(
                    256,
                    256,
                    kernel_size=3,
                    padding=1,

                    # IMPORTANT
                    # Le checkpoint n'a pas de bias
                    bias=False
                ),

                nn.BatchNorm2d(256),

                nn.ReLU()
            ),


            # ------------------------------------------------
            # CONVOLUTION 2
            # ------------------------------------------------

            nn.Sequential(

                nn.Conv2d(
                    256,
                    256,
                    kernel_size=3,
                    padding=1,

                    bias=False
                ),

                nn.BatchNorm2d(256),

                nn.ReLU()
            ),


            # ------------------------------------------------
            # CONVOLUTION 3
            # ------------------------------------------------

            nn.Sequential(

                nn.Conv2d(
                    256,
                    256,
                    kernel_size=3,
                    padding=1,

                    bias=False
                ),

                nn.BatchNorm2d(256),

                nn.ReLU()
            ),


            # ------------------------------------------------
            # CONVOLUTION 4
            # ------------------------------------------------

            nn.Sequential(

                nn.Conv2d(
                    256,
                    256,
                    kernel_size=3,
                    padding=1,

                    bias=False
                ),

                nn.BatchNorm2d(256),

                nn.ReLU()
            ),


            # ------------------------------------------------
            # FLATTEN
            # ------------------------------------------------

            nn.Flatten(),


            # ------------------------------------------------
            # FULLY CONNECTED
            # ------------------------------------------------

            nn.Linear(
                256 * 7 * 7,
                1024
            ),


            nn.ReLU()
        )


box_head = CustomBoxHead()


print("OK")
print("4 Conv + BatchNorm")
print("Flatten")
print("12544 -> 1024")


# ============================================================
# BOX PREDICTOR
# ============================================================

print("\nConstruction du Box Predictor...")


class CustomBoxPredictor(nn.Module):

    def __init__(
        self,
        in_channels,
        num_classes
    ):

        super().__init__()


        # Classification

        self.cls_score = nn.Linear(
            in_channels,
            num_classes
        )


        # Bounding boxes

        self.bbox_pred = nn.Linear(
            in_channels,
            num_classes * 4
        )


    def forward(self, x):

        if x.dim() > 2:

            x = torch.flatten(
                x,
                start_dim=1
            )


        scores = self.cls_score(x)

        bbox_deltas = self.bbox_pred(x)


        return scores, bbox_deltas


box_predictor = CustomBoxPredictor(

    in_channels=1024,

    num_classes=NUM_CLASSES
)


print("Classification : 1024 -> 2")
print("BBox           : 1024 -> 8")


# ============================================================
# FASTER R-CNN
# ============================================================

print("\n[6/6] Construction du Faster R-CNN...")


model = FasterRCNN(

    backbone=backbone,

    num_classes=NUM_CLASSES,

    rpn_anchor_generator=anchor_generator,

    rpn_head=rpn_head,

    box_roi_pool=roi_pooler
)


# ============================================================
# REMPLACEMENT DU ROI HEAD
# ============================================================

model.roi_heads.box_head = box_head

model.roi_heads.box_predictor = box_predictor


print("OK")


# ============================================================
# INFORMATIONS DU MODELE
# ============================================================

print("\n")
print("=" * 70)
print("MODELE")
print("=" * 70)

print()

for idx, name in CLASS_NAMES.items():

    print(
        f"{idx} -> {name}"
    )


print()

print("Architecture :")
print("  Faster R-CNN")
print("  Backbone : ResNet-50")
print("  FPN : 256 channels")
print("  RPN : 2 convolutions")
print("  RPN : 3 anchors")
print("  ROIAlign : 7 x 7")
print("  ROI Box Head : 4 Conv + BatchNorm")
print("  FC : 12544 -> 1024")
print("  Classes : 2")
print("  BBox : 8")


# ============================================================
# CHARGEMENT DU CHECKPOINT
# ============================================================

print("\n")
print("=" * 70)
print("CHARGEMENT CHECKPOINT")
print("=" * 70)


print()

print(
    "Checkpoint :",
    CHECKPOINT_PATH
)


checkpoint = torch.load(

    CHECKPOINT_PATH,

    map_location="cpu",

    weights_only=False
)


print()

print(
    "Epoch :",
    checkpoint.get("epoch")
)


print(
    "Loss  :",
    checkpoint.get("loss")
)


# ============================================================
# MODEL STATE DICT
# ============================================================

state_dict = checkpoint[
    "model_state_dict"
]


print()

print(
    "Nombre de paramètres dans le checkpoint :",
    len(state_dict)
)

 
# ============================================================
# CHARGEMENT STRICT
# ============================================================

print("\nChargement des poids...")


try:

    model.load_state_dict(

        state_dict,

        strict=True
    )


    print("\n")
    print("=" * 70)
    print("SUCCES TOTAL")
    print("=" * 70)

    print()

    print(
        "Le checkpoint est parfaitement compatible "
        "avec le modèle reconstruit."
    )


except RuntimeError as e:

    print("\n")
    print("=" * 70)
    print("ERREUR DE COMPATIBILITE")
    print("=" * 70)

    print()

    print(e)


# ============================================================
# MODE EVALUATION
# ============================================================

model.eval()


print()

print(
    "Mode evaluation activé."
)


print()

print(
    "Modèle prêt."
)
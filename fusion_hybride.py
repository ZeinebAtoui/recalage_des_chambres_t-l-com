# ============================================================
# ablation_hybride.py
# ============================================================
#
# OBJECTIF :
#   Tester independamment chaque correctif du pipeline hybride
#   (seuil Faster R-CNN, marge de crop, TTA, post-traitement
#   morphologique) afin d'isoler leur contribution respective
#   sur la Precision / Recall / F1 de la classe chambre_telecom.
#
#   Les modeles sont charges UNE SEULE FOIS, puis chaque
#   configuration est evaluee sur le meme jeu de test.
#
# ============================================================

import os
import json
import cv2
import torch
import torch.nn as nn
import numpy as np
import csv

from collections import defaultdict

# ============================================================
# CONFIGURATION GLOBALE (chemins, inchange)
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print()
print("=" * 70)
print("DEVICE")
print("=" * 70)
print("Device utilise :", DEVICE)
if torch.cuda.is_available():
    print("GPU :", torch.cuda.get_device_name(0))

FASTER_CHECKPOINT = "./faster_rcnn_epoch_9.pth"
DEEPLAB_CHECKPOINT = "./best_mIoU_iter_2000.pth"
DEEPLAB_CONFIG = "./deeplab_lyon (1).py"

DATASET_DIR = "./dataset_test_hyb"
COCO_JSON = os.path.join(DATASET_DIR, "annotations", "test_faster_rcnn.json")
IMAGE_DIR = os.path.join(DATASET_DIR, "images")
MASK_DIR = os.path.join(DATASET_DIR, "masks")

OUTPUT_DIR = "./resultats_ablation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FASTER_CHAMBRE_ID = 1

DEEPLAB_BACKGROUND_ID = 0
DEEPLAB_HARD_NEGATIVE_ID = 1
DEEPLAB_CHAMBRE_ID = 2

CLASS_NAMES = {
    0: "background",
    1: "hard_negative",
    2: "chambre_telecom"
}

# Limiter le nombre d'images pour un test rapide (None = toutes les images)
LIMITE_IMAGES = None   # ex: 100 pour un test rapide, None pour tout le dataset


# ============================================================
# CONFIGURATIONS A TESTER (ABLATION)
# ============================================================
#
# score_th        : seuil de detection Faster R-CNN
# margin_init     : marge relative initiale du crop
# margin_max      : marge relative maximale autorisee
# margin_step     : increment de marge a chaque iteration
# use_tta         : active/desactive la Test-Time Augmentation
# tta_mode        : "union" (hard, ancienne version) ou "soft" (moyenne des logits)
# use_morph       : active/desactive le post-traitement morphologique
# morph_kernel    : taille du noyau de fermeture morphologique
# keep_largest    : ne garder que les composantes connexes significatives
#
CONFIGURATIONS = {

    "baseline_v1": {
        "score_th": 0.60, "margin_init": 0.20, "margin_max": 0.20, "margin_step": 0.10,
        "use_tta": False, "tta_mode": "soft",
        "use_morph": False, "morph_kernel": 3, "keep_largest": False
    },

    "seuil_seul": {
        "score_th": 0.50, "margin_init": 0.20, "margin_max": 0.20, "margin_step": 0.10,
        "use_tta": False, "tta_mode": "soft",
        "use_morph": False, "morph_kernel": 3, "keep_largest": False
    },

    "marge_seule": {
        "score_th": 0.60, "margin_init": 0.35, "margin_max": 0.50, "margin_step": 0.10,
        "use_tta": False, "tta_mode": "soft",
        "use_morph": False, "morph_kernel": 3, "keep_largest": False
    },

    "tta_soft_seul": {
        "score_th": 0.60, "margin_init": 0.20, "margin_max": 0.20, "margin_step": 0.10,
        "use_tta": True, "tta_mode": "soft",
        "use_morph": False, "morph_kernel": 3, "keep_largest": False
    },

    "morph_seul": {
        "score_th": 0.60, "margin_init": 0.20, "margin_max": 0.20, "margin_step": 0.10,
        "use_tta": False, "tta_mode": "soft",
        "use_morph": True, "morph_kernel": 3, "keep_largest": True
    },

    "combinaison_moderee": {
        "score_th": 0.55, "margin_init": 0.30, "margin_max": 0.40, "margin_step": 0.10,
        "use_tta": True, "tta_mode": "soft",
        "use_morph": True, "morph_kernel": 3, "keep_largest": True
    },

    "combinaison_v2_precedente": {
        # reproduit approximativement votre v2 (qui a surcorrige)
        "score_th": 0.50, "margin_init": 0.35, "margin_max": 0.90, "margin_step": 0.15,
        "use_tta": True, "tta_mode": "union",
        "use_morph": True, "morph_kernel": 7, "keep_largest": True
    },
}


# ============================================================
# CHARGEMENT FASTER R-CNN (inchange, factorise)
# ============================================================

def charger_faster_rcnn():
    print()
    print("=" * 70)
    print("CHARGEMENT FASTER R-CNN")
    print("=" * 70)

    from torchvision.models import resnet50
    from torchvision.models._utils import IntermediateLayerGetter
    from torchvision.ops import FeaturePyramidNetwork
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.rpn import AnchorGenerator
    from torchvision.ops import MultiScaleRoIAlign

    NUM_CLASSES = 2

    class ConvBN(nn.Sequential):
        def __init__(self, in_channels, out_channels, kernel_size):
            padding = kernel_size // 2
            super().__init__(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                           padding=padding, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    resnet = resnet50(weights=None)
    return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
    body = IntermediateLayerGetter(resnet, return_layers=return_layers)

    fpn = FeaturePyramidNetwork(
        in_channels_list=[256, 512, 1024, 2048],
        out_channels=256
    )
    fpn.inner_blocks = nn.ModuleList([
        ConvBN(256, 256, 1), ConvBN(512, 256, 1),
        ConvBN(1024, 256, 1), ConvBN(2048, 256, 1)
    ])
    fpn.layer_blocks = nn.ModuleList([
        ConvBN(256, 256, 3), ConvBN(256, 256, 3),
        ConvBN(256, 256, 3), ConvBN(256, 256, 3)
    ])

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

    backbone = CustomBackbone(body, fpn)

    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4
    )

    class CustomRPNHead(nn.Module):
        def __init__(self, in_channels, num_anchors):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1), nn.ReLU()),
                nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1), nn.ReLU())
            )
            self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1)
            self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)

        def forward(self, x):
            logits, bbox_reg = [], []
            for feature in x:
                t = self.conv(feature)
                logits.append(self.cls_logits(t))
                bbox_reg.append(self.bbox_pred(t))
            return logits, bbox_reg

    rpn_head = CustomRPNHead(in_channels=256, num_anchors=3)

    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=7,
        sampling_ratio=2
    )

    class CustomBoxHead(nn.Sequential):
        def __init__(self):
            super().__init__(
                nn.Sequential(nn.Conv2d(256, 256, 3, padding=1, bias=False),
                               nn.BatchNorm2d(256), nn.ReLU()),
                nn.Sequential(nn.Conv2d(256, 256, 3, padding=1, bias=False),
                               nn.BatchNorm2d(256), nn.ReLU()),
                nn.Sequential(nn.Conv2d(256, 256, 3, padding=1, bias=False),
                               nn.BatchNorm2d(256), nn.ReLU()),
                nn.Sequential(nn.Conv2d(256, 256, 3, padding=1, bias=False),
                               nn.BatchNorm2d(256), nn.ReLU()),
                nn.Flatten(),
                nn.Linear(256 * 7 * 7, 1024),
                nn.ReLU()
            )

    box_head = CustomBoxHead()

    class CustomBoxPredictor(nn.Module):
        def __init__(self, in_channels, num_classes):
            super().__init__()
            self.cls_score = nn.Linear(in_channels, num_classes)
            self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

        def forward(self, x):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            return self.cls_score(x), self.bbox_pred(x)

    box_predictor = CustomBoxPredictor(in_channels=1024, num_classes=NUM_CLASSES)

    model = FasterRCNN(
        backbone=backbone,
        num_classes=NUM_CLASSES,
        rpn_anchor_generator=anchor_generator,
        rpn_head=rpn_head,
        box_roi_pool=roi_pooler
    )
    model.roi_heads.box_head = box_head
    model.roi_heads.box_predictor = box_predictor

    checkpoint = torch.load(FASTER_CHECKPOINT, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned_state_dict[key] = value

    model.load_state_dict(cleaned_state_dict, strict=True)
    model.to(DEVICE)
    model.eval()

    print("Faster R-CNN charge avec succes.")
    return model


# ============================================================
# CHARGEMENT DEEPLABV3+ (inchange)
# ============================================================

def charger_deeplab():
    print()
    print("=" * 70)
    print("CHARGEMENT DEEPLABV3+")
    print("=" * 70)

    from mmengine.config import Config
    from mmseg.apis import init_model

    cfg = Config.fromfile(DEEPLAB_CONFIG)
    model = init_model(cfg, DEEPLAB_CHECKPOINT, device=str(DEVICE))
    model.eval()

    print("DeepLabV3+ charge avec succes.")
    return model


# ============================================================
# CHARGEMENT COCO / GT (inchange)
# ============================================================

def charger_coco(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    annotations_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        annotations_by_image[ann["image_id"]].append(ann)

    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    return coco, images, annotations_by_image, categories


def charger_masque_png(image_name, height, width):
    base = os.path.splitext(image_name)[0]
    candidats = [
        os.path.join(MASK_DIR, image_name),
        os.path.join(MASK_DIR, base + ".png"),
        os.path.join(MASK_DIR, base + ".jpg"),
        os.path.join(MASK_DIR, base + ".jpeg")
    ]
    mask_path = next((p for p in candidats if os.path.exists(p)), None)
    if mask_path is None:
        return None

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask.astype(np.uint8)


def construire_masque_gt_coco(image_info, annotations):
    width = image_info["width"]
    height = image_info["height"]
    gt_mask = np.zeros((height, width), dtype=np.uint8)

    for ann in annotations:
        category_id = ann["category_id"]
        segmentation = ann.get("segmentation")
        if not segmentation:
            continue
        if isinstance(segmentation, list):
            mask = np.zeros((height, width), dtype=np.uint8)
            for polygon in segmentation:
                if len(polygon) < 6:
                    continue
                points = np.array(polygon, dtype=np.float32).reshape(-1, 2)
                points = np.round(points).astype(np.int32)
                cv2.fillPoly(mask, [points], 1)
            if category_id == 1:
                gt_mask[mask == 1] = 1
            elif category_id == 2:
                gt_mask[mask == 1] = 2

    return gt_mask


def construire_gt(image_info, annotations):
    image_name = image_info["file_name"]
    height = image_info["height"]
    width = image_info["width"]

    png_mask = charger_masque_png(image_name, height, width)
    if png_mask is not None:
        return png_mask
    return construire_masque_gt_coco(image_info, annotations)


# ============================================================
# INFERENCE FASTER R-CNN (parametrable par score_th)
# ============================================================

def inference_faster(faster_model, image, score_th):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        prediction = faster_model([tensor])[0]

    boxes = prediction["boxes"].detach().cpu().numpy()
    labels = prediction["labels"].detach().cpu().numpy()
    scores = prediction["scores"].detach().cpu().numpy()

    detections = []
    for box, label, score in zip(boxes, labels, scores):
        if int(label) != FASTER_CHAMBRE_ID:
            continue
        if float(score) < score_th:
            continue
        x1, y1, x2, y2 = box
        detections.append({
            "box": (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
            "score": float(score)
        })

    return detections


# ============================================================
# CROP AVEC MARGE (parametrable)
# ============================================================

def extraire_crop_avec_marge(image, box, margin_ratio, margin_min_pixels=15):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return None, None

    margin_x = max(int(bw * margin_ratio), margin_min_pixels)
    margin_y = max(int(bh * margin_ratio), margin_min_pixels)

    x1e = max(0, x1 - margin_x)
    y1e = max(0, y1 - margin_y)
    x2e = min(w, x2 + margin_x)
    y2e = min(h, y2 + margin_y)

    crop = image[y1e:y2e, x1e:x2e]
    if crop.size == 0:
        return None, None

    return crop, (x1e, y1e, x2e, y2e)


def masque_touche_le_bord(mask, classe=DEEPLAB_CHAMBRE_ID, epaisseur=2):
    h, w = mask.shape
    haut = mask[0:epaisseur, :]
    bas = mask[h - epaisseur:h, :]
    gauche = mask[:, 0:epaisseur]
    droite = mask[:, w - epaisseur:w]
    return (
        np.any(haut == classe) or np.any(bas == classe)
        or np.any(gauche == classe) or np.any(droite == classe)
    )


# ============================================================
# INFERENCE DEEPLAB (normale + TTA union + TTA soft)
# ============================================================

def inference_deeplab_logits(deeplab_model, crop):
    """Retourne les logits bruts (C, H, W) avant argmax."""
    from mmseg.apis import inference_model
    result = inference_model(deeplab_model, crop)
    logits = result.seg_logits.data  # tensor (C, H, W)
    return logits


def inference_deeplab_single(deeplab_model, crop):
    logits = inference_deeplab_logits(deeplab_model, crop)
    mask = torch.argmax(logits, dim=0).cpu().numpy().astype(np.uint8)
    return mask


def inference_deeplab_avec_config(deeplab_model, crop, cfg):
    """Applique (ou non) la TTA selon la configuration testee."""
    if not cfg["use_tta"]:
        return inference_deeplab_single(deeplab_model, crop)

    logits_normal = inference_deeplab_logits(deeplab_model, crop)
    mask_normal = torch.argmax(logits_normal, dim=0).cpu().numpy().astype(np.uint8)

    crop_flip = cv2.flip(crop, 1)
    logits_flip = inference_deeplab_logits(deeplab_model, crop_flip)
    logits_flip_retour = torch.flip(logits_flip, dims=[-1])

    if cfg["tta_mode"] == "soft":
        # Fusion par moyenne des logits (recommande : plus stable)
        if logits_flip_retour.shape != logits_normal.shape:
            logits_flip_retour = torch.nn.functional.interpolate(
                logits_flip_retour.unsqueeze(0),
                size=logits_normal.shape[-2:],
                mode="nearest"
            ).squeeze(0)
        logits_moyen = (logits_normal + logits_flip_retour) / 2.0
        mask_final = torch.argmax(logits_moyen, dim=0).cpu().numpy().astype(np.uint8)
        return mask_final

    else:
        # Mode "union" (ancienne version, hard, a but de comparaison)
        mask_flip = torch.argmax(logits_flip, dim=0).cpu().numpy().astype(np.uint8)
        mask_flip_retour = cv2.flip(mask_flip, 1)
        if mask_flip_retour.shape != mask_normal.shape:
            mask_flip_retour = cv2.resize(
                mask_flip_retour,
                (mask_normal.shape[1], mask_normal.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        mask_final = mask_normal.copy()
        union_chambre = (mask_normal == DEEPLAB_CHAMBRE_ID) | (mask_flip_retour == DEEPLAB_CHAMBRE_ID)
        mask_final[union_chambre] = DEEPLAB_CHAMBRE_ID
        return mask_final


# ============================================================
# CROP ADAPTATIF (parametrable)
# ============================================================

def extraire_crop_adaptatif(image, box, deeplab_model, cfg):
    margin = cfg["margin_init"]
    dernier_crop, dernier_crop_box, dernier_mask = None, None, None

    while margin <= cfg["margin_max"]:
        crop, crop_box = extraire_crop_avec_marge(image, box, margin)
        if crop is None:
            break

        mask_crop = inference_deeplab_avec_config(deeplab_model, crop, cfg)
        dernier_crop, dernier_crop_box, dernier_mask = crop, crop_box, mask_crop

        if not masque_touche_le_bord(mask_crop):
            return crop, crop_box, mask_crop

        margin += cfg["margin_step"]

    return dernier_crop, dernier_crop_box, dernier_mask


# ============================================================
# POST-TRAITEMENT MORPHOLOGIQUE (parametrable)
# ============================================================

def post_traiter_masque_crop(mask_crop, cfg):
    if not cfg["use_morph"]:
        return mask_crop

    binaire = (mask_crop == DEEPLAB_CHAMBRE_ID).astype(np.uint8)
    if binaire.sum() == 0:
        return mask_crop

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg["morph_kernel"], cfg["morph_kernel"]))
    ferme = cv2.morphologyEx(binaire, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(ferme, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rempli = np.zeros_like(ferme)
    cv2.drawContours(rempli, contours, -1, 1, thickness=cv2.FILLED)

    if cfg["keep_largest"]:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(rempli, connectivity=8)
        if num_labels > 1:
            aires = stats[1:, cv2.CC_STAT_AREA]
            aire_max = aires.max()
            filtre = np.zeros_like(rempli)
            for label_id in range(1, num_labels):
                if stats[label_id, cv2.CC_STAT_AREA] >= aire_max * 0.05:
                    filtre[labels == label_id] = 1
            rempli = filtre

    resultat = mask_crop.copy()
    resultat[(mask_crop == DEEPLAB_CHAMBRE_ID) & (rempli == 0)] = DEEPLAB_BACKGROUND_ID
    resultat[rempli == 1] = DEEPLAB_CHAMBRE_ID

    return resultat


# ============================================================
# FUSION IMAGE (parametrable par configuration)
# ============================================================

def fusion_image(faster_model, deeplab_model, image, cfg):
    height, width = image.shape[:2]
    final_mask = np.zeros((height, width), dtype=np.uint8)

    detections = inference_faster(faster_model, image, cfg["score_th"])

    for detection in detections:
        box = detection["box"]
        crop, crop_box, mask_crop = extraire_crop_adaptatif(image, box, deeplab_model, cfg)

        if crop is None or mask_crop is None:
            continue

        x1, y1, x2, y2 = crop_box
        crop_height = y2 - y1
        crop_width = x2 - x1

        if mask_crop.shape != (crop_height, crop_width):
            mask_crop = cv2.resize(
                mask_crop, (crop_width, crop_height),
                interpolation=cv2.INTER_NEAREST
            )

        mask_crop = post_traiter_masque_crop(mask_crop, cfg)

        region = final_mask[y1:y2, x1:x2]
        region[mask_crop == DEEPLAB_HARD_NEGATIVE_ID] = DEEPLAB_HARD_NEGATIVE_ID
        region[mask_crop == DEEPLAB_CHAMBRE_ID] = DEEPLAB_CHAMBRE_ID
        final_mask[y1:y2, x1:x2] = region

    return final_mask, detections


# ============================================================
# METRIQUES (inchange)
# ============================================================

def calculer_matrice_confusion(gt, pred, num_classes=3):
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    gt = gt.flatten()
    pred = pred.flatten()
    valid = (gt >= 0) & (gt < num_classes) & (pred >= 0) & (pred < num_classes)
    gt = gt[valid]
    pred = pred[valid]
    np.add.at(confusion, (gt, pred), 1)
    return confusion


def calculer_metriques(confusion):
    metrics = {}
    for class_id in range(confusion.shape[0]):
        tp = confusion[class_id, class_id]
        fp = confusion[:, class_id].sum() - tp
        fn = confusion[class_id, :].sum() - tp
        union = tp + fp + fn

        iou = tp / union if union > 0 else np.nan
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan

        if not np.isnan(precision) and not np.isnan(recall) and precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = np.nan

        dice_den = 2 * tp + fp + fn
        dice = 2 * tp / dice_den if dice_den > 0 else np.nan

        metrics[class_id] = {
            "IoU": iou, "Dice": dice, "Precision": precision,
            "Recall": recall, "F1": f1,
            "TP": int(tp), "FP": int(fp), "FN": int(fn)
        }
    return metrics


def calculer_pixel_accuracy(confusion):
    total = confusion.sum()
    if total == 0:
        return np.nan
    return np.trace(confusion) / total


# ============================================================
# EVALUATION D'UNE CONFIGURATION
# ============================================================

def evaluer_configuration(nom_config, cfg, faster_model, deeplab_model, images, annotations_by_image):
    print()
    print("=" * 70)
    print(f"EVALUATION CONFIGURATION : {nom_config}")
    print("=" * 70)
    print("Parametres :", cfg)

    confusion_hybride = np.zeros((3, 3), dtype=np.int64)

    liste_ids = list(images.keys())
    if LIMITE_IMAGES is not None:
        liste_ids = liste_ids[:LIMITE_IMAGES]

    total = len(liste_ids)

    for index, image_id in enumerate(liste_ids):
        image_info = images[image_id]
        file_name = image_info["file_name"]
        image_path = os.path.join(IMAGE_DIR, file_name)

        if not os.path.exists(image_path):
            continue

        image = cv2.imread(image_path)
        if image is None:
            continue

        height, width = image.shape[:2]
        annotations = annotations_by_image.get(image_id, [])

        gt_mask = construire_gt(image_info, annotations)
        if gt_mask.shape != (height, width):
            gt_mask = cv2.resize(gt_mask, (width, height), interpolation=cv2.INTER_NEAREST)

        pred_mask, _ = fusion_image(faster_model, deeplab_model, image, cfg)
        confusion_hybride += calculer_matrice_confusion(gt_mask, pred_mask, num_classes=3)

        if (index + 1) % 20 == 0 or (index + 1) == total:
            print(f"  [{index + 1}/{total}] images traitees")

    metrics = calculer_metriques(confusion_hybride)
    pixel_accuracy = calculer_pixel_accuracy(confusion_hybride)

    valid_ious = [metrics[i]["IoU"] for i in range(3) if not np.isnan(metrics[i]["IoU"])]
    miou = np.mean(valid_ious) if valid_ious else np.nan

    resultat = {
        "configuration": nom_config,
        "pixel_accuracy": pixel_accuracy,
        "mIoU": miou,
        "IoU_chambre": metrics[2]["IoU"],
        "Dice_chambre": metrics[2]["Dice"],
        "Precision_chambre": metrics[2]["Precision"],
        "Recall_chambre": metrics[2]["Recall"],
        "F1_chambre": metrics[2]["F1"],
        "TP": metrics[2]["TP"],
        "FP": metrics[2]["FP"],
        "FN": metrics[2]["FN"],
    }

    print()
    print(f"--- Resultats {nom_config} ---")
    for cle, valeur in resultat.items():
        if isinstance(valeur, float):
            print(f"  {cle:<20}: {valeur:.4f}")
        else:
            print(f"  {cle:<20}: {valeur}")

    return resultat


# ============================================================
# AFFICHAGE DU TABLEAU COMPARATIF FINAL
# ============================================================

def afficher_tableau_comparatif(resultats):
    print()
    print("=" * 110)
    print("TABLEAU COMPARATIF FINAL")
    print("=" * 110)

    entete = f"{'Configuration':<28}{'Precision':>12}{'Recall':>12}{'F1':>12}{'IoU':>12}{'mIoU':>12}{'PixelAcc':>12}"
    print(entete)
    print("-" * 110)

    for res in resultats:
        ligne = (
            f"{res['configuration']:<28}"
            f"{res['Precision_chambre']:>12.4f}"
            f"{res['Recall_chambre']:>12.4f}"
            f"{res['F1_chambre']:>12.4f}"
            f"{res['IoU_chambre']:>12.4f}"
            f"{res['mIoU']:>12.4f}"
            f"{res['pixel_accuracy']:>12.4f}"
        )
        print(ligne)

    print("=" * 110)

    meilleur_f1 = max(resultats, key=lambda r: r["F1_chambre"] if not np.isnan(r["F1_chambre"]) else -1)
    print()
    print(f"Meilleure configuration (F1 chambre_telecom) : {meilleur_f1['configuration']}")
    print(f"  Precision : {meilleur_f1['Precision_chambre']:.4f}")
    print(f"  Recall    : {meilleur_f1['Recall_chambre']:.4f}")
    print(f"  F1        : {meilleur_f1['F1_chambre']:.4f}")
    print(f"  IoU       : {meilleur_f1['IoU_chambre']:.4f}")


def exporter_csv(resultats, chemin_csv):
    if not resultats:
        return
    with open(chemin_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(resultats[0].keys()))
        writer.writeheader()
        for res in resultats:
            writer.writerow(res)
    print()
    print("Resultats exportes vers :", os.path.abspath(chemin_csv))


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 70)
    print("ABLATION DU PIPELINE HYBRIDE FASTER R-CNN + DEEPLABV3+")
    print("=" * 70)

    faster_model = charger_faster_rcnn()
    deeplab_model = charger_deeplab()

    coco, images, annotations_by_image, categories = charger_coco(COCO_JSON)

    resultats = []
    for nom_config, cfg in CONFIGURATIONS.items():
        resultat = evaluer_configuration(
            nom_config, cfg, faster_model, deeplab_model, images, annotations_by_image
        )
        resultats.append(resultat)

    afficher_tableau_comparatif(resultats)
    exporter_csv(resultats, os.path.join(OUTPUT_DIR, "comparatif_ablation.csv"))

    print()
    print("=" * 70)
    print("ABLATION TERMINEE")
    print("=" * 70)


if __name__ == "__main__":
    main()

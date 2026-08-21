# ============================================================
# fusion_finale.py
# ============================================================
# Architecture complete :
#   Faster R-CNN (localisation)
#       -> DeepLabV3+ TTA soft (classification, meme crop qu'avant)
#       -> SAM2 (contours precis, appele via subprocess dans sam2_env)
#       -> Fusion semantique (vote DeepLabV3+ dans le masque SAM2)
#       -> Visualisations overlay (GT vs prediction)
#
# A executer avec l'interpreteur de env_hybrid_311.
# ============================================================

import os
import json
import subprocess
import shutil
import cv2
import torch
import torch.nn as nn
import numpy as np
import csv
from collections import defaultdict

import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION GLOBALE
# ============================================================

DEVICE = torch.device("cpu")

print()
print("=" * 70)
print("DEVICE")
print("=" * 70)
print("Device utilise :", DEVICE)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FASTER_CHECKPOINT = os.path.join(BASE_DIR, "faster_rcnn_epoch_9.pth")
DEEPLAB_CHECKPOINT = os.path.join(BASE_DIR, "best_mIoU_iter_2000.pth")
DEEPLAB_CONFIG = os.path.join(BASE_DIR, "deeplab_lyon (1).py")

DATASET_DIR = os.path.join(BASE_DIR, "dataset_test_hyb")
COCO_JSON = os.path.join(DATASET_DIR, "annotations", "test_faster_rcnn.json")
IMAGE_DIR = os.path.join(DATASET_DIR, "images")
MASK_DIR = os.path.join(DATASET_DIR, "masks")

OUTPUT_DIR = os.path.join(BASE_DIR, "resultats_finale")
os.makedirs(OUTPUT_DIR, exist_ok=True)

VIZ_DIR = os.path.join(BASE_DIR, "visualisations")
os.makedirs(VIZ_DIR, exist_ok=True)

TEMP_IPC_DIR = os.path.join(BASE_DIR, "temp_ipc")
TEMP_CROPS_DIR = os.path.join(TEMP_IPC_DIR, "crops")
TEMP_MASQUES_DIR = os.path.join(TEMP_IPC_DIR, "masques_sam2")
REQUETES_JSON = os.path.join(TEMP_IPC_DIR, "requetes_sam2.json")
RESULTATS_JSON = os.path.join(TEMP_IPC_DIR, "resultats_sam2.json")

SAM2_WORKER_SCRIPT = os.path.join(BASE_DIR, "sam2_worker.py")

# --- A ADAPTER selon votre machine ---
SAM2_ENV_PYTHON = "/home/zeineb/envs/sam2_env/bin/python3"
SAM2_REPO_PATH = "/home/zeineb/projets/sam2"
SAM2_CHECKPOINT = os.path.join(SAM2_REPO_PATH, "checkpoints", "sam2.1_hiera_tiny.pt")
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"

FASTER_CHAMBRE_ID = 1

DEEPLAB_BACKGROUND_ID = 0
DEEPLAB_HARD_NEGATIVE_ID = 1
DEEPLAB_CHAMBRE_ID = 2

CONFIG_DEEPLAB = {
    "score_th": 0.60,
    "margin_init": 0.20,
    "margin_max": 0.20,
    "margin_step": 0.10,
    "use_tta": True,
    "tta_mode": "soft",
}

SEUIL_VALIDATION_SEMANTIQUE = 0.5

LIMITE_IMAGES = None   # None = toutes les images ; mettre un entier pour un test rapide

# --- Parametres de visualisation overlay ---
COULEUR_OVERLAY_BGR = (60, 220, 60)    # vert (format B, G, R pour OpenCV)
ALPHA_OVERLAY = 0.55                    # opacite du masque (0=invisible, 1=opaque)

RESULTAT_ANCIEN_REFERENCE = {
    "configuration": "tta_soft_seul",
    "Precision_chambre": 0.6265,
    "Recall_chambre": 0.7254,
    "F1_chambre": 0.6723,
    "IoU_chambre": 0.5064,
}


# ============================================================
# CHARGEMENT FASTER R-CNN
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
# CHARGEMENT DEEPLABV3+
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
# COCO / GT
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
# FASTER R-CNN : inference
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
# CROP AVEC MARGE
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
# DEEPLABV3+ : inference avec TTA soft
# ============================================================

def inference_deeplab_logits(deeplab_model, crop):
    from mmseg.apis import inference_model
    result = inference_model(deeplab_model, crop)
    return result.seg_logits.data


def inference_deeplab_tta_soft(deeplab_model, crop):
    logits_normal = inference_deeplab_logits(deeplab_model, crop)

    crop_flip = cv2.flip(crop, 1)
    logits_flip = inference_deeplab_logits(deeplab_model, crop_flip)
    logits_flip_retour = torch.flip(logits_flip, dims=[-1])

    if logits_flip_retour.shape != logits_normal.shape:
        logits_flip_retour = torch.nn.functional.interpolate(
            logits_flip_retour.unsqueeze(0),
            size=logits_normal.shape[-2:],
            mode="nearest"
        ).squeeze(0)

    logits_moyen = (logits_normal + logits_flip_retour) / 2.0
    mask_final = torch.argmax(logits_moyen, dim=0).cpu().numpy().astype(np.uint8)
    return mask_final


def extraire_crop_adaptatif_deeplab(image, box, deeplab_model, cfg):
    margin = cfg["margin_init"]
    dernier_crop, dernier_crop_box, dernier_mask = None, None, None

    while margin <= cfg["margin_max"]:
        crop, crop_box = extraire_crop_avec_marge(image, box, margin)
        if crop is None:
            break

        if cfg["use_tta"] and cfg["tta_mode"] == "soft":
            mask_crop = inference_deeplab_tta_soft(deeplab_model, crop)
        else:
            logits = inference_deeplab_logits(deeplab_model, crop)
            mask_crop = torch.argmax(logits, dim=0).cpu().numpy().astype(np.uint8)

        dernier_crop, dernier_crop_box, dernier_mask = crop, crop_box, mask_crop

        if not masque_touche_le_bord(mask_crop):
            return crop, crop_box, mask_crop

        margin += cfg["margin_step"]

    return dernier_crop, dernier_crop_box, dernier_mask


# ============================================================
# ETAPE 1 : COLLECTE DE TOUTES LES DETECTIONS
# ============================================================

def collecter_detections(faster_model, deeplab_model, images, annotations_by_image):
    print()
    print("=" * 70)
    print("ETAPE 1 : COLLECTE FASTER R-CNN + DEEPLABV3+")
    print("=" * 70)

    detections_completes = []
    gt_masks = {}

    liste_ids = list(images.keys())
    if LIMITE_IMAGES is not None:
        liste_ids = liste_ids[:LIMITE_IMAGES]

    id_global = 0

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
        gt_masks[image_id] = gt_mask

        detections = inference_faster(faster_model, image, CONFIG_DEEPLAB["score_th"])

        for detection in detections:
            box = detection["box"]

            crop, crop_box, mask_deeplab_crop = extraire_crop_adaptatif_deeplab(
                image, box, deeplab_model, CONFIG_DEEPLAB
            )
            if crop is None or mask_deeplab_crop is None:
                continue

            x1, y1, x2, y2 = crop_box
            crop_h, crop_w = y2 - y1, x2 - x1
            if mask_deeplab_crop.shape != (crop_h, crop_w):
                mask_deeplab_crop = cv2.resize(
                    mask_deeplab_crop, (crop_w, crop_h),
                    interpolation=cv2.INTER_NEAREST
                )

            box_locale = [
                box[0] - x1, box[1] - y1,
                box[2] - x1, box[3] - y1
            ]

            crop_filename = f"crop_{id_global}.png"
            crop_path = os.path.join(TEMP_CROPS_DIR, crop_filename)
            cv2.imwrite(crop_path, crop)

            detections_completes.append({
                "id": id_global,
                "image_id": image_id,
                "crop_box": crop_box,
                "box_locale": box_locale,
                "crop_path": crop_path,
                "mask_deeplab_crop": mask_deeplab_crop,
            })

            id_global += 1

        if (index + 1) % 5 == 0 or (index + 1) == len(liste_ids):
            print(f"  [{index + 1}/{len(liste_ids)}] images traitees "
                  f"({id_global} detections collectees)")

    print(f"Total detections collectees : {id_global}")
    return detections_completes, gt_masks


# ============================================================
# ETAPE 2 : APPEL DE SAM2 VIA SUBPROCESS
# ============================================================

def appeler_sam2(detections_completes):
    print()
    print("=" * 70)
    print("ETAPE 2 : APPEL DE SAM2 (subprocess vers sam2_env)")
    print("=" * 70)

    requetes = [
        {
            "id": d["id"],
            "crop_path": d["crop_path"],
            "box_locale": d["box_locale"]
        }
        for d in detections_completes
    ]

    with open(REQUETES_JSON, "w", encoding="utf-8") as f:
        json.dump(requetes, f, indent=2)

    print(f"Nombre de requetes envoyees a SAM2 : {len(requetes)}")

    commande = [
        SAM2_ENV_PYTHON,
        SAM2_WORKER_SCRIPT,
        "--requetes", REQUETES_JSON,
        "--resultats", RESULTATS_JSON,
        "--masques_dir", TEMP_MASQUES_DIR,
        "--sam2_repo", SAM2_REPO_PATH,
        "--checkpoint", SAM2_CHECKPOINT,
        "--config", SAM2_CONFIG,
    ]

    print("Commande executee :")
    print(" ".join(commande))
    print()

    resultat_process = subprocess.run(
        commande,
        capture_output=True,
        text=True
    )

    print("--- Sortie standard de sam2_worker.py ---")
    print(resultat_process.stdout)

    if resultat_process.returncode != 0:
        print("--- Erreur (stderr) ---")
        print(resultat_process.stderr)
        raise RuntimeError("L'appel a SAM2 (sam2_env) a echoue. Voir le detail ci-dessus.")

    with open(RESULTATS_JSON, "r", encoding="utf-8") as f:
        resultats_sam2 = json.load(f)

    resultats_par_id = {r["id"]: r for r in resultats_sam2}
    print(f"Resultats SAM2 recus : {len(resultats_par_id)}")

    return resultats_par_id


# ============================================================
# ETAPE 3 : FUSION SEMANTIQUE
# ============================================================

def fusionner_sam_deeplab(masque_sam_crop, mask_deeplab_crop,
                            classe_chambre=DEEPLAB_CHAMBRE_ID,
                            seuil_validation=SEUIL_VALIDATION_SEMANTIQUE):
    if masque_sam_crop.shape != mask_deeplab_crop.shape:
        masque_sam_crop = cv2.resize(
            masque_sam_crop.astype(np.uint8),
            (mask_deeplab_crop.shape[1], mask_deeplab_crop.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    zone_sam = masque_sam_crop.astype(bool)
    total_pixels_sam = zone_sam.sum()

    if total_pixels_sam == 0:
        return None, 0.0

    pixels_chambre = (mask_deeplab_crop[zone_sam] == classe_chambre).sum()
    proportion_chambre = pixels_chambre / total_pixels_sam

    if proportion_chambre >= seuil_validation:
        resultat = np.zeros_like(mask_deeplab_crop)
        resultat[zone_sam] = classe_chambre
        return resultat, proportion_chambre
    else:
        return None, proportion_chambre


def construire_masques_finaux(detections_completes, resultats_sam2, images):
    print()
    print("=" * 70)
    print("ETAPE 3 : FUSION SEMANTIQUE ET CONSTRUCTION DES MASQUES FINAUX")
    print("=" * 70)

    masques_finaux = {}
    for image_id, image_info in images.items():
        height, width = image_info["height"], image_info["width"]
        masques_finaux[image_id] = np.zeros((height, width), dtype=np.uint8)

    nb_validees = 0
    nb_rejetees = 0

    for detection in detections_completes:
        id_det = detection["id"]
        image_id = detection["image_id"]
        crop_box = detection["crop_box"]
        mask_deeplab_crop = detection["mask_deeplab_crop"]

        resultat_sam = resultats_sam2.get(id_det)
        if resultat_sam is None or resultat_sam.get("mask_path") is None:
            nb_rejetees += 1
            continue
 
        masque_sam_crop = cv2.imread(resultat_sam["mask_path"], cv2.IMREAD_GRAYSCALE)
        if masque_sam_crop is None:
            nb_rejetees += 1
            continue
        masque_sam_crop = (masque_sam_crop > 127).astype(np.uint8)

        masque_valide, proportion = fusionner_sam_deeplab(masque_sam_crop, mask_deeplab_crop)

        x1, y1, x2, y2 = crop_box
        final_mask = masques_finaux[image_id]

        if masque_valide is not None:
            region = final_mask[y1:y2, x1:x2]
            region[masque_valide == DEEPLAB_CHAMBRE_ID] = DEEPLAB_CHAMBRE_ID
            region[(mask_deeplab_crop == DEEPLAB_HARD_NEGATIVE_ID) &
                   (region == DEEPLAB_BACKGROUND_ID)] = DEEPLAB_HARD_NEGATIVE_ID
            final_mask[y1:y2, x1:x2] = region
            nb_validees += 1
        else:
            nb_rejetees += 1

    print(f"Detections validees semantiquement : {nb_validees}")
    print(f"Detections rejetees (proportion chambre trop faible) : {nb_rejetees}")

    return masques_finaux


# ============================================================
# METRIQUES
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


def evaluer_masques_finaux(masques_finaux, gt_masks):
    print()
    print("=" * 70)
    print("ETAPE 4 : CALCUL DES METRIQUES FINALES")
    print("=" * 70)

    confusion_totale = np.zeros((3, 3), dtype=np.int64)

    for image_id, gt_mask in gt_masks.items():
        pred_mask = masques_finaux[image_id]
        confusion_totale += calculer_matrice_confusion(gt_mask, pred_mask, num_classes=3)

    metrics = calculer_metriques(confusion_totale)
    pixel_accuracy = calculer_pixel_accuracy(confusion_totale)

    valid_ious = [metrics[i]["IoU"] for i in range(3) if not np.isnan(metrics[i]["IoU"])]
    miou = np.mean(valid_ious) if valid_ious else np.nan

    resultat = {
        "configuration": "faster_sam2_deeplab_final",
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
    print("--- Resultats finaux (Faster R-CNN + SAM2 + DeepLabV3+) ---")
    for cle, valeur in resultat.items():
        if isinstance(valeur, float):
            print(f"  {cle:<20}: {valeur:.4f}")
        else:
            print(f"  {cle:<20}: {valeur}")

    return resultat


def exporter_csv(resultat, chemin_csv):
    with open(chemin_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(resultat.keys()))
        writer.writeheader()
        writer.writerow(resultat)
    print()
    print("Resultats exportes vers :", os.path.abspath(chemin_csv))


# ============================================================
# VISUALISATION (overlay transparent sur l'image originale)
# ============================================================

def appliquer_overlay(image, mask, classe_cible=DEEPLAB_CHAMBRE_ID,
                        couleur=COULEUR_OVERLAY_BGR, alpha=ALPHA_OVERLAY):
    """
    Superpose un masque colore en semi-transparence sur l'image originale.
    Seuls les pixels ou mask == classe_cible sont colores, la texture
    du sol reste visible en transparence (comme dans vos exemples).
    """
    overlay = image.copy()
    zone = (mask == classe_cible)

    couche_couleur = np.zeros_like(image)
    couche_couleur[:, :] = couleur

    melange = cv2.addWeighted(image, 1 - alpha, couche_couleur, alpha, 0)
    overlay[zone] = melange[zone]

    return overlay


def visualiser_gt_et_prediction(image, gt_mask, pred_mask, nom_base):
    """
    Genere deux fichiers JPG :
      - <nom_base>_gt.jpg      : overlay du Ground Truth
      - <nom_base>_overlay.jpg : overlay de la prediction finale
    """
    image_gt_overlay = appliquer_overlay(image, gt_mask)
    image_pred_overlay = appliquer_overlay(image, pred_mask)

    chemin_gt = os.path.join(VIZ_DIR, f"{nom_base}_gt.jpg")
    chemin_pred = os.path.join(VIZ_DIR, f"{nom_base}_overlay.jpg")

    cv2.imwrite(chemin_gt, image_gt_overlay)
    cv2.imwrite(chemin_pred, image_pred_overlay)

    print(f"  Genere : {os.path.basename(chemin_gt)}  |  {os.path.basename(chemin_pred)}")


def visualiser_toutes_images(images, gt_masks, masques_finaux, detections_completes=None):
    """
    Genere les overlays GT + prediction pour chaque image du dataset.
    """
    print()
    print("=" * 70)
    print("GENERATION DES VISUALISATIONS (overlay transparent)")
    print("=" * 70)

    for image_id, gt_mask in gt_masks.items():
        image_info = images[image_id]
        file_name = image_info["file_name"]
        image_path = os.path.join(IMAGE_DIR, file_name)

        image = cv2.imread(image_path)
        if image is None:
            continue

        pred_mask = masques_finaux[image_id]
        nom_base = os.path.splitext(file_name)[0]

        visualiser_gt_et_prediction(image, gt_mask, pred_mask, nom_base)

    print(f"Toutes les visualisations sont disponibles dans : {VIZ_DIR}")


def visualiser_comparaison_globale(resultat_nouveau, resultat_ancien=None):
    """
    Genere un graphique en barres comparant les metriques principales
    entre l'ancienne meilleure configuration et la nouvelle.
    """
    if resultat_ancien is None:
        resultat_ancien = RESULTAT_ANCIEN_REFERENCE

    metriques = ["Precision_chambre", "Recall_chambre", "F1_chambre", "IoU_chambre"]
    labels = ["Precision", "Recall", "F1", "IoU"]

    valeurs_ancien = [resultat_ancien[m] for m in metriques]
    valeurs_nouveau = [resultat_nouveau[m] for m in metriques]

    x = np.arange(len(labels))
    largeur = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    barres1 = ax.bar(x - largeur / 2, valeurs_ancien, largeur,
                       label=resultat_ancien["configuration"], color="#888888")
    barres2 = ax.bar(x + largeur / 2, valeurs_nouveau, largeur,
                       label=resultat_nouveau["configuration"], color="#2E8B57")

    ax.set_ylabel("Score")
    ax.set_title("Comparaison des metriques : avant / apres integration SAM2")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for barres in [barres1, barres2]:
        for barre in barres:
            hauteur = barre.get_height()
            ax.annotate(f"{hauteur:.3f}",
                        xy=(barre.get_x() + barre.get_width() / 2, hauteur),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=9)

    plt.tight_layout()
    chemin_sortie = os.path.join(VIZ_DIR, "comparaison_globale_metriques.png")
    plt.savefig(chemin_sortie, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"Graphique comparatif global sauvegarde : {chemin_sortie}")


# ============================================================
# PREPARATION DOSSIERS TEMPORAIRES
# ============================================================

def preparer_dossiers_temp():
    if os.path.exists(TEMP_IPC_DIR):
        shutil.rmtree(TEMP_IPC_DIR)
    os.makedirs(TEMP_CROPS_DIR, exist_ok=True)
    os.makedirs(TEMP_MASQUES_DIR, exist_ok=True)


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 70)
    print("PIPELINE FINAL : FASTER R-CNN + SAM2 + DEEPLABV3+")
    print("=" * 70)

    preparer_dossiers_temp()

    faster_model = charger_faster_rcnn()
    deeplab_model = charger_deeplab()

    coco, images, annotations_by_image, categories = charger_coco(COCO_JSON)

    detections_completes, gt_masks = collecter_detections(
        faster_model, deeplab_model, images, annotations_by_image
    )

    resultats_sam2 = appeler_sam2(detections_completes)

    masques_finaux = construire_masques_finaux(detections_completes, resultats_sam2, images)

    resultat = evaluer_masques_finaux(masques_finaux, gt_masks)
    exporter_csv(resultat, os.path.join(OUTPUT_DIR, "resultat_final.csv"))

    # --- Visualisations ---
    visualiser_toutes_images(images, gt_masks, masques_finaux, detections_completes)
    visualiser_comparaison_globale(resultat)

    print()
    print("=" * 70)
    print("PIPELINE TERMINE")
    print("=" * 70)
    print()
    print(f"Toutes les visualisations sont disponibles dans : {VIZ_DIR}")


if __name__ == "__main__":
    main()

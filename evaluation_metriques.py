import os
import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# MASQUES GROUND TRUTH
# ------------------------------------------------------------

GT_MASK_DIR = "./dataset_test_hyb/masks"


# ------------------------------------------------------------
# MASQUES PREDITS PAR LE SYSTEME HYBRIDE
# ------------------------------------------------------------

PRED_MASK_DIR = "./resultats_hyb2/pred_masks"


# ------------------------------------------------------------
# NOMBRE DE CLASSES
# ------------------------------------------------------------

NUM_CLASSES = 3


# ------------------------------------------------------------
# NOMS DES CLASSES
# ------------------------------------------------------------

CLASS_NAMES = {
    0: "background",
    1: "hard_negative",
    2: "chambre_telecom"
}


# ============================================================
# AFFICHAGE CONFIGURATION
# ============================================================

print()
print("=" * 75)
print("EVALUATION DES MASQUES HYBRIDES")
print("=" * 75)

print()
print("Ground Truth :")
print(os.path.abspath(GT_MASK_DIR))

print()
print("Predictions :")
print(os.path.abspath(PRED_MASK_DIR))


# ============================================================
# VERIFICATION DES DOSSIERS
# ============================================================

if not os.path.isdir(GT_MASK_DIR):

    raise RuntimeError(
        f"\nERREUR : dossier GT introuvable :\n"
        f"{os.path.abspath(GT_MASK_DIR)}"
    )


if not os.path.isdir(PRED_MASK_DIR):

    raise RuntimeError(
        f"\nERREUR : dossier predictions introuvable :\n"
        f"{os.path.abspath(PRED_MASK_DIR)}"
    )


# ============================================================
# LISTE DES MASQUES
# ============================================================

gt_files = sorted([
    f
    for f in os.listdir(GT_MASK_DIR)
    if f.lower().endswith(".png")
])


pred_files = sorted([
    f
    for f in os.listdir(PRED_MASK_DIR)
    if f.lower().endswith(".png")
])


print()
print("=" * 75)
print("VERIFICATION DES FICHIERS")
print("=" * 75)

print()
print("Nombre de masques GT         :", len(gt_files))
print("Nombre de masques predictions:", len(pred_files))


# ============================================================
# CORRESPONDANCE DES FICHIERS
# ============================================================

gt_set = set(gt_files)
pred_set = set(pred_files)


missing_predictions = sorted(
    gt_set - pred_set
)


extra_predictions = sorted(
    pred_set - gt_set
)


print()
print("Predictions manquantes :", len(missing_predictions))
print("Predictions supplementaires :", len(extra_predictions))


if missing_predictions:

    print()
    print("Fichiers GT sans prediction :")

    for filename in missing_predictions:

        print("  -", filename)


if extra_predictions:

    print()
    print("Predictions sans GT :")

    for filename in extra_predictions:

        print("  -", filename)


# ============================================================
# MATRICE DE CONFUSION
# ============================================================

total_confusion = np.zeros(
    (NUM_CLASSES, NUM_CLASSES),
    dtype=np.int64
)


# ============================================================
# COMPTEURS
# ============================================================

images_evaluees = 0


# ============================================================
# EVALUATION IMAGE PAR IMAGE
# ============================================================

print()
print("=" * 75)
print("EVALUATION IMAGE PAR IMAGE")
print("=" * 75)


for index, filename in enumerate(gt_files):

    gt_path = os.path.join(
        GT_MASK_DIR,
        filename
    )

    pred_path = os.path.join(
        PRED_MASK_DIR,
        filename
    )


    # --------------------------------------------------------
    # Prediction inexistante
    # --------------------------------------------------------

    if not os.path.isfile(pred_path):

        print(
            f"[{index + 1}/{len(gt_files)}] "
            f"{filename} -> prediction absente"
        )

        continue


    # --------------------------------------------------------
    # Lecture GT
    # --------------------------------------------------------

    gt = cv2.imread(
        gt_path,
        cv2.IMREAD_GRAYSCALE
    )


    # --------------------------------------------------------
    # Lecture prediction
    # --------------------------------------------------------

    pred = cv2.imread(
        pred_path,
        cv2.IMREAD_GRAYSCALE
    )


    if gt is None:

        print(
            f"[{index + 1}/{len(gt_files)}] "
            f"{filename} -> erreur lecture GT"
        )

        continue


    if pred is None:

        print(
            f"[{index + 1}/{len(gt_files)}] "
            f"{filename} -> erreur lecture prediction"
        )

        continue


    # --------------------------------------------------------
    # Verification dimensions
    # --------------------------------------------------------

    if gt.shape != pred.shape:

        print()
        print(
            "ATTENTION : dimensions différentes"
        )

        print(
            "Fichier :", filename
        )

        print(
            "GT      :", gt.shape
        )

        print(
            "Prediction :", pred.shape
        )

        # On remet la prediction à la taille du GT
        pred = cv2.resize(
            pred,
            (
                gt.shape[1],
                gt.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        )


    # --------------------------------------------------------
    # Vérification des valeurs
    # --------------------------------------------------------

    gt_values = np.unique(gt)

    pred_values = np.unique(pred)


    # --------------------------------------------------------
    # Matrice de confusion
    # --------------------------------------------------------

    gt_flat = gt.flatten()
    pred_flat = pred.flatten()


    valid = (
        (gt_flat >= 0)
        &
        (gt_flat < NUM_CLASSES)
        &
        (pred_flat >= 0)
        &
        (pred_flat < NUM_CLASSES)
    )


    gt_valid = gt_flat[valid]
    pred_valid = pred_flat[valid]


    confusion = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )


    np.add.at(
        confusion,
        (gt_valid, pred_valid),
        1
    )


    total_confusion += confusion


    images_evaluees += 1


    # --------------------------------------------------------
    # Progression
    # --------------------------------------------------------

    print(
        f"[{index + 1}/{len(gt_files)}] "
        f"{filename} -> OK"
    )


# ============================================================
# FONCTION METRIQUES
# ============================================================

def calculer_metriques(confusion):

    metrics = {}


    for class_id in range(NUM_CLASSES):

        tp = confusion[
            class_id,
            class_id
        ]


        fp = (
            confusion[:, class_id].sum()
            - tp
        )


        fn = (
            confusion[class_id, :].sum()
            - tp
        )


        union = (
            tp
            + fp
            + fn
        )


        # ----------------------------------------------------
        # IoU
        # ----------------------------------------------------

        if union > 0:

            iou = tp / union

        else:

            iou = np.nan


        # ----------------------------------------------------
        # Precision
        # ----------------------------------------------------

        if tp + fp > 0:

            precision = (
                tp /
                (tp + fp)
            )

        else:

            precision = np.nan


        # ----------------------------------------------------
        # Recall
        # ----------------------------------------------------

        if tp + fn > 0:

            recall = (
                tp /
                (tp + fn)
            )

        else:

            recall = np.nan


        # ----------------------------------------------------
        # Dice
        # ----------------------------------------------------

        dice_denominator = (
            2 * tp
            + fp
            + fn
        )


        if dice_denominator > 0:

            dice = (
                2 * tp
                /
                dice_denominator
            )

        else:

            dice = np.nan


        # ----------------------------------------------------
        # F1
        # ----------------------------------------------------

        if (
            not np.isnan(precision)
            and
            not np.isnan(recall)
            and
            precision + recall > 0
        ):

            f1 = (
                2
                * precision
                * recall
                /
                (
                    precision
                    + recall
                )
            )

        else:

            f1 = np.nan


        metrics[class_id] = {

            "IoU": iou,

            "Dice": dice,

            "Precision": precision,

            "Recall": recall,

            "F1": f1,

            "TP": int(tp),

            "FP": int(fp),

            "FN": int(fn)
        }


    return metrics


# ============================================================
# CALCUL METRIQUES
# ============================================================

metrics = calculer_metriques(
    total_confusion
)


# ============================================================
# MATRICE DE CONFUSION
# ============================================================

print()
print()
print("=" * 75)
print("RESULTATS")
print("=" * 75)


print()
print("Images GT              :", len(gt_files))
print("Images évaluées        :", images_evaluees)
print(
    "Predictions manquantes :",
    len(missing_predictions)
)


print()
print("=" * 75)
print("MATRICE DE CONFUSION")
print("=" * 75)


print()

print(
    f"{'':20}"
    f"{'Pred background':>18}"
    f"{'Pred hard_negative':>20}"
    f"{'Pred chambre':>18}"
)


for i in range(NUM_CLASSES):

    print(
        f"{'GT ' + CLASS_NAMES[i]:20}"
        f"{total_confusion[i, 0]:18d}"
        f"{total_confusion[i, 1]:20d}"
        f"{total_confusion[i, 2]:18d}"
    )


# ============================================================
# METRIQUES PAR CLASSE
# ============================================================

print()
print("=" * 75)
print("METRIQUES PAR CLASSE")
print("=" * 75)


for class_id in range(NUM_CLASSES):

    m = metrics[class_id]


    print()
    print(
        f"Classe {class_id} : "
        f"{CLASS_NAMES[class_id]}"
    )


    print(
        f"  IoU       : {m['IoU']:.4f}"
    )


    print(
        f"  Dice      : {m['Dice']:.4f}"
    )


    print(
        f"  Precision : {m['Precision']:.4f}"
    )


    print(
        f"  Recall    : {m['Recall']:.4f}"
    )


    print(
        f"  F1        : {m['F1']:.4f}"
    )


    print(
        f"  TP        : {m['TP']}"
    )


    print(
        f"  FP        : {m['FP']}"
    )


    print(
        f"  FN        : {m['FN']}"
    )


# ============================================================
# PIXEL ACCURACY
# ============================================================

total_pixels = total_confusion.sum()

correct_pixels = np.trace(
    total_confusion
)


if total_pixels > 0:

    pixel_accuracy = (
        correct_pixels
        /
        total_pixels
    )

else:

    pixel_accuracy = np.nan


# ============================================================
# mIoU
# ============================================================

ious = []

dices = []

precisions = []

recalls = []

f1s = []


for class_id in range(NUM_CLASSES):

    iou = metrics[class_id]["IoU"]

    dice = metrics[class_id]["Dice"]

    precision = metrics[class_id]["Precision"]

    recall = metrics[class_id]["Recall"]

    f1 = metrics[class_id]["F1"]


    if not np.isnan(iou):

        ious.append(iou)


    if not np.isnan(dice):

        dices.append(dice)


    if not np.isnan(precision):

        precisions.append(precision)


    if not np.isnan(recall):

        recalls.append(recall)


    if not np.isnan(f1):

        f1s.append(f1)


miou = (
    np.mean(ious)
    if len(ious) > 0
    else np.nan
)


mean_dice = (
    np.mean(dices)
    if len(dices) > 0
    else np.nan
)


mean_precision = (
    np.mean(precisions)
    if len(precisions) > 0
    else np.nan
)


mean_recall = (
    np.mean(recalls)
    if len(recalls) > 0
    else np.nan
)


mean_f1 = (
    np.mean(f1s)
    if len(f1s) > 0
    else np.nan
)


# ============================================================
# RESULTATS GLOBAUX
# ============================================================

print()
print("=" * 75)
print("RESULTATS GLOBAUX")
print("=" * 75)


print(
    f"\nPixel Accuracy : "
    f"{pixel_accuracy:.4f}"
)


print(
    f"mIoU           : "
    f"{miou:.4f}"
)


print(
    f"Mean Dice      : "
    f"{mean_dice:.4f}"
)


print(
    f"Mean Precision : "
    f"{mean_precision:.4f}"
)


print(
    f"Mean Recall    : "
    f"{mean_recall:.4f}"
)


print(
    f"Mean F1        : "
    f"{mean_f1:.4f}"
)


# ============================================================
# RESULTATS CHAMBRE TELECOM
# ============================================================

chambre = metrics[
    2
]


print()
print("=" * 75)
print("RESULTATS CHAMBRE_TELECOM")
print("=" * 75)


print(
    f"\nIoU       : "
    f"{chambre['IoU']:.4f}"
)


print(
    f"Dice      : "
    f"{chambre['Dice']:.4f}"
)


print(
    f"Precision : "
    f"{chambre['Precision']:.4f}"
)


print(
    f"Recall    : "
    f"{chambre['Recall']:.4f}"
)


print(
    f"F1        : "
    f"{chambre['F1']:.4f}"
)


print(
    f"TP        : "
    f"{chambre['TP']}"
)


print(
    f"FP        : "
    f"{chambre['FP']}"
)


print(
    f"FN        : "
    f"{chambre['FN']}"
)


# ============================================================
# FIN
# ============================================================

print()
print("=" * 75)
print("FIN DE L'EVALUATION")
print("=" * 75)
print()
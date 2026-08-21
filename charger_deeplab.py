import torch

from mmseg.apis import init_model


# ============================================================
# CHEMINS
# ============================================================

CONFIG_FILE = "./deeplab_lyon (1).py"

CHECKPOINT_FILE = "./best_mIoU_iter_2000.pth"


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


print("=" * 70)
print("CHARGEMENT DEEPLABV3+")
print("=" * 70)

print()
print("Config    :", CONFIG_FILE)
print("Checkpoint:", CHECKPOINT_FILE)
print("Device    :", device)


# ============================================================
# CHARGEMENT
# ============================================================

print()
print("Construction du modèle à partir du config...")

model = init_model(
    CONFIG_FILE,
    CHECKPOINT_FILE,
    device=device
)


# ============================================================
# VERIFICATION
# ============================================================

print()
print("=" * 70)
print("MODELE CHARGE AVEC SUCCES")
print("=" * 70)

print()

print("Type du modèle :")
print(type(model))

print()

print("Classes :")

if hasattr(model, "dataset_meta"):

    dataset_meta = model.dataset_meta

    print(dataset_meta)

else:

    print("dataset_meta non disponible")


print()
print("Device du modèle :")
print(next(model.parameters()).device)
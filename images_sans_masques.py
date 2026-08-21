import os
import shutil

# ============================================================
# DOSSIERS
# ============================================================

TOTAL_DIR = "/mnt/c/Users/zatoui/dataset_chambres/dataset_lyon_high_res/images"

TRAIN_DIR = "/mnt/c/Users/zatoui/dataset_chambres/full_dataset_lyon/train/images"

TEST_DIR = "/mnt/c/Users/zatoui/dataset_chambres/full_dataset_lyon/test/images"

OUTPUT_DIR = "/mnt/c/Users/zatoui/dataset_chambres/images_sans_masques"


# ============================================================
# EXTENSIONS
# ============================================================

EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


# ============================================================
# VIDER LE DOSSIER DE SORTIE
# ============================================================

if os.path.exists(OUTPUT_DIR):

    for filename in os.listdir(OUTPUT_DIR):

        path = os.path.join(OUTPUT_DIR, filename)

        if os.path.isfile(path):
            os.remove(path)

        elif os.path.isdir(path):
            shutil.rmtree(path)

else:

    os.makedirs(OUTPUT_DIR)


print("=" * 70)
print("DOSSIER DE SORTIE NETTOYE")
print("=" * 70)


# ============================================================
# 1. IMAGES ORIGINALES
# ============================================================

total_images = {}

for filename in os.listdir(TOTAL_DIR):

    path = os.path.join(TOTAL_DIR, filename)

    if not os.path.isfile(path):
        continue

    name, ext = os.path.splitext(filename)

    if ext in EXTENSIONS:

        total_images[name] = filename


print()
print("Images totales :", len(total_images))


# ============================================================
# 2. IMAGES ANNOTEES DU TRAIN
# ============================================================

annotated_originals = set()

for filename in os.listdir(TRAIN_DIR):

    path = os.path.join(TRAIN_DIR, filename)

    if not os.path.isfile(path):
        continue

    name, ext = os.path.splitext(filename)

    if ext not in EXTENSIONS:
        continue

    # Exemple :
    #
    # orig_LYON_1001.jpg
    #
    # devient :
    #
    # LYON_1001

    if name.startswith("orig_"):

        original_name = name[len("orig_"):]

        annotated_originals.add(original_name)


# ============================================================
# 3. VERIFIER LES ANNOTATIONS QUI EXISTENT
#    DANS LE DATASET ORIGINAL
# ============================================================

annotated_found = set()

for original_name in annotated_originals:

    if original_name in total_images:

        annotated_found.add(original_name)


# ============================================================
# 4. IDENTIFIER UNIQUEMENT LES IMAGES SANS ANNOTATION
# ============================================================

images_without_masks = []

for original_name, filename in total_images.items():

    if original_name not in annotated_found:

        images_without_masks.append(filename)


# ============================================================
# 5. COPIER UNIQUEMENT CES IMAGES
# ============================================================

for filename in images_without_masks:

    source = os.path.join(TOTAL_DIR, filename)

    destination = os.path.join(OUTPUT_DIR, filename)

    shutil.copy2(source, destination)


# ============================================================
# 6. COMPTER CE QUI A REELLEMENT ETE COPIE
# ============================================================

copied_images = []

for filename in os.listdir(OUTPUT_DIR):

    path = os.path.join(OUTPUT_DIR, filename)

    if not os.path.isfile(path):
        continue

    name, ext = os.path.splitext(filename)

    if ext in EXTENSIONS:

        copied_images.append(filename)


# ============================================================
# 7. RESULTATS
# ============================================================

print()
print("=" * 70)
print("RESULTAT FINAL")
print("=" * 70)

print("Images totales        :", len(total_images))

print("Images annotées       :", len(annotated_found))

print("Images sans masques   :", len(images_without_masks))

print("Images copiées        :", len(copied_images))

print()
print("Calcul :")
print(
    len(total_images),
    "-",
    len(annotated_found),
    "=",
    len(total_images) - len(annotated_found)
)

print()
print("=" * 70)

if len(copied_images) == len(images_without_masks):

    print("✅ SUCCES")
    print("Le dossier contient exactement les images sans annotation.")

else:

    print("❌ ERREUR")
    print(
        "Attendu :",
        len(images_without_masks),
        "| Trouvé :",
        len(copied_images)
    )

print("=" * 70)
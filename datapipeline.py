import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

# Folder containing the road images
IMAGE_FOLDER = Path(
    r"D:\Sarthak's World\MetroVision_Artificial_Intelligence\RDD2022_India\India\train\images"
)

# Folder containing the XML annotation files
XML_FOLDER = Path(
    r"D:\Sarthak's World\MetroVision_Artificial_Intelligence\RDD2022_India\India\train\annotations\xmls"
)

# Folder where the new classified dataset will be created
OUTPUT_FOLDER = Path(
    r"D:\Sarthak's World\MetroVision_Artificial_Intelligence\explored_data"
)


# ============================================================
# CLASS DEFINITIONS
# ============================================================

# Numerical labels for the CNN
CLASS_MAPPING = {
    "Normal": 0,
    "Crack": 1,
    "Pothole": 2,
    "Both": 3
}

# Crack-related RDD labels that we WANT to recognize
CRACK_LABELS = {
    "D00",
    "D01",
    "D10",
    "D11",
    "D20"
}

# Pothole label
POTHOLE_LABEL = "D40"

# These labels are deliberately ignored.
# They will NOT be treated as cracks or potholes.
IGNORED_LABELS = {
    "D43",
    "D44",
    "D50"
}


# ============================================================
# CHECK PATHS
# ============================================================

if not IMAGE_FOLDER.exists():
    raise FileNotFoundError(
        f"Image folder does not exist:\n{IMAGE_FOLDER}"
    )

if not XML_FOLDER.exists():
    raise FileNotFoundError(
        f"XML folder does not exist:\n{XML_FOLDER}"
    )

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

output_folders = {}

for class_name in CLASS_MAPPING:

    folder = OUTPUT_FOLDER / class_name
    folder.mkdir(parents=True, exist_ok=True)

    output_folders[class_name] = folder

    print(f"Output folder ready: {folder}")


# ============================================================
# FUNCTION: CLASSIFY ONE XML FILE
# ============================================================

def classify_image_from_xml(xml_path):
    """
    Reads one Pascal VOC XML annotation file and determines
    whether the corresponding image is:

        Normal
        Crack
        Pothole
        Both

    D43, D44 and D50 are ignored completely.
    """

    try:

        tree = ET.parse(xml_path)
        root = tree.getroot()

    except ET.ParseError:
        print(f"\nWARNING: Could not parse XML: {xml_path}")
        return None, []

    crack_found = False
    pothole_found = False

    labels_found = []

    # Find every <object> in the XML
    objects = root.findall("object")

    for obj in objects:

        name_element = obj.find("name")

        if name_element is None:
            continue

        label = name_element.text

        if label is None:
            continue

        label = label.strip()

        labels_found.append(label)

        # ----------------------------------------------------
        # Pothole
        # ----------------------------------------------------

        if label == POTHOLE_LABEL:
            pothole_found = True

        # ----------------------------------------------------
        # Crack
        # ----------------------------------------------------

        elif label in CRACK_LABELS:
            crack_found = True

        # ----------------------------------------------------
        # D43, D44, D50
        # ----------------------------------------------------
        # Do absolutely nothing with these.
        #
        # They are intentionally ignored.
        # ----------------------------------------------------

        elif label in IGNORED_LABELS:
            continue

        # ----------------------------------------------------
        # Any completely unknown label
        # ----------------------------------------------------

        else:
            print(
                f"\nWARNING: Unknown label '{label}' "
                f"found in {xml_path.name}. Ignoring it."
            )

    # ========================================================
    # CLASSIFICATION LOGIC
    # ========================================================

    if crack_found and pothole_found:

        classification = "Both"

    elif pothole_found:

        classification = "Pothole"

    elif crack_found:

        classification = "Crack"

    else:

        # No recognized damage was found.
        # This includes:
        #
        # 1. XML with no <object>
        # 2. XML containing ONLY D43/D44/D50
        #
        classification = "Normal"

    return classification, labels_found


# ============================================================
# FIND XML FILES
# ============================================================

xml_files = sorted(XML_FOLDER.glob("*.xml"))

if len(xml_files) == 0:

    raise FileNotFoundError(
        f"No XML files found in:\n{XML_FOLDER}"
    )

print("\n" + "=" * 60)
print(f"Found {len(xml_files)} XML annotation files.")
print("=" * 60)


# ============================================================
# PROCESS DATASET
# ============================================================

image_classifications = {}

classification_records = []

image_counts = {
    "Normal": 0,
    "Crack": 0,
    "Pothole": 0,
    "Both": 0
}

missing_images = []
failed_xmls = []


print("\nProcessing XML annotations...\n")


for i, xml_file in enumerate(xml_files, start=1):

    # --------------------------------------------------------
    # Read the filename directly from XML if possible
    # --------------------------------------------------------

    try:

        tree = ET.parse(xml_file)
        root = tree.getroot()

        filename_element = root.find("filename")

        if filename_element is not None and filename_element.text:

            image_filename = filename_element.text.strip()

        else:

            # Fallback
            image_filename = xml_file.stem + ".jpg"

    except ET.ParseError:

        failed_xmls.append(xml_file.name)
        continue

    # --------------------------------------------------------
    # Classify image
    # --------------------------------------------------------

    classification, labels_found = classify_image_from_xml(xml_file)

    if classification is None:

        failed_xmls.append(xml_file.name)
        continue

    # --------------------------------------------------------
    # Locate corresponding image
    # --------------------------------------------------------

    source_image = IMAGE_FOLDER / image_filename

    if not source_image.is_file():

        missing_images.append(image_filename)

        continue

    # --------------------------------------------------------
    # Destination
    # --------------------------------------------------------

    destination_folder = output_folders[classification]

    destination_image = destination_folder / image_filename

    # --------------------------------------------------------
    # Copy image
    # --------------------------------------------------------

    try:

        shutil.copy2(
            source_image,
            destination_image
        )

    except Exception as e:

        print(
            f"\nERROR copying {image_filename}: {e}"
        )

        continue

    # --------------------------------------------------------
    # Record classification
    # --------------------------------------------------------

    image_classifications[image_filename] = classification

    image_counts[classification] += 1

    classification_records.append({
        "image_name": image_filename,
        "class": classification,
        "class_index": CLASS_MAPPING[classification],
        "original_labels": ", ".join(labels_found)
    })

    # Progress
    if i % 100 == 0 or i == len(xml_files):

        print(
            f"Processed {i}/{len(xml_files)} XML files..."
        )


# ============================================================
# CREATE LABELS CSV
# ============================================================

labels_df = pd.DataFrame(classification_records)

csv_path = OUTPUT_FOLDER / "labels.csv"

labels_df.to_csv(
    csv_path,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("DATASET CLASSIFICATION COMPLETE")
print("=" * 60)

print(f"\nTotal XML files       : {len(xml_files)}")
print(f"Successfully classified: {len(image_classifications)}")

print("\nClass distribution:")
print("-" * 40)

for class_name, count in image_counts.items():

    print(
        f"{class_name:<15}: {count}"
    )

print("-" * 40)

print(f"\nCSV saved to:")
print(csv_path)

if missing_images:

    print(
        f"\nWARNING: {len(missing_images)} images "
        f"were referenced by XML but not found."
    )

if failed_xmls:

    print(
        f"\nWARNING: {len(failed_xmls)} XML files "
        f"could not be processed."
    )

print("\nOutput folders:")

for class_name, folder in output_folders.items():

    print(
        f"{class_name:<15}: {folder}"
    )

print("\nDone.")
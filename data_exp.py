import os
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm

print("Libraries imported successfully.")

# ==============================
# DATASET PATHS
# ==============================

image_folder = r"D:\Sarthak's World\MetroVision_Artificial_Intelligence\RDD2022_India\India\train\images"

xml_annotation_folder = r"D:\Sarthak's World\MetroVision_Artificial_Intelligence\RDD2022_India\India\train\annotations\xmls"

output_root_folder = r"D:\Sarthak's World\MetroVision_Artificial_Intelligence\explored_data"

# ==============================

image_folder_path = Path(image_folder)
xml_folder_path = Path(xml_annotation_folder)
output_root_path = Path(output_root_folder)

print(f"Image folder: {image_folder_path}")
print(f"XML folder: {xml_folder_path}")
print(f"Output folder: {output_root_path}")

output_root_path.mkdir(parents=True, exist_ok=True)

print("Output folder created (or already exists).")


def get_unique_labels(xml_folder_path):

    unique_labels = set()

    xml_files = list(xml_folder_path.glob("*.xml"))

    if len(xml_files) == 0:
        print("No XML files found!")
        return unique_labels

    print(f"\nFound {len(xml_files)} XML files.\n")

    for xml_file in tqdm(xml_files):

        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            for obj in root.findall("object"):

                name_tag = obj.find("name")

                if name_tag is not None:

                    unique_labels.add(name_tag.text)

        except ET.ParseError:

            print(f"Could not parse: {xml_file.name}")

    return sorted(unique_labels)


labels = get_unique_labels(xml_folder_path)

print("\n=============================")
print("Unique labels")
print("=============================")

for label in labels:
    print(label)

print("=============================")
print(f"Total unique labels: {len(labels)}")
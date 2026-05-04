import json
import os
import subprocess

from pdf2image import convert_from_path

# 1. Setup paths
DOCX_DIR = "../15_Apr/vignettes_nature_paper/docs_new_indian"
PDF_DIR = "../15_Apr/vignettes_nature_paper/pdf_new_indian"
JPG_DIR = "../15_Apr/vignettes_nature_paper/jpg_new_indian"
MAPPING_FILE = "../15_Apr/vignettes_nature_paper/ground_truth_mapping_new_indian.json"

# Create output directories
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(JPG_DIR, exist_ok=True)

# 2. Load the mapping JSON
with open(MAPPING_FILE, "r") as f:
    mapping = json.load(f)

print(f"Loaded {len(mapping)} files from mapping. Starting conversion...")

# 3. Process each document
for docx_name, encoded_name in mapping.items():
    # The JSON keys don't have the .docx extension, so we add it
    docx_path = os.path.join(DOCX_DIR, f"{docx_name}.docx")

    if not os.path.exists(docx_path):
        print(f"File not found: {docx_path}, skipping.")
        continue

    # Setup output paths
    pdf_temp_path = os.path.join(PDF_DIR, f"{docx_name}.pdf")
    encoded_pdf_path = os.path.join(PDF_DIR, f"{encoded_name}.pdf")
    encoded_jpg_path = os.path.join(JPG_DIR, f"{encoded_name}.jpg")

    # --- STEP A: Convert DOCX to PDF using LibreOffice ---
    # We run libreoffice in headless mode via subprocess
    subprocess.run(
        [
            "libreoffice",
            "--headless",
            "--convert-to",
            "pdf",
            docx_path,
            "--outdir",
            PDF_DIR,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # LibreOffice saves it with the original name, so we rename it to the encoded name
    if os.path.exists(pdf_temp_path):
        os.rename(pdf_temp_path, encoded_pdf_path)
        print(f"Created PDF: {encoded_pdf_path}")

        # --- STEP B: Convert PDF to JPG ---
        # convert_from_path returns a list of images (one per page).
        # Our EHR is just 1 page, so we take index [0]
        pages = convert_from_path(encoded_pdf_path, dpi=300)
        if pages:
            pages[0].save(encoded_jpg_path, "JPEG")
            print(f"Created JPG: {encoded_jpg_path}")
    else:
        print(f"Failed to convert {docx_name}.docx to PDF.")

print("\nAll conversions completed successfully!")

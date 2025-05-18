import json
import os
import csv
import logging
import re
import base64
from io import BytesIO

# --- Required Libraries ---
try:
    import datasets
    from PIL import Image # Pillow library for image handling
    # google.cloud.storage is no longer needed
except ImportError as e:
    print(f"Error importing libraries: {e}")
    print("Please install required libraries: pip install datasets Pillow")
    exit()

# --- Configuration ---
# Input JSONL file (with the 70k Gemini-generated answers)
INPUT_JSONL_FILE = r"C:\Path\To\Your\train_or_validation_gemini_structured_answers.jsonl" # ADJUST THIS PATH

# --- >> NEW: Path to the BASE directory containing your LOCAL image folders << ---
# Assumes structure like: LOCAL_IMAGE_BASE_DIR/Tomato___healthy/image1.jpg
LOCAL_IMAGE_BASE_DIR = r"C:\Path\To\Your\Downloaded_Images\train" # ADJUST THIS PATH (e.g., point to your local 'train' or 'valid' folder)

# Output directory path (where the Hugging Face dataset will be saved)
OUTPUT_HF_DATASET_DIR = r"C:\Path\To\Your\hf_plant_disease_dataset" # ADJUST THIS PATH

# Fixed text for the 'question' column
FIXED_QUESTION_TEXT = "Analyze the plant disease"

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---
def normalize_label(label):
    """Converts label to a standard format: lowercase, spaces, no underscores."""
    if not label: return ""
    label = label.lower(); label = re.sub(r'_+', ' ', label); label = re.sub(r'\s+', ' ', label); label = label.strip()
    return label

def extract_class_and_filename_from_uri(gcs_uri):
    """
    Extracts the class name and filename from the GCS URI.
    Example: gs://bucket/path/ClassName/filename.jpg -> (ClassName, filename.jpg)
    """
    if not gcs_uri:
        return None, None
    try:
        # Split the path part after the bucket name
        path_part = gcs_uri.split('/', 3)[-1] # Get everything after gs://bucket/
        # Split by forward slash
        parts = path_part.split('/')
        # Class is second-to-last, filename is last
        if len(parts) >= 2:
            class_name = parts[-2]
            filename = parts[-1]
            return class_name, filename
        else:
            logging.warning(f"Could not determine class/filename from URI structure: {gcs_uri}")
            return None, None
    except Exception as e:
        logging.error(f"Error parsing class/filename from URI {gcs_uri}: {e}")
        return None, None

# --- Main Conversion Logic ---
logging.info(f"--- Starting Conversion to Hugging Face Dataset Format from Local Files ---")
logging.info(f"Input JSONL: {INPUT_JSONL_FILE}")
logging.info(f"Local Image Base Directory: {LOCAL_IMAGE_BASE_DIR}")
logging.info(f"Output Directory: {OUTPUT_HF_DATASET_DIR}")

# Accumulate data in lists
data_dict = {
    "image": [],
    "question": [],
    "answer": [],
    "class": []
}

lines_processed = 0
errors_skipped = 0

try:
    with open(INPUT_JSONL_FILE, 'r', encoding='utf-8') as infile:
        for line in infile:
            lines_processed += 1
            if not line.strip():
                continue

            try:
                data = json.loads(line)

                # Initialize variables
                gcs_uri = None
                answer_text = None
                class_label = None
                filename = None
                image_object = None # Will hold PIL Image

                # Extract GCS URI (still needed to find class/filename)
                user_content = data.get('contents', [{}])[0]
                user_parts = user_content.get('parts', [])
                for part in user_parts:
                    file_data = part.get('fileData', {})
                    if file_data and 'fileUri' in file_data:
                        gcs_uri = file_data['fileUri']
                        break

                # Extract Answer Text
                model_content = data.get('contents', [{}, {}])[1]
                model_parts = model_content.get('parts', [{}])
                if model_parts:
                    answer_text = model_parts[0].get('text')

                # Process only if URI and Answer are found
                if gcs_uri and answer_text:
                    # Extract Class Label and Filename from URI
                    class_label, filename = extract_class_and_filename_from_uri(gcs_uri)
                    if not class_label or not filename:
                        logging.warning(f"Skipping line {lines_processed}: Could not extract class/filename from URI {gcs_uri}")
                        errors_skipped += 1
                        continue

                    # --- Load Image from LOCAL Path ---
                    local_image_path = os.path.join(LOCAL_IMAGE_BASE_DIR, class_label, filename)

                    try:
                        # Check if file exists locally
                        if not os.path.exists(local_image_path):
                            logging.warning(f"Skipping line {lines_processed}: Local image file not found at {local_image_path} (derived from URI {gcs_uri})")
                            errors_skipped += 1
                            continue

                        # Open the local image file
                        image_object = Image.open(local_image_path)
                        # Ensure image data is loaded, prevent "too many open files"
                        image_object.load()
                        # Optional: Convert to RGB if needed
                        # image_object = image_object.convert("RGB")
                        logging.debug(f"Loaded local image: {local_image_path}")

                    except FileNotFoundError:
                         logging.warning(f"Skipping line {lines_processed}: Local image file not found at {local_image_path}")
                         errors_skipped += 1
                         continue
                    except Exception as img_e:
                         logging.error(f"Skipping line {lines_processed}: Failed to open local image {local_image_path}: {img_e}")
                         errors_skipped += 1
                         continue
                    # --- End Local Image Loading ---

                    # If image loaded successfully, add data to lists
                    if image_object:
                        data_dict["image"].append(image_object)
                        data_dict["question"].append(FIXED_QUESTION_TEXT)
                        data_dict["answer"].append(answer_text)
                        data_dict["class"].append(normalize_label(class_label)) # Normalize class label

                else:
                    logging.warning(f"Skipping line {lines_processed}: Missing GCS URI or Answer Text in JSONL.")
                    errors_skipped += 1

            except json.JSONDecodeError:
                logging.warning(f"Skipping line {lines_processed}: Invalid JSON.")
                errors_skipped += 1
            except Exception as e:
                logging.error(f"Error processing line {lines_processed}: {e}", exc_info=False)
                errors_skipped += 1

            if lines_processed % 1000 == 0:
                logging.info(f"Processed {lines_processed} lines...")

except FileNotFoundError:
    logging.error(f"Input file not found: {INPUT_JSONL_FILE}")
    exit()
except Exception as e:
    logging.error(f"An unexpected error occurred during file reading/processing: {e}", exc_info=True)
    exit()

# --- Create and Save Hugging Face Dataset ---
if data_dict["image"]:
    logging.info(f"Creating Hugging Face Dataset with {len(data_dict['image'])} entries...")
    try:
        # Define the features
        features = datasets.Features({
            "image": datasets.Image(),
            "question": datasets.Value("string"),
            "answer": datasets.Value("string"),
            "class": datasets.Value("string")
        })

        # Create the dataset
        hf_dataset = datasets.Dataset.from_dict(data_dict, features=features)
        logging.info(f"Dataset created: {hf_dataset}")

        # Save the dataset to disk
        logging.info(f"Saving dataset to disk at: {OUTPUT_HF_DATASET_DIR}")
        # Make sure the output directory exists
        os.makedirs(OUTPUT_HF_DATASET_DIR, exist_ok=True)
        hf_dataset.save_to_disk(OUTPUT_HF_DATASET_DIR)
        logging.info("Dataset successfully saved.")
        logging.info(f"You can now upload the contents of the '{OUTPUT_HF_DATASET_DIR}' directory to the Hugging Face Hub.")

    except Exception as e:
        logging.error(f"Failed to create or save the Hugging Face dataset: {e}", exc_info=True)
else:
    logging.warning("No data was successfully processed to create a dataset.")

logging.info("--- Conversion Finished ---")
logging.info(f"Total lines read: {lines_processed}")
logging.info(f"Rows converted to Dataset entries: {len(data_dict['image'])}")
logging.info(f"Lines skipped due to errors or missing files: {errors_skipped}")
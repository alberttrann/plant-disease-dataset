import os
import random
import json
import math
import logging
from collections import defaultdict, Counter

# --- Configuration ---

# --- Input Data ---
# Path to the original, LARGE validation set directory on your local machine
# Assumes structure like: .../validation_original/DiseaseA/img1.jpg
ORIGINAL_VALID_DIR = r"C:\Users\alberttran\Downloads\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\valid" # ADJUST THIS PATH

# --- Output Data ---
# Where to save the input file for the *label generation* script
OUTPUT_JSONL_PATH = r"C:\Users\alberttran\Downloads\validation_gemini_format_sampled_5k.jsonl"
# Optional: Where to save a simple list of the sampled file paths (for reference)
SAMPLED_FILES_LIST_PATH = r"C:\Users\alberttran\Downloads\validation_sampled_5k_filepaths.txt"

# --- Sampling Configuration ---
TARGET_SAMPLE_SIZE = 5000

# --- GCS Configuration (Needed to build URIs for the JSONL) ---
# Assumes your validation images are ALSO in GCS mirroring the local structure
GCS_BUCKET = "plantdiseasee" # Your GCS bucket name
# The GCS path prefix *corresponding* to ORIGINAL_VALID_DIR
# e.g., if ORIGINAL_VALID_DIR is 'C:\Data\valid' which maps to 'gs://bucket/dataset/valid'
# then GCS_VALID_PREFIX should be 'dataset/valid' (no leading/trailing slashes)
GCS_VALID_PREFIX = "plant_disease_dataset/valid" # ADJUST IF NEEDED

# --- Label Generation Script Input Format ---
# This should match the format your *existing* generation script expects
USER_PROMPT_TEXT = "Identify the plant disease shown in the image." # Or whatever you used

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Function ---
def get_mime_type(filename):
    """Basic function to guess mime type from extension."""
    ext = filename.lower().split('.')[-1]
    if ext == 'jpg' or ext == 'jpeg':
        return 'image/jpeg'
    elif ext == 'png':
        return 'image/png'
    else:
        return 'application/octet-stream' # Default or raise error

# --- Main Script Logic ---

logging.info("--- Starting Stratified Sampling and Input JSONL Creation ---")

# 1. Scan original validation directory and group files by class
logging.info(f"Scanning original validation directory: {ORIGINAL_VALID_DIR}")
class_files = defaultdict(list)
total_original_files = 0
valid_image_extensions = ('.png', '.jpg', '.jpeg')

if not os.path.isdir(ORIGINAL_VALID_DIR):
    logging.error(f"Directory not found: {ORIGINAL_VALID_DIR}")
    exit()

for class_name in os.listdir(ORIGINAL_VALID_DIR):
    class_dir = os.path.join(ORIGINAL_VALID_DIR, class_name)
    if os.path.isdir(class_dir):
        count_in_class = 0
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(valid_image_extensions):
                file_path = os.path.join(class_dir, filename)
                class_files[class_name].append(file_path)
                count_in_class += 1
        if count_in_class > 0:
            total_original_files += count_in_class
            logging.info(f"Found {count_in_class} images for class '{class_name}'")
        else:
            logging.warning(f"No images found in directory: {class_dir}")

if not class_files:
    logging.error("No image classes found in the specified directory.")
    exit()

logging.info(f"Found {total_original_files} total images across {len(class_files)} classes.")

if total_original_files < TARGET_SAMPLE_SIZE:
    logging.warning(f"Original dataset size ({total_original_files}) is smaller than target sample size ({TARGET_SAMPLE_SIZE}). Will use all original files.")
    TARGET_SAMPLE_SIZE = total_original_files # Adjust target if needed

# 2. Determine sample counts per class (Stratified Sampling Logic)
logging.info(f"Calculating samples per class for a target size of {TARGET_SAMPLE_SIZE}...")
samples_per_class = {}
initial_total = 0
class_proportions = {label: len(files) / total_original_files for label, files in class_files.items()}
fractional_parts = {}

# Calculate initial allocation based on proportion (floor)
for label, files in class_files.items():
    num_files_in_class = len(files)
    ideal_samples = class_proportions[label] * TARGET_SAMPLE_SIZE
    initial_samples = min(math.floor(ideal_samples), num_files_in_class) # Cannot take more than available
    samples_per_class[label] = initial_samples
    fractional_parts[label] = ideal_samples - initial_samples # Store remainder for later distribution
    initial_total += initial_samples
    # Ensure at least 1 sample if possible and target is >= num_classes
    if TARGET_SAMPLE_SIZE >= len(class_files) and initial_samples == 0 and num_files_in_class > 0:
         samples_per_class[label] = 1
         initial_total += 1
         fractional_parts[label] = 0 # Remove from remainder distribution if forced to 1

logging.info(f"Initial allocation: {initial_total} samples.")

# Distribute remaining samples based on fractional parts to reach target size
shortfall = TARGET_SAMPLE_SIZE - initial_total
logging.info(f"Distributing shortfall of {shortfall} samples...")

# Sort classes by descending fractional part to prioritize
sorted_classes_by_fractional = sorted(fractional_parts.items(), key=lambda item: item[1], reverse=True)

for i in range(shortfall):
    # Cycle through classes sorted by fractional part
    class_to_increment = sorted_classes_by_fractional[i % len(sorted_classes_by_fractional)][0]
    num_files_in_class = len(class_files[class_to_increment])
    # Only increment if we haven't already hit the max for that class
    if samples_per_class[class_to_increment] < num_files_in_class:
        samples_per_class[class_to_increment] += 1
    else:
        # If class is full, try the next one in the fractional sort order
        # (This simple loop might slightly under-fill if many classes hit max early,
        # but should be very close for large datasets)
         logging.warning(f"Class '{class_to_increment}' is full ({num_files_in_class}), skipping increment for shortfall item {i+1}. Consider alternative distribution if many classes are full.")
         # A more complex approach would re-distribute these skipped increments

final_allocated_total = sum(samples_per_class.values())
logging.info(f"Final allocation: {final_allocated_total} samples across {len(samples_per_class)} classes.")
if final_allocated_total != TARGET_SAMPLE_SIZE:
     logging.warning(f"Final allocated total ({final_allocated_total}) does not exactly match target ({TARGET_SAMPLE_SIZE}). This might happen if many small classes reached their maximum size.")

# 3. Perform random sampling for each class
logging.info("Performing random sampling within each class...")
sampled_file_paths = []
for label, count in samples_per_class.items():
    if count > 0:
        if count > len(class_files[label]):
             logging.warning(f"Attempting to sample {count} from class {label} which only has {len(class_files[label])} images. Sampling all available.")
             count = len(class_files[label]) # Ensure we don't sample more than available
        
        sampled_for_class = random.sample(class_files[label], count)
        sampled_file_paths.extend(sampled_for_class)
        logging.info(f"  Sampled {len(sampled_for_class)} images for class '{label}'")

# Shuffle the final list (optional, but good practice)
random.shuffle(sampled_file_paths)
logging.info(f"Total files selected: {len(sampled_file_paths)}")

# 4. Create the input JSONL file for the generation script
logging.info(f"Creating input JSONL file at: {OUTPUT_JSONL_PATH}")
lines_written = 0
try:
    with open(OUTPUT_JSONL_PATH, 'w') as outfile:
        for local_path in sampled_file_paths:
            try:
                # Extract class label and filename from path
                parts = local_path.replace(ORIGINAL_VALID_DIR, '').strip(os.sep).split(os.sep)
                if len(parts) != 2:
                     logging.warning(f"Skipping file due to unexpected path structure: {local_path}")
                     continue
                class_label, filename = parts[0], parts[1]

                # Construct GCS URI
                gcs_path = f"{GCS_VALID_PREFIX}/{class_label}/{filename}".replace("\\", "/") # Ensure forward slashes
                gcs_uri = f"gs://{GCS_BUCKET}/{gcs_path}"

                mime_type = get_mime_type(filename)

                # Create the JSON structure
                json_data = {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {"text": USER_PROMPT_TEXT},
                                {"fileData": {"mimeType": mime_type, "fileUri": gcs_uri}}
                            ]
                        },
                        {
                            "role": "model", # This part holds the 'simple' label for the generation script
                            "parts": [{"text": class_label}] # Use the folder name as the simple label
                        }
                    ]
                }
                outfile.write(json.dumps(json_data) + '\n')
                lines_written += 1
            except Exception as e:
                 logging.error(f"Error processing file {local_path}: {e}", exc_info=True)

    logging.info(f"Successfully wrote {lines_written} lines to {OUTPUT_JSONL_PATH}")

except Exception as e:
    logging.error(f"Failed to write output JSONL file: {e}", exc_info=True)


# 5. Optional: Save the list of sampled file paths
if SAMPLED_FILES_LIST_PATH:
     logging.info(f"Saving list of sampled file paths to: {SAMPLED_FILES_LIST_PATH}")
     try:
         with open(SAMPLED_FILES_LIST_PATH, 'w') as f:
             for path in sampled_file_paths:
                 f.write(path + '\n')
     except Exception as e:
         logging.error(f"Failed to write sampled file list: {e}")


logging.info("--- Script Finished ---")
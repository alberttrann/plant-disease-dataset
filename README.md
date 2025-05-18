The original dataset is at https://huggingface.co/datasets/Hemg/new-plant-diseases-dataset
The new dataset is here: https://huggingface.co/datasets/minhhungg/plant-disease-dataset

Since it only consists of two columns image and class, i decided to spice it up with more detailed, dynamic content, so the tuned model can have more "depth" in its answer rather than just providing simple lablels 

First, upload all of the train and valid folders to a bucket on Google Cloud Storage, and convert them all to jsonl format for Google Cloud finetuning
```
#Upload dataset and convert to google's jsonl format template
import os
import json
import logging
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor, as_completed
import mimetypes # Added for better MIME type detection

# --- Configuration --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

# REQUIRED: Set these variables
LOCAL_DATASET_BASE_DIR = "C:\\Users\\alberttran\\Downloads\\New Plant Diseases Dataset(Augmented)\\New Plant Diseases Dataset(Augmented)"  # Path to the folder containing 'train' and 'valid' subfolders
BUCKET_NAME = "plantdiseasee" # <-- Replace with YOUR GCS bucket name
GCS_DESTINATION_PREFIX = "plant_disease_dataset_gemini" # Prefix for the GCS path (consider changing if needed)

# Optional: Customize these
OUTPUT_DIR = "C:\\Users\\alberttran\\Downloads\\jsonl_gemini"      # Local directory to save the generated JSONL files
# --- NOTE: The 'prompt' is now part of the structured 'contents' ---
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif', '.tiff'} # Allowed image file extensions
MAX_WORKERS_UPLOAD = 16             # Number of parallel uploads (adjust based on your machine/network)
LOG_LEVEL = logging.INFO           # Change to logging.DEBUG for more verbose output
DEFAULT_PROMPT = "Identify the plant disease shown in the image." # Default text prompt part

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

# --- Setup Logging ---
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Ensure output directory exists ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Initialize GCS Client ---
try:
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    logging.info(f"Successfully connected to GCS bucket: {BUCKET_NAME}")
except Exception as e:
    logging.error(f"Failed to initialize GCS client or connect to bucket '{BUCKET_NAME}': {e}")
    logging.error("Please ensure the bucket exists and you have authenticated correctly (`gcloud auth application-default login`) with permissions.")
    exit(1)

# --- Function to get MIME type ---
def get_mime_type(file_path):
    """Guesses the MIME type of a file based on its extension."""
    mime_type, _ = mimetypes.guess_type(file_path)
    # Provide common defaults if guess fails, as Vertex AI often needs one
    if mime_type is None:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            return 'image/jpeg'
        elif ext == '.png':
            return 'image/png'
        # Add more specific types if needed
        else:
            logging.warning(f"Could not determine MIME type for {file_path}, defaulting to application/octet-stream.")
            return 'application/octet-stream' # Generic fallback
    return mime_type

# --- Function to upload a single file ---
def upload_blob(local_file_path, gcs_blob_name):
    """Uploads a file to the bucket."""
    try:
        blob = bucket.blob(gcs_blob_name)
        blob.upload_from_filename(local_file_path)
        # logging.debug(f"Uploaded {local_file_path} to gs://{BUCKET_NAME}/{gcs_blob_name}")
        return gcs_blob_name # Return blob name on success
    except Exception as e:
        logging.error(f"Failed to upload {local_file_path} to {gcs_blob_name}: {e}")
        return None # Return None on failure

# --- Main Processing Function ---
def process_split(split_name):
    """
    Processes a data split ('train' or 'valid'):
    1. Finds all image files locally.
    2. Uploads them to GCS in parallel.
    3. Generates the JSONL data using the GCS URIs and the specified Gemini format.
    4. Writes the JSONL file.
    """
    logging.info(f"--- Processing split: {split_name} ---")
    local_split_dir = os.path.join(LOCAL_DATASET_BASE_DIR, split_name)
    if not os.path.isdir(local_split_dir):
        logging.warning(f"Local directory not found: {local_split_dir}. Skipping split '{split_name}'.")
        return

    output_jsonl_path = os.path.join(OUTPUT_DIR, f"{split_name}_gemini_format.jsonl") # Changed output filename
    jsonl_data = []
    upload_tasks = []
    file_count = 0
    image_file_count = 0

    logging.info(f"Scanning local directory: {local_split_dir}...")
    # Walk through the local directory structure
    for root, _, files in os.walk(local_split_dir):
        for filename in files:
            file_count += 1
            file_ext = os.path.splitext(filename)[1].lower()
            # Check if it's an allowed image file
            if file_ext in IMAGE_EXTENSIONS:
                image_file_count += 1
                local_file_path = os.path.join(root, filename)

                # --- Determine Label ---
                label = os.path.basename(root)

                # --- Construct GCS Blob Name ---
                relative_path = os.path.relpath(local_file_path, LOCAL_DATASET_BASE_DIR)
                gcs_blob_name = os.path.join(GCS_DESTINATION_PREFIX, relative_path).replace("\\", "/")

                # Add upload task: (local_path, gcs_blob_name, label)
                upload_tasks.append((local_file_path, gcs_blob_name, label))

    logging.info(f"Found {image_file_count} image files (out of {file_count} total files) for split '{split_name}'.")
    if not upload_tasks:
        logging.warning(f"No image files found to process for split '{split_name}'.")
        return

    # --- Perform Uploads in Parallel ---
    logging.info(f"Starting parallel upload of {len(upload_tasks)} images to gs://{BUCKET_NAME}/{GCS_DESTINATION_PREFIX}/{split_name}/ using up to {MAX_WORKERS_UPLOAD} workers...")
    successful_uploads = {} # Store gcs_blob_name -> label for successful uploads
    upload_counter = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_UPLOAD) as executor:
        future_to_task = {executor.submit(upload_blob, task[0], task[1]): task for task in upload_tasks}

        for future in as_completed(future_to_task):
            local_path, gcs_blob_name, label = future_to_task[future]
            try:
                result = future.result()
                if result:
                    successful_uploads[result] = label
                    upload_counter += 1
                    if upload_counter % 100 == 0 or upload_counter == len(upload_tasks):
                         logging.info(f"Uploaded {upload_counter}/{len(upload_tasks)} images for {split_name}...")
                else:
                    logging.warning(f"Upload failed for: {local_path}")
            except Exception as exc:
                logging.error(f"Upload task for {local_path} generated an exception: {exc}")

    logging.info(f"Finished uploading for split '{split_name}'. Successfully uploaded {len(successful_uploads)} images.")

    if not successful_uploads:
        logging.error(f"No images were successfully uploaded for split '{split_name}'. Cannot generate JSONL file.")
        return

    # --- Generate JSONL Data from Successful Uploads ---
    logging.info(f"Generating JSONL data for {len(successful_uploads)} successfully uploaded images using Gemini format...")
    for gcs_blob_name, label in successful_uploads.items():
         # Construct GCS URI
        gcs_uri = f"gs://{BUCKET_NAME}/{gcs_blob_name}"
        mime_type = get_mime_type(gcs_blob_name) # Use the helper function

        # --- Create JSON object according to the Gemini schema ---
        # NOTE: This assumes a standard input/output structure where the prompt/image
        # are the 'user' turn and the label is the 'model' turn.
        # Verify this against the specific tuning task documentation if issues arise.
        data_item = {
            "contents": [
                # User turn (Input: Prompt + Image)
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": DEFAULT_PROMPT # The text part of the input
                        },
                        {
                            "fileData": {           # The image part of the input
                                "mimeType": mime_type,
                                "fileUri": gcs_uri
                            }
                        }
                    ]
                },
                # Model turn (Output: Label)
                {
                    "role": "model",
                    "parts": [
                        {
                            "text": label # The expected text response (the disease label)
                        }
                    ]
                }
            ]
            # --- Removed 'systemInstruction' as it wasn't in the core example ---
            # If needed, structure it as per the schema and add it here:
            # "systemInstruction": { ... }
        }
        jsonl_data.append(data_item)

    # --- Write JSONL File ---
    logging.info(f"Writing JSONL file: {output_jsonl_path}")
    try:
        with open(output_jsonl_path, 'w') as outfile:
            for item in jsonl_data:
                outfile.write(json.dumps(item) + '\n')
        logging.info(f"Successfully created JSONL file for {split_name} with {len(jsonl_data)} entries.")
    except IOError as e:
        logging.error(f"Failed to write JSONL file {output_jsonl_path}: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    logging.info("Starting dataset upload and JSONL generation process (Gemini Format).")

    # --- Add MIME types if needed (often not necessary for common types) ---
    # mimetypes.add_type('image/heic', '.heic') # Example

    if not os.path.isdir(LOCAL_DATASET_BASE_DIR):
         logging.error(f"Local dataset base directory not found: {LOCAL_DATASET_BASE_DIR}")
         logging.error("Please ensure the `LOCAL_DATASET_BASE_DIR` variable points to the correct path.")
         exit(1)

    process_split("train")
    process_split("valid")
    # Add more splits here if needed (e.g., "test")

    logging.info("--- Process Complete ---")
    logging.info(f"JSONL files (Gemini format) saved in: {OUTPUT_DIR}")
    logging.info(f"Images uploaded to: gs://{BUCKET_NAME}/{GCS_DESTINATION_PREFIX}/")
```
![Screenshot 2025-05-02 152148](https://github.com/user-attachments/assets/3776205d-a767-4132-b6cb-fea08bb00f56)

The original json file looks something like this:
```
{"contents": [{"role": "user", "parts": [{"text": "Identify the plant disease shown in the image."}, {"fileData": {"mimeType": "image/jpeg", "fileUri": "gs://plantdiseasee/plant_disease_dataset/train/Apple___Apple_scab/01a66316-0e98-4d3b-a56f-d78752cd043f___FREC_Scab 3003_new30degFlipLR.JPG"}}]}, {"role": "model", "parts": [{"text": "Apple___Apple_scab"}]}]}
{"contents": [{"role": "user", "parts": [{"text": "Identify the plant disease shown in the image."}, {"fileData": {"mimeType": "image/jpeg", "fileUri": "gs://plantdiseasee/plant_disease_dataset/train/Pepper,_bell___Bacterial_spot/a020fc8f-13d9-47d9-9a06-cf5dcf87c2c3___JR_B.Spot 8896.JPG"}}]}, {"role": "model", "parts": [{"text": "Pepper,_bell___Bacterial_spot"}]}]}
```

Then, I do the batch label-generation tasks with Gemini Flash 2.0 lite, for its generous 4000 RPM:
```
import os
import json
import logging 
import time
# --- CORRECTED IMPORTS ---
from google import genai 
from google.genai import types 
from google.api_core import exceptions as google_api_exceptions 

# --- Configuration ---
PROJECT_ID = "black-acronym-457503-t5"
LOCATION = "us-central1"
GENERATION_MODEL_ID = "gemini-2.0-flash-lite-001" 

# --- Input/Output Files ---
INPUT_JSONL_FILE = r""
OUTPUT_JSONL_FILE = r"" 

# --- API & Generation Settings ---
API_CALL_DELAY = 0.016 
GENERATION_TEMP = 0.6
GENERATION_MAX_TOKENS = 2048
GENERATION_TOP_P = 0.95
# GENERATION_TOP_K = 40

MAX_EXAMPLES_TO_PROCESS = None # 
LOG_LEVEL = logging.DEBUG 

# --- User Prompt Template for Generation ---
GENERATION_USER_PROMPT_TEMPLATE = "Analyze the provided image, which shows symptoms of '{label}', following the detailed 5-section format specified in the system instructions."

# --- System Instruction Template ---
SYSTEM_INSTRUCTION_TEXT = """Disease Identification
• Name the disease(s) or disorder(s), with scientific and common names.
Observable Symptoms
• Describe the visible signs on leaves, stems, roots, flowers, or fruit.
• Note severity (mild, moderate, severe) and distribution (localized, systemic).
Likely Causes & Risk Factors
• Explain probable pathogens (fungi, bacteria, viruses, nematodes) or abiotic stress (nutrient deficiency, drought, chemical damage).
• Mention environmental or cultural conditions that may have contributed.
Management & Treatment Recommendations
• Give both immediate treatments (e.g., fungicides, pruning, biocontrols) and long‐term strategies (crop rotation, resistant varieties).
• Include dosage/rate guidelines, timing, and safety precautions.
Preventive Measures & Monitoring
• Suggest ongoing practices to prevent recurrence (irrigation management, sanitation, soil health).
• Propose a simple scouting schedule and key indicators to watch.
Keep each section concise (2–4 sentences)."""

# --- Safety Settings ---
SAFETY_SETTINGS = [
    types.SafetySetting(
        category="HARM_CATEGORY_HATE_SPEECH",
        threshold=types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold=types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
        threshold=types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_HARASSMENT",
        threshold=types.HarmBlockThreshold.BLOCK_NONE
    )
]

# --- Setup Logging ---
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

# --- Initialize genai Client for Vertex AI ---
try:
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    logging.info(f"genai Client initialized for Vertex AI project '{PROJECT_ID}' in location '{LOCATION}'.")
except Exception as e:
    logging.error(f"Failed to initialize genai Client for Vertex AI: {e}")
    exit()

# --- Prepare **Combined** Generation Configuration ---
try:
    generation_config_combined = types.GenerateContentConfig(
        temperature=GENERATION_TEMP,
        top_p=GENERATION_TOP_P,
        max_output_tokens=GENERATION_MAX_TOKENS,
        safety_settings=SAFETY_SETTINGS,
        system_instruction=[types.Part(text=SYSTEM_INSTRUCTION_TEXT)],
    )
    logging.info("Combined GenerateContentConfig created successfully.")
except Exception as config_err:
    logging.error(f"Failed to create GenerateContentConfig: {config_err}", exc_info=True)
    exit()


# --- Process the JSONL file ---
processed_count = 0
error_count = 0
full_model_name_vertex = f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{GENERATION_MODEL_ID}"
logging.info(f"Starting structured answer generation. Input: '{INPUT_JSONL_FILE}', Output: '{OUTPUT_JSONL_FILE}'")
logging.info(f"Using model: {full_model_name_vertex}")


try:
    with open(INPUT_JSONL_FILE, 'r') as infile, open(OUTPUT_JSONL_FILE, 'w') as outfile:
        for line_num, line in enumerate(infile):
            if MAX_EXAMPLES_TO_PROCESS is not None and processed_count >= MAX_EXAMPLES_TO_PROCESS:
                logging.info(f"Reached MAX_EXAMPLES_TO_PROCESS limit ({MAX_EXAMPLES_TO_PROCESS}). Stopping.")
                break

            if not line.strip(): continue

            try:
                original_data = json.loads(line)
                user_content = original_data['contents'][0]
                model_content = original_data['contents'][1]

                image_part_data = None
                original_user_prompt = ""
                for part in user_content['parts']:
                    if 'fileData' in part: image_part_data = part['fileData']
                    if 'text' in part: original_user_prompt = part['text']

                original_label = model_content['parts'][0]['text']

                if not image_part_data or not original_label or not original_user_prompt:
                    logging.warning(f"Skipping line {line_num + 1}: Missing image, label, or original user prompt.")
                    error_count += 1; continue

                gcs_uri = image_part_data['fileUri']
                mime_type = image_part_data['mimeType']

                # Construct API Input Contents
                generation_user_prompt = GENERATION_USER_PROMPT_TEMPLATE.format(label=original_label)
                user_parts_for_api = [
                    types.Part(text=generation_user_prompt),
                    types.Part(file_data={'mime_type': mime_type, 'file_uri': gcs_uri})
                ]
                contents_for_api = [types.Content(role="user", parts=user_parts_for_api)]

                logging.debug(f"Line {line_num + 1}: Generating structured answer for '{original_label}' ({gcs_uri})")

                # --- API Call using the genai.Client instance ---
                response = None # Initialize response to None
                try:
                    logging.debug(f"Line {line_num + 1}: Calling generate_content...")
                    response = client.models.generate_content(
                        model=full_model_name_vertex,
                        contents=contents_for_api,
                        config=generation_config_combined,
                    )
                    # Log the raw response object for debugging
                    logging.debug(f"Line {line_num + 1}: Received API response object: {response}")


                    generated_structured_answer = ""
                    # Add checks to prevent errors if response structure is unexpected
                    if response and hasattr(response, 'candidates') and response.candidates:
                         candidate = response.candidates[0]
                         if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') and candidate.content.parts:
                             generated_structured_answer = candidate.content.parts[0].text.strip()

                    # Check for blocked prompt more carefully
                    if not generated_structured_answer and response and hasattr(response, 'prompt_feedback') and response.prompt_feedback and hasattr(response.prompt_feedback,'block_reason') and response.prompt_feedback.block_reason:
                         logging.warning(f"Skipping line {line_num + 1}: Response blocked for {gcs_uri}. Reason: {response.prompt_feedback.block_reason}")
                         error_count += 1; continue

                    # Check if still empty after parsing attempts
                    if not generated_structured_answer:
                         logging.warning(f"Skipping line {line_num + 1}: Could not extract answer text for {gcs_uri}. Response object: {response}")
                         error_count += 1; continue

                except google_api_exceptions.NotFound as not_found_err:
                    logging.error(f"Skipping line {line_num + 1}: API call failed for {gcs_uri}. Reason: 404 Not Found.")
                    logging.debug(f"NotFound Error Details: {not_found_err}")
                    error_count += 1; continue
                except google_api_exceptions.ResourceExhausted as quota_err:
                    logging.error(f"Quota Error on line {line_num + 1} for {gcs_uri}. Pausing.")
                    time.sleep(API_CALL_DELAY * 20)
                    error_count += 1; continue
                except Exception as api_err:
                    logging.error(f"Skipping line {line_num + 1}: API call failed for {gcs_uri}: {str(api_err)}")
                    error_count += 1; continue

                # --- Construct the NEW JSONL line FOR FINE-TUNING ---
                new_data_for_finetune = {
                    "contents": [
                        { "role": "user", "parts": [{"text": original_user_prompt}, {"fileData": image_part_data}] },
                        { "role": "model", "parts": [{"text": generated_structured_answer}] }
                    ]
                }

                outfile.write(json.dumps(new_data_for_finetune) + '\n')
                processed_count += 1
                if processed_count % 50 == 0: logging.info(f"Processed {processed_count} examples...")
                time.sleep(API_CALL_DELAY)

            except json.JSONDecodeError as json_err:
                logging.warning(f"Skipping line {line_num + 1}: Invalid JSON: {str(json_err)}")
                error_count += 1
            except KeyError as key_err:
                 logging.warning(f"Skipping line {line_num + 1}: Missing expected key {key_err} in JSON.")
                 error_count += 1
            except Exception as e:
                logging.error(f"Skipping line {line_num + 1}: Unexpected error processing line: {str(e)}")
                error_count += 1

except FileNotFoundError:
    logging.error(f"Error: Input file not found at '{INPUT_JSONL_FILE}'")
    exit()
except Exception as e:
    logging.error(f"An critical error occurred: {e}", exc_info=True)
    exit()

# --- Final Summary Logging ---
logging.info("--- Structured Answer Generation Complete ---")
logging.info(f"Successfully processed and wrote {processed_count} examples.")
logging.info(f"Encountered {error_count} errors/skipped lines.")
logging.info(f"Output file for fine-tuning saved to: '{OUTPUT_JSONL_FILE}'")
logging.warning("<<< IMPORTANT: Review the generated answers for quality BEFORE fine-tuning! >>>")
```

Each line in the new json file now looks something like this:
```
{"contents": [{"role": "user", "parts": [{"text": "Identify the plant disease shown in the image."}, {"fileData": {"mimeType": "image/jpeg", "fileUri": "gs://plantdiseasee/plant_disease_dataset/train/Apple___Apple_scab/03354abb-aa1c-4f9d-a1ef-9f40505cd539___FREC_Scab 3355_90deg.JPG"}}]}, {"role": "model", "parts": [{"text": "Here's an analysis of the image, following your specified format:\n\n**Disease Identification**\n*   **Disease:** Apple scab ( *Venturia inaequalis*)\n*   **Common Name:** Apple scab\n\n**Observable Symptoms**\n*   The image shows olive-green to brown spots with feathery margins on the leaf.\n*   The severity appears to be moderate, with localized spots across the leaf surface.\n\n**Likely Causes & Risk Factors**\n*   The primary cause is the fungus *Venturia inaequalis*.\n*   Risk factors include cool, wet spring weather, which favors fungal spore germination and infection.\n\n**Management & Treatment Recommendations**\n*   Apply protectant fungicides like captan or mancozeb, following label instructions.\n*   Prune infected branches to improve air circulation and remove inoculum.\n\n**Preventive Measures & Monitoring**\n*   Practice good sanitation by removing fallen leaves.\n*   Monitor for initial symptoms in early spring, especially after rain, and scout regularly."}]}]}
{"contents": [{"role": "user", "parts": [{"text": "Identify the plant disease shown in the image."}, {"fileData": {"mimeType": "image/jpeg", "fileUri": "gs://plantdiseasee/plant_disease_dataset/train/Apple___Cedar_apple_rust/85b2d11e-4682-4699-9196-fba13b6232d5___FREC_C.Rust 3718_new30degFlipLR.JPG"}}]}, {"role": "model", "parts": [{"text": "Here's an analysis of the apple leaf, following your requested format:\n\n**Disease Identification**\n*   **Disease:** Cedar Apple Rust ( *Gymnosporangium juniperi-virginianae*). This fungal disease affects apples and other related plants.\n\n**Observable Symptoms**\n*   The image shows circular, orange-red spots (lesions) on the apple leaf, indicating a mild to moderate infection. The spots are distributed across the leaf surface.\n\n**Likely Causes & Risk Factors**\n*   The disease is caused by the fungus *Gymnosporangium juniperi-virginianae*. Infection is favored by wet spring weather. Proximity to Eastern Red Cedar trees increases the risk, as they are alternate hosts.\n\n**Management & Treatment Recommendations**\n*   Remove and destroy infected leaves. Apply a fungicide containing captan, mancozeb, or myclobutanil before bud break and every 7-10 days during the wet spring. Always follow label instructions.\n\n**Preventive Measures & Monitoring**\n*   Plant apple varieties resistant to cedar apple rust. Prune trees to improve air circulation. Regularly scout trees for symptoms, beginning in early spring, and monitor weather conditions."}]}]}
{"contents": [{"role": "user", "parts": [{"text": "Identify the plant disease shown in the image."}, {"fileData": {"mimeType": "image/jpeg", "fileUri": "gs://plantdiseasee/plant_disease_dataset/train/Potato___Late_blight/d5bf2575-1e57-4792-81a1-5770e701be1b___RS_LB 3209_180deg.JPG"}}]}, {"role": "model", "parts": [{"text": "**Disease Identification**\n*   The disease is Potato Late Blight, caused by the oomycete *Phytophthora infestans*. This is a serious disease affecting potato crops worldwide.\n\n**Observable Symptoms**\n*   The image shows a leaf with dark, water-soaked lesions that have progressed to a necrotic, blighted appearance. The affected area is brown and shrunken.\n*   The severity appears moderate, with a localized distribution on the leaf, but the disease can rapidly become systemic if conditions are favorable.\n\n**Likely Causes & Risk Factors**\n*   The primary cause is the oomycete *Phytophthora infestans*, which thrives in cool, humid conditions. Spores are spread by wind and rain.\n*   Risk factors include dense planting, overhead irrigation, and periods of high humidity and moderate temperatures (60-75\u00b0F/15-24\u00b0C).\n\n**Management & Treatment Recommendations**\n*   Immediate treatment involves applying protectant fungicides like chlorothalonil or copper-based products. Follow label instructions for dosage and timing.\n*   Long-term strategies include planting resistant varieties, improving air circulation, and removing and destroying infected plant debris.\n\n**Preventive Measures & Monitoring**\n*   Preventative measures include using certified disease-free seed potatoes and avoiding excessive nitrogen fertilization.\n*   Scout weekly, inspecting leaves for initial symptoms, especially after periods of rain or high humidity. Look for water-soaked spots and the appearance of the white, fuzzy growth on the underside of leaves."}]}]}
```

Then for the validation split, I have to a sampling of 5k examples because that is the limit of Google Cloud:
```
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
```

Then the batch-label generation for validation split still uses same script for the train split but the path for jsonl file needs to be replaced

Then goes the tuning process, with 5 epochs, 0.6 learning rate multiplier and an adapter size of 8. Here is what was going on 
![Screenshot 2025-05-18 185725](https://github.com/user-attachments/assets/b501a692-ccb4-4e17-b65b-9f9d9d33ceae)

I also did some eval, and here was the result:
![Screenshot 2025-05-05 090554](https://github.com/user-attachments/assets/5e0955af-d6e5-409e-8d25-83b5b5a52e2f)

This is eval script:
```
import os
import base64
import logging
import re
import time # <-- Import the time module
import json
import numpy as np
import pandas as pd
# PIL is needed for mime type guessing if not already standard
try:
    from PIL import Image
    import mimetypes
except ImportError:
    logging.warning("Pillow not installed. Mime type guessing might be basic. Install with: pip install Pillow")
    Image = None # Set to None if not available
    mimetypes = None
from io import BytesIO


# --- USE GenAI Client for BOTH ---
from google import genai
from google.genai import types
from google.api_core import exceptions as google_api_exceptions # Keep for error handling


# Scikit-learn & Plotting
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# --- Configuration ---
# Fine-tuned Model Endpoint Configuration
ENDPOINT_PROJECT_ID = "962134623582" # Project hosting the endpoint
ENDPOINT_LOCATION = "us-central1"
ENDPOINT_ID = "1913982562632466432" # Your specific fine-tuned model Endpoint ID
# Construct the full endpoint string needed for genai client model parameter
ENDPOINT_NAME_STR = f"projects/{ENDPOINT_PROJECT_ID}/locations/{ENDPOINT_LOCATION}/endpoints/{ENDPOINT_ID}"


# Judge LLM Configuration
JUDGE_PROJECT_ID = "black-acronym-457503-t5" # Project for calling the judge LLM
JUDGE_LOCATION = "us-central1"
JUDGE_MODEL_ID = "gemini-2.0-flash-lite-001" # Judge LLM
# Use the simple model name string for genai client model parameter
JUDGE_MODEL_NAME_STR = JUDGE_MODEL_ID # Using the short name as per user example


# Test Data Configuration
LOCAL_TEST_DATA_DIR = r"C:\Users\alberttran\Downloads\test\test" # Path containing disease folders


# --- Evaluation Settings ---
# Placeholder if Judge fails to identify label or other errors occur
PREDICTION_LABEL_ERROR_PLACEHOLDER = "<LABEL_ERROR_OR_UNKNOWN>"
JUDGE_EVALUATION_ERROR_PLACEHOLDER = {"error": "Evaluation Failed", "predicted_label": PREDICTION_LABEL_ERROR_PLACEHOLDER}
ENDPOINT_USER_PROMPT = "Identify the plant disease shown in the image and provide details."




# --- API Call Delays (in seconds) ---
ENDPOINT_CALL_DELAY = 10.0  # Delay AFTER calling the fine-tuned endpoint
JUDGE_LLM_CALL_DELAY = 0.016 # Delay AFTER calling the judge LLM




# --- Generation Configs (Based on User Examples) ---


# Config for Fine-tuned Endpoint Call (Matches user example structure)
ENDPOINT_GENERATION_CONFIG = types.GenerateContentConfig(
    temperature = 1.0, # As per user example
    top_p = 0.95,      # As per user example
    max_output_tokens = 8192, # As per user example
    # response_modalities = ["TEXT"], # Usually not needed/valid for genai client call
    # No system_instruction needed for endpoint call usually
    # Safety settings default to BLOCK_SOME, add specific OFF settings if truly needed
    # safety_settings = [...]
)


# --- System Instruction for Judge LLM (V2 - Defined Directly) ---
JUDGE_SYSTEM_INSTRUCTION = """You are an AI assistant acting as an expert botanist and agricultural consultant. Your task is to analyze a generated plant disease description, identify the primary disease or plant status mentioned, and evaluate the description's quality. You will be given the true disease context (from the image source folder) and the full generated description.


**Instructions:**


1.  **Identify Predicted Label:** Carefully read the "Generated Description to Evaluate". Determine the primary common name of the disease or plant status (e.g., "Tomato Septoria leaf spot", "Apple Scab", "Soybean healthy") being described. This is the "Predicted Label".
2.  **Evaluate Quality:** Evaluate the "Generated Description to Evaluate" based on the following criteria using a 1-5 scale (1=Poor, 2=Fair, 3=Good, 4=Very Good, 5=Excellent), considering the "True Disease Context" provided in the user prompt for accuracy checks:
    * **Disease Identification Accuracy Score:** How accurately does YOUR identified "Predicted Label" match the "True Disease Context"? (Score 1-5)
    * **Symptom Description Score:** How accurately/relevantly are symptoms described for the identified "Predicted Label"? (Score 1-5)
    * **Causes/Risk Factors Plausibility Score:** Are mentioned causes/risks plausible for the identified "Predicted Label"? (Score 1-5)
    * **Management/Treatment Actionability Score:** Are recommendations appropriate/actionable for the identified "Predicted Label"? (Score 1-5)
    * **Prevention/Monitoring Usefulness Score:** Is prevention advice useful for the identified "Predicted Label"? (Score 1-5)
    * **Clarity and Fluency Score:** Is the overall description clear and well-written? (Score 1-5)


3.  **Provide Output:** Respond ONLY in the following JSON format:


{
  "predicted_label": "[The common name you identified from the generated text, normalized to lowercase with spaces, e.g., 'tomato septoria leaf spot' or 'soybean healthy']",
  "disease_accuracy_score": [1-5],
  "symptoms_score": [1-5],
  "causes_score": [1-5],
  "management_score": [1-5],
  "prevention_score": [1-5],
  "clarity_fluency_score": [1-5],
  "overall_justification": "[Brief summary of description's strengths/weaknesses.]"
}


**Important:** If you cannot confidently identify a primary disease/status from the generated text, output "unknown" for the "predicted_label". Ensure the predicted label is normalized (lowercase, spaces only, no underscores).
"""


# Config for Judge LLM Call (Matches user example structure, NO system instruction here)
JUDGE_GENERATION_CONFIG = types.GenerateContentConfig(
    temperature = 1.0, # As per user example
    top_p = 0.95,      # As per user example
    max_output_tokens = 2048, # Reduced as JSON output is smaller
    # response_modalities = ["TEXT"], # Usually not needed/valid
    safety_settings = [ # As per user example
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold=types.HarmBlockThreshold.BLOCK_NONE),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold=types.HarmBlockThreshold.BLOCK_NONE),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold=types.HarmBlockThreshold.BLOCK_NONE),
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold=types.HarmBlockThreshold.BLOCK_NONE)
    ],
    # System instruction is passed separately in the API call now
)




# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Helper Functions ---
def get_mime_type(filename):
    """Guesses mime type. Uses mimetypes if available, otherwise basic."""
    if mimetypes:
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type or 'application/octet-stream'
    else:
        ext = filename.lower().split('.')[-1]
        if ext == 'jpg' or ext == 'jpeg': return 'image/jpeg'
        elif ext == 'png': return 'image/png'
        else: return 'application/octet-stream'


def encode_image_bytes(image_path):
    """Reads an image file and returns raw bytes."""
    try:
        with open(image_path, "rb") as image_file:
            return image_file.read()
    except FileNotFoundError: logging.error(f"Image file not found: {image_path}"); return None
    except Exception as e: logging.error(f"Error reading image bytes {image_path}: {e}"); return None


def normalize_label(label):
    """Converts label to a standard format: lowercase, spaces, no underscores."""
    # Keep this to normalize the TRUE labels from folders
    if not label: return ""
    label = label.lower(); label = re.sub(r'_+', ' ', label); label = re.sub(r'\s+', ' ', label); label = label.strip()
    return label


# --- Main Evaluation Logic ---
def evaluate_model_with_judge():
    logging.info("--- Starting Simplified Evaluation (Judge Determines Label) ---")


    # --- Initialize Separate GenAI Clients ---
    try:
        endpoint_client = genai.Client(vertexai=True, project=ENDPOINT_PROJECT_ID, location=ENDPOINT_LOCATION)
        logging.info(f"Initialized genai Client for Endpoint project {ENDPOINT_PROJECT_ID}.")
    except Exception as e: logging.error(f"Failed to initialize genai Client for Endpoint: {e}", exc_info=True); return


    try:
        judge_client = genai.Client(vertexai=True, project=JUDGE_PROJECT_ID, location=JUDGE_LOCATION)
        logging.info(f"Initialized genai Client for Judge LLM project {JUDGE_PROJECT_ID}.")
    except Exception as e: logging.error(f"Failed to initialize genai Client for Judge LLM: {e}", exc_info=True); return


    # --- Data Storage Initialization ---
    classification_true_labels = []; classification_predicted_labels = [] # Predicted labels now come from Judge
    judge_evaluations = []; processed_files_info = []
    prediction_errors = 0; judge_errors = 0; file_process_count = 0


    # --- Finding Test Files ---
    test_files = []
    if not os.path.isdir(LOCAL_TEST_DATA_DIR): logging.error(f"Local test data directory not found: {LOCAL_TEST_DATA_DIR}"); return
    start_find_time = time.time()
    for label_folder in os.listdir(LOCAL_TEST_DATA_DIR):
        label_folder_path = os.path.join(LOCAL_TEST_DATA_DIR, label_folder)
        if os.path.isdir(label_folder_path):
            true_label_raw = label_folder
            # Normalize the TRUE label here
            normalized_true_label = normalize_label(true_label_raw)
            for filename in os.listdir(label_folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(label_folder_path, filename)
                    # Store the NORMALIZED true label
                    test_files.append({"path": image_path, "true_label_normalized": normalized_true_label})
    end_find_time = time.time()
    logging.info(f"Found {len(test_files)} test images in {end_find_time - start_find_time:.2f} seconds.")
    if not test_files: logging.error("No test image files found."); return


    # --- Evaluation Loop ---
    logging.info("--- Starting Evaluation Loop ---")
    start_eval_time = time.time()
    for file_info in test_files:
        image_path = file_info["path"]
        # Retrieve the already normalized true label
        true_label_normalized = file_info["true_label_normalized"]
        file_process_count += 1
        logging.info(f"Processing ({file_process_count}/{len(test_files)}): {os.path.basename(image_path)} (True Label: {true_label_normalized})")


        # --- Part A: Call Fine-tuned Endpoint using genai.Client.generate_content_stream ---
        generated_text = None
        endpoint_failed = False
        try:
            image_bytes = encode_image_bytes(image_path) # Get raw bytes
            if not image_bytes: raise ValueError("Image encoding failed")


            mime_type = get_mime_type(image_path)
            image_part = types.Part(inline_data=types.Blob(mime_type=mime_type, data=image_bytes))
            text_part = types.Part(text=ENDPOINT_USER_PROMPT)
            # Construct contents as per genai library structure
            endpoint_contents = [types.Content(role="user", parts=[text_part, image_part])] # Text first, then image


            # === API CALL 1: Fine-tuned Endpoint (Using STREAMING as per user example) ===
            logging.debug(f"  Calling endpoint stream: {ENDPOINT_NAME_STR}")
            endpoint_response_stream = endpoint_client.models.generate_content_stream(
                model=ENDPOINT_NAME_STR,      # Use the full endpoint path string
                contents=endpoint_contents,
                config=ENDPOINT_GENERATION_CONFIG # Pass the config object to 'config'
            )
            # === END API CALL 1 ===


            # Process the stream to get the full text
            generated_text_parts = []
            for chunk in endpoint_response_stream:
                # Check if chunk has text part and append if so
                try:
                     if chunk.text:
                          generated_text_parts.append(chunk.text)
                except ValueError:
                     # Handle potential empty chunks or other issues gracefully
                     logging.debug("  Received stream chunk without text.")
                     pass # Continue processing stream


            if generated_text_parts:
                generated_text = "".join(generated_text_parts).strip()
                logging.debug(f"  Full endpoint response text length: {len(generated_text)}")
            else:
                logging.warning(f"Endpoint stream yielded no text content for {os.path.basename(image_path)}.")


        except Exception as e:
            # Log the full traceback for endpoint errors to help debug
            logging.error(f"Fine-tuned endpoint prediction failed for {os.path.basename(image_path)}: {e}", exc_info=True)
            prediction_errors += 1
            endpoint_failed = True
        finally:
             # === DELAY 1 ===
             logging.debug(f"Applying {ENDPOINT_CALL_DELAY}s delay after endpoint call attempt.")
             time.sleep(ENDPOINT_CALL_DELAY)
             # === END DELAY 1 ===


        # --- Part B: Store True Label (Predicted label determined later) ---
        classification_true_labels.append(true_label_normalized)
        # Placeholder for predicted label, will be filled after judge call
        judge_determined_predicted_label = PREDICTION_LABEL_ERROR_PLACEHOLDER
        # Store basic info for joining results later
        processed_files_info.append({"filename": os.path.basename(image_path), "true_label": true_label_normalized})




        # --- Part C: Call Judge LLM (Includes Label Identification) ---
        judge_evaluation_result = JUDGE_EVALUATION_ERROR_PLACEHOLDER # Default full result
        judge_call_failed = False
        if generated_text: # Only call judge if we got text
            try:
                # Pass the NORMALIZED true label for context comparison by the judge
                judge_user_prompt = f"""Please evaluate the following plant disease description.
True Disease Context (from folder): {true_label_normalized}


Generated Description to Evaluate:
---
{generated_text}
---
Provide your evaluation in JSON format based on the system instructions."""
                judge_contents = [types.Content(role="user", parts=[types.Part.from_text(judge_user_prompt)])]
                # Define the system instruction Part object here, before the call
                judge_system_instruction_part = types.Part.from_text(JUDGE_SYSTEM_INSTRUCTION) # Use V2 instruction text


                # === API CALL 2: Judge LLM (Using STREAMING as per user example) ===
                logging.debug(f"  Calling judge stream: {JUDGE_MODEL_NAME_STR}")
                judge_response_stream = judge_client.models.generate_content_stream(
                    model=JUDGE_MODEL_NAME_STR,      # Use the judge model name string
                    contents=judge_contents,
                    config=JUDGE_GENERATION_CONFIG,  # Pass judge's config
                    system_instruction=[judge_system_instruction_part] # Pass system instruction separately
                )
                # === END API CALL 2 ===


                # Process the stream to get the full judge response text
                judge_text_parts = []
                for chunk in judge_response_stream:
                     try:
                          if chunk.text: judge_text_parts.append(chunk.text)
                     except ValueError: logging.debug("  Received judge stream chunk without text."); pass


                if judge_text_parts:
                    judge_text_raw = "".join(judge_text_parts).strip()
                    logging.debug(f"  Judge Raw Response Snippet: {judge_text_raw[:150]}...")
                    # Parse judge JSON response
                    try:
                        # Try to find JSON block more reliably
                        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', judge_text_raw, re.DOTALL)
                        if not json_match: json_match = re.search(r'(\{.*?\})', judge_text_raw, re.DOTALL)


                        if json_match:
                            json_str = json_match.group(1) # Extract content within {} or ```json {}```
                            judge_evaluation_result = json.loads(json_str) # Store the full dict


                            # --- Extract the predicted label identified by the judge ---
                            extracted_label = judge_evaluation_result.get("predicted_label") # Use .get for safety
                            if extracted_label and isinstance(extracted_label, str) and extracted_label.lower() not in ["unknown", "unclear", ""]:
                                 # Use the label extracted by the judge
                                 # Assume judge followed normalization instructions (lowercase, spaces)
                                 judge_determined_predicted_label = extracted_label
                                 logging.info(f"  Judge identified label: '{judge_determined_predicted_label}'")
                            else:
                                 logging.warning(f"  Judge did not identify a valid label (Output: {extracted_label}). Using placeholder.")
                                 # Keep placeholder PREDICTION_LABEL_ERROR_PLACEHOLDER
                        else:
                             logging.warning(f"  Judge response for {os.path.basename(image_path)} did not contain valid JSON structure.")
                             judge_errors += 1; judge_call_failed = True
                             # Keep placeholder PREDICTION_LABEL_ERROR_PLACEHOLDER
                    except json.JSONDecodeError as json_e:
                         logging.warning(f"  Failed to parse Judge LLM JSON response for {os.path.basename(image_path)}: {json_e}")
                         judge_errors += 1; judge_call_failed = True
                         # Keep placeholder PREDICTION_LABEL_ERROR_PLACEHOLDER
                else:
                     logging.warning(f"  Judge LLM stream yielded no text content for {os.path.basename(image_path)}.")
                     judge_errors += 1; judge_call_failed = True
                     # Keep placeholder PREDICTION_LABEL_ERROR_PLACEHOLDER


            except Exception as e:
                logging.error(f"  Judge LLM call failed for {os.path.basename(image_path)}: {e}", exc_info=False)
                judge_errors += 1
                judge_call_failed = True
                # Keep placeholder PREDICTION_LABEL_ERROR_PLACEHOLDER
            finally:
                # === DELAY 2 ===
                logging.debug(f"Applying {JUDGE_LLM_CALL_DELAY}s delay after judge call attempt.")
                time.sleep(JUDGE_LLM_CALL_DELAY)
                # === END DELAY 2 ===
        else:
            logging.warning(f"  Skipping Judge LLM call for {os.path.basename(image_path)} due to missing generated text.")
            judge_evaluation_result = {"error": "Skipped - No Input Text", "predicted_label": PREDICTION_LABEL_ERROR_PLACEHOLDER}
            # Ensure predicted label list gets a placeholder if judge skipped
            judge_determined_predicted_label = PREDICTION_LABEL_ERROR_PLACEHOLDER


        # Append the label determined (or placeholder) by the judge for classification report
        classification_predicted_labels.append(judge_determined_predicted_label)
        # Store the full judge evaluation result (dict or error placeholder)
        judge_evaluations.append(judge_evaluation_result)
        logging.debug(f"  Stored Judge Evaluation: {judge_evaluation_result}")
        # Update processed_files_info with the predicted label from judge and full eval
        processed_files_info[-1]['predicted_label_class'] = judge_determined_predicted_label
        processed_files_info[-1]['judge_evaluation'] = json.dumps(judge_evaluation_result) if isinstance(judge_evaluation_result, dict) else str(judge_evaluation_result)




    end_eval_time = time.time()
    logging.info(f"--- Evaluation Loop Finished ({end_eval_time - start_eval_time:.2f} seconds) ---")
    logging.info(f"Total images processed: {file_process_count}")
    logging.info(f"Endpoint prediction errors/no text: {prediction_errors}")
    logging.info(f"Judge LLM errors (call/parsing/label extraction): {judge_errors}")




    # --- Generate Reports ---
    logging.info("\n--- Generating Classification Report & Confusion Matrix (Labels from Judge LLM) ---")
    if not classification_true_labels or not classification_predicted_labels:
        logging.error("No classification labels collected. Cannot generate report.")
    else:
        if len(classification_true_labels) != len(classification_predicted_labels):
             logging.error(f"Length mismatch between true ({len(classification_true_labels)}) and predicted ({len(classification_predicted_labels)}) labels. Cannot generate classification report.")
        else:
            # Add error/unknown placeholders to ensure they appear in the report if present
            class_labels_set = set(classification_true_labels + classification_predicted_labels)
            class_labels = sorted(list(class_labels_set))


            print("\n--- Classification Report (Based on Judge LLM Identified Label) ---")
            try:
                # Use labels parameter to include all potential classes in the report
                report = classification_report(classification_true_labels, classification_predicted_labels, labels=class_labels, zero_division=0)
                print(report)
            except Exception as e: logging.error(f"Could not generate classification report: {e}")


            print("\n--- Generating Confusion Matrix ---")
            try:
                cm = confusion_matrix(classification_true_labels, classification_predicted_labels, labels=class_labels)
                cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
                plt.figure(figsize=(max(10, len(class_labels)*0.6), max(8, len(class_labels)*0.5))) # Dynamic sizing
                sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix (Judge Predicted Label vs True Label)')
                plt.ylabel('Actual Label'); plt.xlabel('Judge Predicted Label')
                plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout()
                plt.savefig("confusion_matrix_judge_label.png") # Updated filename
                logging.info("Confusion matrix saved to confusion_matrix_judge_label.png")
                # plt.show() # Uncomment to display plot
            except Exception as e: logging.error(f"Could not generate confusion matrix plot: {e}")




    # --- Judge LLM Quality Score Summary Report ---
    logging.info("\n--- Generating Judge LLM Quality Score Summary ---")
    if not judge_evaluations: logging.warning("No Judge LLM evaluations collected.")
    else:
        # Filter out error placeholders before processing scores
        valid_evals = [eval_res for eval_res in judge_evaluations if isinstance(eval_res, dict) and 'error' not in eval_res]
        if not valid_evals: logging.warning("No valid Judge LLM evaluations parsed for scoring.")
        else:
             try:
                eval_df = pd.DataFrame(valid_evals)
                avg_scores = {}
                print("\n--- Judge LLM Average Quality Scores (1-5 Scale) ---")
                score_columns = [col for col in eval_df.columns if col.endswith('_score')]
                for col in score_columns:
                     if col in eval_df.columns:
                         numeric_col = pd.to_numeric(eval_df[col], errors='coerce')
                         mean_val = numeric_col.dropna().mean()
                         avg_scores[col] = mean_val
                         print(f"- Average {col}: {mean_val:.2f}" if not np.isnan(mean_val) else f"- Average {col}: N/A (Invalid data)")
                     else: print(f"- Score column '{col}' not found in all results.")


                print("\n--- Sample Judge Justifications ---")
                if 'overall_justification' in eval_df.columns:
                     valid_justifications = eval_df['overall_justification'].dropna()
                     sample_size = min(5, len(valid_justifications))
                     if sample_size > 0:
                         sample_justifications = valid_justifications.sample(sample_size).tolist()
                         for i, just in enumerate(sample_justifications): print(f"[{i+1}] {just}")
                     else: print("No valid justifications found.")
                else: print("Overall justification column not found in judge results.")


                # Save full results to CSV
                try:
                     full_results_df = pd.DataFrame(processed_files_info) # Use the updated info
                     full_results_df.to_csv("full_evaluation_results_judge_label.csv", index=False) # Updated filename
                     logging.info("Full evaluation results saved to full_evaluation_results_judge_label.csv")
                except Exception as save_e: logging.error(f"Failed to save full results to CSV: {save_e}")
             except Exception as report_e: logging.error(f"Failed to process and report Judge LLM evaluations: {report_e}", exc_info=True)


    logging.info("--- Evaluation Script Finished ---")




# --- Run the evaluation ---
if __name__ == "__main__":
    evaluate_model_with_judge()
```

This was much better than the previous attempt of tuning(on the original dataset), with 1 epoch, adapter size 8, LR multiplier 0.5.
![Screenshot 2025-05-04 195019](https://github.com/user-attachments/assets/d1f81fea-6871-42a9-8489-65c13f5d6d62)
![Screenshot 2025-05-18 190055](https://github.com/user-attachments/assets/ba91052c-57a5-41f9-9cf5-39ae8449957e)

The answers from the 2nd tuned model looks something like this:
![dynamic](https://github.com/user-attachments/assets/7657d70c-d6f0-492d-a551-78b842f24a9f)
![dynamic2](https://github.com/user-attachments/assets/322b0240-7575-4960-91f9-fe554a43528b)

Then, I wrote a script to convert the newly created jsonl file to the huggingface-compatible form:
```
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
```

And upload it to huggingface:
```
import os
import logging
from datasets import load_from_disk
from huggingface_hub import login, HfApi # Import login for programmatic option

# --- Configuration ---

# 1. Path to the LOCAL directory where your dataset was saved by save_to_disk()
LOCAL_SAVED_DATASET_DIR = r"C:\Users\alberttran\Downloads\hf_plant_disease_dataset" # ADJUST THIS PATH

# 2. Your desired Hugging Face repository ID
# Format: "YourUsername/YourDatasetName" or "YourOrganizationName/YourDatasetName"
# Example: "AlbertTran/PlantDiseaseStructured"
# The repository will be created if it doesn't exist.
HF_REPO_ID = "minhhungg/plant-disease-dataset" # <<< CHANGE THIS

# 3. Set to True if you want the dataset repository to be private, False for public
IS_PRIVATE = False

# 4. Optional: Hugging Face Token (alternative to CLI login)
# If you logged in via CLI, you can leave this as None.
# Otherwise, uncomment and paste your WRITE token here (less secure).
# HF_TOKEN = "hf_YOUR_TOKEN_HERE"
HF_TOKEN = None

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main Upload Logic ---

logging.info("--- Starting Dataset Upload to Hugging Face Hub ---")

# --- Authentication (Optional: Programmatic Login) ---
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        logging.info("Programmatically logged into Hugging Face Hub.")
    except Exception as e:
        logging.error(f"Programmatic login failed: {e}")
        logging.warning("Please ensure you are logged in via 'huggingface-cli login' or provide a valid HF_TOKEN.")
        exit()
else:
    logging.info("Assuming already logged in via 'huggingface-cli login'.")
    # You might want to add a check here to see if login info exists
    # from huggingface_hub import HfFolder
    # if not HfFolder.get_token():
    #     logging.error("Not logged in. Please use 'huggingface-cli login' first or provide HF_TOKEN.")
    #     exit()

# --- Load Dataset from Disk ---
if not os.path.isdir(LOCAL_SAVED_DATASET_DIR):
    logging.error(f"Dataset directory not found: {LOCAL_SAVED_DATASET_DIR}")
    exit()

try:
    logging.info(f"Loading dataset from {LOCAL_SAVED_DATASET_DIR}...")
    dataset = load_from_disk(LOCAL_SAVED_DATASET_DIR)
    logging.info(f"Dataset loaded successfully: {dataset}")
except Exception as e:
    logging.error(f"Failed to load dataset from disk: {e}", exc_info=True)
    exit()

# --- Push to Hub ---
try:
    logging.info(f"Uploading dataset to Hugging Face Hub repository: {HF_REPO_ID}")
    # The push_to_hub method handles creating the repo if it doesn't exist
    # and uploading the dataset files (Arrow/Parquet).
    dataset.push_to_hub(
        repo_id=HF_REPO_ID,
        private=IS_PRIVATE
        # token=HF_TOKEN # Not needed if logged in via CLI or login()
        # You can add commit_message="Your message" here too
    )
    logging.info("--- Dataset Upload Successful! ---")
    hub_url = f"https://huggingface.co/datasets/{HF_REPO_ID}"
    logging.info(f"View your dataset at: {hub_url}")

except Exception as e:
    logging.error(f"Failed to upload dataset to the Hub: {e}", exc_info=True)
    logging.error("Please check your authentication, repository ID, and network connection.")
```

And that's it : https://huggingface.co/datasets/minhhungg/plant-disease-dataset

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
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
INPUT_JSONL_FILE = r"C:\Users\alberttran\Downloads\validation_gemini_format_sampled_5k.jsonl"
OUTPUT_JSONL_FILE = r"C:\Users\alberttran\Downloads\validation_gemini_structured_answers_5k.jsonl" 

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
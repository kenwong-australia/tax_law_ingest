import os
import json
import openai
import pinecone
import logging
import datetime
import time
import tiktoken
import re
import sys
from dotenv import load_dotenv
from pathlib import Path
# import requests
import threading

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# -----------------------------
# Configuration
# -----------------------------
pinecone_env = os.getenv("PINECONE_ENV", "us-east-1-aws")
index_name = os.getenv("PINECONE_INDEX", "legislation")
VECTOR_DIMENSION = 3072  # Embedding dimension for "text-embedding-3-large"

# Directory containing JSON files for either legislation or ATO rulings
LOCAL_JSON_DIR = os.getenv("LOCAL_JSON_DIR", "/Users/kenmacpro/pinecone-upsert/testfiles_law/json")

# Create local log directory
LOG_DIR = "/Users/kenmacpro/pinecone-upsert/logs/upsert"
os.makedirs(LOG_DIR, exist_ok=True)

# Set up logging with timestamp
timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE_NAME = os.path.join(LOG_DIR, f"log_pinecone_law_{timestamp_str}.txt")

# Log the current working directory
print(f"Current working directory: {os.getcwd()}")
print(f"Log files will be saved to: {LOG_DIR}")

# Configure logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_NAME),
        logging.StreamHandler(sys.stdout)
    ]
)

# Global debug flag.
DEBUG = True

# Define the number of retries and delay between them
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# -----------------------------
# Pinecone Initialization
# -----------------------------
def init_pinecone(api_key: str, env: str, index_name: str):
    """
    Initialize Pinecone and return the target Index object.
    """
    # New Pinecone v2.x initialization
    from pinecone import Pinecone, ServerlessSpec
    
    pc = Pinecone(api_key=api_key)
    
    # Check if index already exists; create if it doesn't
    existing_indexes = pc.list_indexes().names()
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=VECTOR_DIMENSION,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region=env.split('-')[0]  # Extract region from env string
            )
        )
    
    # Return the index
    return pc.Index(index_name)


# -----------------------------
# Embedding Function
# -----------------------------
def get_embedding(text: str, model="text-embedding-3-large") -> list:
    """
    Obtain an embedding vector for the given text using OpenAI's Embeddings API.
    """
    response = openai.Embedding.create(input=[text], model=model)
    return response['data'][0]['embedding']


# -----------------------------
# Main Upsert Logic
# -----------------------------
def main():
    # Ask user if they want to run in test mode
    test_mode = False
    while True:
        user_input = input("Do you want to run in test mode (only process 10 records)? (yes/no): ").lower().strip()
        if user_input in ['yes', 'y']:
            test_mode = True
            break
        elif user_input in ['no', 'n']:
            test_mode = False
            break
        else:
            print("Please enter 'yes' or 'no'.")

    # Track overall start time
    overall_start_time = time.time()
    
    # Print welcome banner
    print("\n" + "=" * 80)
    print(" " * 30 + "LAW DOCUMENT PINECONE INDEXING")
    if test_mode:
        print(" " * 35 + "TEST MODE (10 RECORDS)")
    print("=" * 80 + "\n")
    
    logging.info(f"Starting the law document Pinecone indexing.{' (TEST MODE - 10 records)' if test_mode else ''}")
    
    # Directory containing the JSON files and checkpoint file
    checkpoint_path = os.path.join(LOG_DIR, "pinecone_law_checkpoint.txt")  # Store checkpoint in log directory
    processed_files = []  # Track successfully processed files
    failed_files = []     # Track failed files
    time_per_file = {}    # Track processing time for each file

    # Initialize Pinecone
    try:
        logging.info(f"Initializing Pinecone connection to index {index_name}")
        index = init_pinecone(pinecone_api_key, pinecone_env, index_name)
    except Exception as e:
        logging.error(f"Error initializing Pinecone: {e}")
        return
    
    # Check if JSON directory exists
    if not os.path.exists(LOCAL_JSON_DIR):
        logging.error(f"JSON directory does not exist: {LOCAL_JSON_DIR}")
        return
    
    # Gather all JSON files from the local directory
    json_files = sorted([f for f in os.listdir(LOCAL_JSON_DIR) if f.lower().endswith('.json')])
    total_files = len(json_files)
    logging.info(f"Found {total_files} JSON file(s) to process.")

    # In test mode, only process up to 10 files
    if test_mode and total_files > 10:
        json_files = json_files[:10]
        total_files = 10
        logging.info(f"Test mode: Limited to processing 10 files.")

    # Process JSON files
    for file_idx, file_name in enumerate(json_files, start=1):
        file_path = os.path.join(LOCAL_JSON_DIR, file_name)
        file_start_time = time.time()
        
        # Progress indicator
        progress_str = f"Processing file {file_idx}/{total_files}: {file_name}"
        print("\n" + "-" * len(progress_str))
        print(progress_str)
        print("-" * len(progress_str))
        
        # Calculate elapsed time since start
        elapsed = time.time() - overall_start_time
        h, m = divmod(elapsed, 3600)
        m, s = divmod(m, 60)
        print(f"Time elapsed: {int(h)}h {int(m)}m {int(s)}s")
        
        try:
            # Use timeout context manager to prevent indefinite hanging
            with timeout(seconds=300):  # 5-minute timeout per file
                # Process the file
                logging.info(f"Processing {file_path}")
                
                # Load JSON file
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError as e:
                        error_msg = f"Error parsing JSON: {e}"
                        logging.error(error_msg)
                        failed_files.append((file_name, error_msg))
                        continue

            # Extract text
            text_data = data.get("text", "No Text Provided")

            # We also have a 'metadata' object that may contain "source_file", "full_reference", "url", etc.
            meta = data.get("metadata", {})

            # Distinguish if this JSON is legislation or ATO ruling
            section = data.get("section")  # Legislation typically has "section"
            title   = data.get("title")    # ATO rulings may have "title"
            # The ATO's JSON might also have a top-level "url" (or in metadata)

            # Decide source
            if section and section.lower() != "no section provided":
                source = "legislation"
            else:
                source = "ato_ruling"

            # If the ATO JSON has a top-level 'url', or meta-level 'url', fetch it
            # We'll store it in metadata only if source == "ato_ruling"
            # Or you can store it for both, but presumably, legislation doesn't have one
            url = data.get("url") or meta.get("url")

            # Combine identification + text for embedding
            embed_parts = []
            if source == "legislation" and section:
                embed_parts.append(section)
            elif source == "ato_ruling" and title:
                embed_parts.append(title)
            # Always append the chunk text
            embed_parts.append(text_data)

            embedding_input = "\n\n".join(embed_parts)

            # Truncate if needed before sending to OpenAI
            try:
                safe_input, token_count = truncate_to_token_limit(embedding_input)
                logging.info(f"Input has {token_count} tokens for {file_name}")
                
                # Track if truncation happened
                original_token_count = len(tiktoken.encoding_for_model("text-embedding-3-large").encode(embedding_input))
                if original_token_count > token_count:
                    logging.warning(f"Truncated {file_name} from {original_token_count} to {token_count} tokens")
                
                # Add retry logic specific to embedding
                for attempt in range(MAX_RETRIES):
                    try:
                        embedding = get_embedding(safe_input)
                        break  # Success, exit retry loop
                    except Exception as e:
                        logging.error(f"Embedding failed on attempt {attempt+1}: {str(e)}")
                        if "RemoteDisconnected" in str(e) or "timeout" in str(e).lower():
                            # If connection dropped, try with half the tokens
                            if token_count > 1000:
                                token_count = token_count // 2
                                safe_input, _ = truncate_to_token_limit(
                                    embedding_input, max_tokens=token_count
                                )
                                logging.warning(f"Retrying with reduced token count: {token_count}")
                        
                        if attempt == MAX_RETRIES - 1:
                            raise  # Last attempt failed, re-raise
                        time.sleep(RETRY_DELAY)  # Wait before retry
            except Exception as e:
                logging.error(f"Failed to get embedding for {file_name}: {str(e)}")
                import traceback
                logging.error(traceback.format_exc())
                failed_files.append((file_name, str(e)))
                continue  # Skip to next file

            # Build metadata
            pinecone_metadata = {
                "source": source,  
                "section": section if source == "legislation" else "",
                "title": title if source == "ato_ruling" else "",
                "chunk_text": text_data,
                "source_file": meta.get("source_file", ""),
                "full_reference": meta.get("full_reference", ""),
                # Look for creation_date first, then fall back to upsert_date for backward compatibility
                "creation_date": meta.get("creation_date", meta.get("upsert_date", "")),
                # Add keywords and categories from metadata
                "keywords": meta.get("keywords", []),
                "categories": meta.get("categories", []),
                # Add a Pinecone-specific upsert timestamp
                "pinecone_upsert_date": datetime.datetime.now().isoformat()
            }

            # Only store a "url" field if it's an ATO doc that actually has a URL
            # (Alternatively, store it for both; just put None for legislation)
            if source == "ato_ruling":
                pinecone_metadata["url"] = url if url else None

            # ID for Pinecone: section if legislation, else title or fallback to file name
            if source == "legislation":
                vector_id = section if section else file_name
            else:
                vector_id = title if title else file_name

            # Sanitize the vector ID
            vector_id = vector_id.replace(" ", "_").encode("ascii", "ignore").decode()

            # Determine namespace
            namespace = determine_namespace(file_name, pinecone_metadata)

            # Upsert with retries
            for attempt in range(MAX_RETRIES):
                try:
                    index.upsert([
                        {
                            "id": vector_id,
                            "values": embedding,
                            "metadata": pinecone_metadata
                        }
                    ], namespace=namespace)
                    break  # Break if successful
                except Exception as e:
                    logging.error(f"Upsert failed for {file_name} on attempt {attempt + 1}: {e}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)  # Wait before retrying
                    else:
                        raise  # Raise the exception if the last attempt fails

            # Track success
            processed_files.append(file_name)
            elapsed_time = time.time() - overall_start_time
            time_per_file[file_name] = round(time.time() - file_start_time, 2)

            # Convert elapsed time to hours, minutes, and seconds
            hours, rem = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(rem, 60)
            logging.info(f"Progress: {file_idx}/{total_files} files upserted. Time elapsed: {int(hours)}h {int(minutes)}m {int(seconds)}s.")

            # Create individual summary log
            dt_string = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            summary_filename = f"{file_name}_pinecone_summary_{dt_string}.txt"
            summary_log_path = os.path.join(LOG_DIR, summary_filename)  # Save in log directory
            write_individual_summary_log(summary_log_path, file_name, os.path.getsize(file_path), elapsed_time)

        except Exception as e:
            error_msg = f"Error processing file {file_name}: {e}"
            logging.error(error_msg)
            failed_files.append((file_name, error_msg))
            
            # Create summary log for failed file
            dt_string = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            summary_filename = f"{file_name}_pinecone_error_summary_{dt_string}.txt"
            summary_log_path = os.path.join(LOG_DIR, summary_filename)  # Save in log directory
            file_duration = time.time() - file_start_time
            write_individual_summary_log(summary_log_path, file_name, os.path.getsize(file_path), file_duration, False, error_msg)
            
        # Update checkpoint file after processing each file (successful or not)
        try:
            with open(checkpoint_path, "w") as cp:
                cp.write(str(file_idx))
        except Exception as e:
            logging.error(f"Error writing to checkpoint file: {e}")

    # Track overall end time
    overall_end_time = time.time()
    total_execution_time = overall_end_time - overall_start_time

    # Convert total execution time to hours, minutes, and seconds
    total_hours, total_rem = divmod(total_execution_time, 3600)
    total_minutes, total_seconds = divmod(total_rem, 60)

    # Generate summary report
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"pinecone_law_report_{timestamp}.txt"
    report_path = os.path.join(LOG_DIR, report_filename)  # Save in log directory
    with open(report_path, "w", encoding="utf-8") as report_file:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_file.write(f"Law Pinecone Indexing Report - Generated on {current_time}\n")
        report_file.write(f"{'TEST MODE (10 records)' if test_mode else 'FULL MODE'}\n")
        report_file.write(f"Total .json files processed: {total_files}\n")
        report_file.write(f"Successfully upserted: {len(processed_files)}\n")
        report_file.write(f"Failed to upsert: {len(failed_files)}\n")
        report_file.write(f"All files upserted? {'YES' if len(processed_files) == total_files else 'NO'}\n")
        report_file.write(f"Total execution time: {int(total_hours)}h {int(total_minutes)}m {int(total_seconds)}s\n\n")

        report_file.write("Time taken per file:\n")
        for fname, duration in time_per_file.items():
            report_file.write(f"  - {fname}: {duration:.2f} sec\n")

        if failed_files:
            report_file.write("\nList of failed files:\n")
            for fname, error in failed_files:
                report_file.write(f"  - {fname}: {error}\n")

    print(f"Pinecone indexing completed. Report generated and saved as '{report_path}'.")


def truncate_to_token_limit(text, max_tokens=8000, model="text-embedding-3-large"):
    """Truncate text to fit within token limit."""
    try:
        enc = tiktoken.encoding_for_model(model)
        tokens = enc.encode(text)
        if len(tokens) <= max_tokens:
            return text, len(tokens)
        
        # Truncate to max_tokens
        truncated_tokens = tokens[:max_tokens]
        truncated_text = enc.decode(truncated_tokens)
        logging.warning(f"Text truncated from {len(tokens)} to {len(truncated_tokens)} tokens")
        return truncated_text, len(truncated_tokens)
    except Exception as e:
        logging.warning(f"Error in token counting: {e}, using character-based truncation")
        # Fallback to character-based approximation (assuming ~4 chars per token)
        char_limit = max_tokens * 4
        if len(text) <= char_limit:
            return text, len(text) // 4
        return text[:char_limit], max_tokens


def determine_namespace(file_name, metadata):
    """Determine the namespace based on file name and metadata."""
    # Check file name for ATO prefix
    if file_name.startswith("ATO_"):
        return "ato"
    
    # Check document_type field in metadata
    if metadata.get("document_type") == "ato_ruling":
        return "ato"
        
    # Default to no namespace (for legislation)
    return None


def write_individual_summary_log(filename, file_name, file_size, duration, success=True, error_msg=None):
    with open(filename, "w", encoding="utf-8") as log_f:
        log_f.write("Summary of Pinecone Upsert\n")
        log_f.write("===========================\n")
        log_f.write(f"Timestamp: {datetime.datetime.now()}\n\n")
        log_f.write(f"File name: {file_name}\n")
        log_f.write(f"File size: {file_size} bytes\n")
        log_f.write(f"Success: {'YES' if success else 'NO'}\n")
        log_f.write(f"Duration: {duration:.2f} seconds\n")
        if not success:
            log_f.write(f"Error: {error_msg}\n")


# Define a timeout context manager to prevent indefinite hangs
class TimeoutError(Exception):
    """Custom timeout error"""
    pass

class timeout:
    def __init__(self, seconds=60):
        self.seconds = seconds
        self.timer = None
        
    def handle_timeout(self):
        logging.error(f"Operation timed out after {self.seconds} seconds")
        raise TimeoutError(f"Operation timed out after {self.seconds} seconds")
        
    def __enter__(self):
        self.timer = threading.Timer(self.seconds, self.handle_timeout)
        self.timer.daemon = True
        self.timer.start()
        
    def __exit__(self, type, value, traceback):
        if self.timer:
            self.timer.cancel()
            self.timer = None


def upsert_to_pinecone(index, vectors, namespace=None):
    """Upsert vectors to Pinecone with timeout handling."""
    max_retries = 5
    backoff_factor = 2
    
    for attempt in range(max_retries):
        try:
            # Add timeout to the upsert operation
            start_time = time.time()
            result = index.upsert(vectors=vectors, namespace=namespace, timeout=120)
            processing_time = time.time() - start_time
            logging.info(f"Pinecone upsert completed in {processing_time:.2f} seconds")
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                sleep_time = backoff_factor ** attempt
                logging.warning(f"Pinecone upsert error: {e}. Retrying in {sleep_time}s (attempt {attempt+1}/{max_retries})")
                time.sleep(sleep_time)
            else:
                logging.error(f"Failed to upsert to Pinecone after {max_retries} attempts: {e}")
                raise


if __name__ == "__main__":
    main()

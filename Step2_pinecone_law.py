import os
import json
import openai
import pinecone
import logging
import datetime
import time
import tiktoken
from dotenv import load_dotenv

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

# Log file with timestamp
timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE_NAME = f"log_pinecone_legislation_{timestamp_str}.txt"

# Set up basic logging to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    # Track overall start time
    overall_start_time = time.time()

    # Initialize Pinecone index
    index = init_pinecone(pinecone_api_key, pinecone_env, index_name)
    logging.info(f"Connected to Pinecone index '{index_name}'.")

    # Gather all JSON files from the local directory
    json_files = [f for f in os.listdir(LOCAL_JSON_DIR) if f.lower().endswith('.json')]
    total_files = len(json_files)
    logging.info(f"Found {total_files} JSON file(s) to process.")

    # Track files
    upserted_files = []
    failed_files = []
    time_per_file = {}
    truncated_files = {}  # Add this to track truncated files

    for i, file_name in enumerate(json_files, start=1):
        file_path = os.path.join(LOCAL_JSON_DIR, file_name)
        start_time = time.time()  # Track start time for this file

        try:
            # Load JSON
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

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
                    truncated_files[file_name] = (original_token_count, token_count)
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
                failed_files.append(file_name)
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
            upserted_files.append(file_name)
            elapsed_time = time.time() - overall_start_time
            time_per_file[file_name] = round(time.time() - start_time, 2)

            # Convert elapsed time to hours, minutes, and seconds
            hours, rem = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(rem, 60)
            logging.info(f"Progress: {i}/{total_files} files upserted. Time elapsed: {int(hours)}h {int(minutes)}m {int(seconds)}s.")

        except Exception as e:
            logging.error(f"Failed processing {file_name}: {e}")
            failed_files.append(file_name)

    # Track overall end time
    overall_end_time = time.time()
    total_execution_time = overall_end_time - overall_start_time

    # Convert total execution time to hours, minutes, and seconds
    total_hours, total_rem = divmod(total_execution_time, 3600)
    total_minutes, total_seconds = divmod(total_rem, 60)

    # -----------------------------
    # Generate Log File
    # -----------------------------
    successful_count = len(upserted_files)
    failed_count = len(failed_files)

    with open(LOG_FILE_NAME, "w", encoding="utf-8") as log_f:
        log_f.write("Summary of Pinecone Upserts\n")
        log_f.write("===========================\n")
        log_f.write(f"Timestamp: {datetime.datetime.now()}\n\n")
        log_f.write(f"Total .json files processed: {total_files}\n")
        log_f.write(f"Successfully upserted: {successful_count}\n")
        log_f.write(f"Failed to upsert: {failed_count}\n")
        log_f.write(f"All files upserted? {'YES' if successful_count == total_files else 'NO'}\n")
        log_f.write(f"Total execution time: {int(total_hours)}h {int(total_minutes)}m {int(total_seconds)}s\n\n")

        log_f.write("Time taken per file:\n")
        for fname, duration in time_per_file.items():
            log_f.write(f"  - {fname}: {duration:.2f} sec\n")

        log_f.write("\nList of upserted files:\n")
        for fname in upserted_files:
            log_f.write(f"  - {fname}\n")

        if failed_count > 0:
            log_f.write("\nList of failed files:\n")
            for fname in failed_files:
                log_f.write(f"  - {fname}\n")

        # Add truncation summary
        if truncated_files:
            log_f.write("\nFiles that required truncation:\n")
            for fname, (orig_tokens, trunc_tokens) in truncated_files.items():
                reduction_pct = ((orig_tokens - trunc_tokens) / orig_tokens) * 100
                log_f.write(f"  - {fname}: {orig_tokens} → {trunc_tokens} tokens ({reduction_pct:.1f}% reduction)\n")

    logging.info(f"\nUpsert process complete.")
    logging.info(f"Total execution time: {int(total_hours)}h {int(total_minutes)}m {int(total_seconds)}s")
    logging.info(f"Log file written to: {LOG_FILE_NAME}")


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


if __name__ == "__main__":
    main()

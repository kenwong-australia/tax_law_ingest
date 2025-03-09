#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ATO Document Processor with LLM-Based Extraction and Chunking

- Loads JSON files containing ATO rulings from a specified directory.
- Uses GPT-4o to extract and structure key information from each document.
- Identifies metadata like Doc ID, Title, URL, and Date.
- Extracts issue statements, decisions, and reasoning from the ruling.
- Outputs structured JSON files with "ato_ruling" document type.
- Saves a summary log file for each batch processed showing:
    * Total files processed
    * Success/failure rate
    * Processing time per file and overall
"""

import os
import re
import json
import openai
import datetime
import logging
import time
import sys
from pathlib import Path
from dotenv import load_dotenv
import concurrent.futures  # Add concurrent.futures for parallel processing

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
# Create local log directory
LOG_DIR = "/Users/kenmacpro/pinecone-upsert/logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Set up logging with timestamp
timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE_NAME = os.path.join(LOG_DIR, f"log_chunking_ato_{timestamp_str}.txt")

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

def debug_print(message):
    if DEBUG:
        print(message)

# API Keys
openai.api_key = os.getenv("OPENAI_API_KEY")

# Directory paths
INPUT_DIR = os.getenv("ATO_INPUT_DIR", "/Users/kenmacpro/pinecone-upsert/testfiles_cgt")
OUTPUT_DIR = os.getenv("JSON_OUTPUT_DIR", "/Users/kenmacpro/pinecone-upsert/testfiles_law/json")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Max files to process in one run (0 = no limit)
MAX_FILES_TO_PROCESS = int(os.getenv("MAX_FILES_TO_PROCESS", "0"))

# Number of parallel workers (adjust based on your API rate limits)
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "5"))

# Updated prompt for a single-chunk process with detailed instructions:
single_chunk_prompt = """Please process the attached document to extract and summarize the following, while retaining the important text and structure. Do not invent new statements unless it is necessary for clarity or to fulfill the required format.

Note: This appears to be a reference/metadata document, not the full ruling text. Please extract whatever information is available.

Header/Metadata & References:
- Doc ID: [If available]
- Title: [Title of the document]
- URL: [Any URL present]
- Date: [Any relevant date]
- Legislative references, if any.

Issue & Decision:
- Identify and summarize the main issue addressed by the document, if indicated.

Facts & Reasoning:
- Extract any available information about facts, reasoning or background.

Present your answer as a single comprehensive chunk with clearly labeled sections.

---
Document content:
"""

# NOTE: For a complete solution, these files would need to be enhanced by:
# 1. Using the URLs in each file to fetch the actual full text of each ruling
# 2. Using web scraping to obtain the complete document content
# 3. Then processing the actual full text rather than just these reference files
#
# The current approach will only process the metadata and small snippets available
# in these reference files.

def remove_markdown(text: str) -> str:
    """
    Remove markdown formatting:
    - Converts markdown links [text](url) to just url.
    - Removes bold, italics, and underscore markers.
    """
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\2', text)
    text = re.sub(r'(\*\*|\*|__|_)', '', text)
    return text

def call_llm(text: str, prompt: str) -> str:
    """
    Calls the OpenAI ChatCompletion API with the given prompt and document text.
    Returns the LLM's response as a single processed chunk.
    """
    # Apply token limit - GPT-4o has ~8K token limit, leave room for prompt and response
    # Using rough estimate of 4 chars per token
    MAX_CONTENT_CHARS = 100000  # ~ tokens, leaving room for prompt and response
    
    # Check if text is too long and truncate if needed
    original_length = len(text)
    if original_length > MAX_CONTENT_CHARS:
        text = text[:MAX_CONTENT_CHARS]
        truncation_message = f"Content truncated from {original_length} to {len(text)} characters due to token limits"
        logging.warning(truncation_message)
        print(f"\n⚠️ {truncation_message}")
        
        # Add a note about truncation to the text
        text += "\n\n[Note: This content was truncated due to length limits. Only the beginning portion is shown.]"
    
    full_prompt = prompt + text
    prompt_tokens = len(full_prompt) / 4  # Rough estimate of tokens
    
    # Log the start of processing with size info
    logging.info(f"Starting LLM processing - approx {prompt_tokens:.0f} tokens")
    print(f"Starting LLM call (est. {prompt_tokens:.0f} tokens)...", end="", flush=True)
    
    messages = [
        {"role": "system", "content": "You are an AI assistant specializing in Australian tax law and ATO rulings. Extract and organize the key information from ATO documents."},
        {"role": "user", "content": full_prompt}
    ]
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            start_time = time.time()  # Start timing
            
            # Update shorter timeout to avoid extremely long waits
            response = openai.ChatCompletion.create(
                model="gpt-4o",  # Use the appropriate model name
                messages=messages,
                temperature=0,
                timeout=90,  # 90 second timeout
                request_timeout=90  # Also set request_timeout
            )
            
            end_time = time.time()  # End timing
            processing_time = end_time - start_time
            
            # More detailed success logging
            completion_tokens = len(response.choices[0].message.content) / 4  # Rough estimate
            logging.info(f"LLM processing completed in {processing_time:.2f} seconds - approx {completion_tokens:.0f} output tokens")
            print(f" done in {processing_time:.2f}s", flush=True)
            
            return response.choices[0].message.content
        except openai.error.OpenAIError as e:
            logging.error(f"Error communicating with OpenAI: {e}")
            print(f" ERROR: {str(e)[:50]}...", flush=True)
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logging.info(f"Retrying in {wait_time} seconds... (Attempt {attempt+1}/{max_retries})")
                print(f"\nRetrying in {wait_time}s (Attempt {attempt+1}/{max_retries})...", end="", flush=True)
                time.sleep(wait_time)
            else:
                print(f"\nFailed after {max_retries} attempts.", flush=True)
                raise
        except Exception as e:
            logging.error(f"Unexpected error during LLM call: {e}")
            print(f" UNEXPECTED ERROR: {str(e)[:50]}...", flush=True)
            raise

def parse_metadata(chunk_text: str):
    """
    Attempt to extract metadata such as doc_id, title, URL, and date from the processed chunk.
    Returns a dictionary with metadata fields.
    """
    metadata = {
        "Doc ID": None,
        "Title": "Untitled Document",
        "URL": "No URL Provided",
        "Date": "No date info provided"
    }
    
    lines = chunk_text.splitlines()
    for line in lines:
        line_stripped = remove_markdown(line.strip())
        line_stripped = re.sub(r'^[\-\*\s]+', '', line_stripped).strip()
        if line_stripped.lower().startswith("doc id:"):
            metadata["Doc ID"] = line_stripped.split(":", 1)[1].strip()
        elif line_stripped.lower().startswith("title:"):
            metadata["Title"] = line_stripped.split(":", 1)[1].strip()
        elif line_stripped.lower().startswith("url:") or line_stripped.lower().startswith("href:"):
            metadata["URL"] = line_stripped.split(":", 1)[1].strip()
            if metadata["URL"] and not metadata["URL"].lower().startswith("http"):
                metadata["URL"] = "https://ato.gov.au/" + metadata["URL"].lstrip("/")
        elif line_stripped.lower().startswith("date:") or line_stripped.lower().startswith("date of decision:"):
            metadata["Date"] = line_stripped.split(":", 1)[1].strip()
            
    return metadata

def write_individual_summary_log(log_file, filename, input_size, processing_time, success=True, error_message=None):
    """Write an individual summary log for an ATO file."""
    # Convert time_taken to hours, minutes, and seconds
    hours, rem = divmod(processing_time, 3600)
    minutes, seconds = divmod(rem, 60)
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Processing Summary for {filename}\n")
        f.write("=" * 40 + "\n")
        f.write(f"Processing Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input File Size: {input_size} bytes\n")
        f.write(f"Processing Status: {'SUCCESS' if success else 'FAILED'}\n")
        if not success and error_message:
            f.write(f"Error: {error_message}\n")
        f.write(f"Time taken: {int(hours)}h {int(minutes)}m {int(seconds)}s\n")

def sanitize_filename(filename):
    """Replace characters that are invalid in filenames with underscores."""
    # Replace slashes, colons, and other problematic characters
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def process_single_file(file_data):
    """Process a single file. This is a helper function for parallel processing."""
    file_name, file_index, total_files, start_index = file_data
    file_counter = start_index + file_index
    file_start_time = time.time()  # Track time for this file
    
    progress_str = f"Processing file {file_index}/{total_files} [{file_counter}/{total_files}]: {file_name}"
    print("\n" + "-" * len(progress_str))
    print(progress_str)
    print("-" * len(progress_str))
    
    local_file_path = os.path.join(INPUT_DIR, file_name)
    logging.info(f"Processing file: {local_file_path}")
    
    try:
        # Read the input JSON file
        with open(local_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Debug: Print the keys in the JSON to help diagnose structure
        logging.info(f"JSON keys in file: {list(data.keys())}")
        
        # Pre-extract document ID from filename or file_info for failsafe usage
        doc_id_from_file = None
        
        # Try to get doc_id from file_name first (most reliable)
        if 'file_name' in data and data['file_name']:
            # Try to extract ATO ID from patterns like "AID|AID201126|00001.json"
            file_parts = data['file_name'].split('|')
            if len(file_parts) >= 2:
                doc_id_from_file = file_parts[1]
                logging.info(f"Extracted doc_id '{doc_id_from_file}' from file_name field")
        
        # If not found in file_name, try file_info -> href
        if not doc_id_from_file and 'file_info' in data and isinstance(data['file_info'], dict):
            if 'a_attr' in data['file_info'] and 'href' in data['file_info']['a_attr']:
                href = data['file_info']['a_attr']['href']
                # Extract from patterns like "/law/view/document?docid=AID/AID201126/00001"
                if 'docid=' in href:
                    docid_part = href.split('docid=')[1]
                    doc_parts = docid_part.split('/')
                    if len(doc_parts) >= 2:
                        doc_id_from_file = doc_parts[1]
                        logging.info(f"Extracted doc_id '{doc_id_from_file}' from URL")
        
        # If still not found, use the file name itself
        if not doc_id_from_file:
            doc_id_from_file = file_name.replace('.json', '')
            logging.info(f"Using filename '{doc_id_from_file}' as doc_id")
        
        # Extract text content - try different possible fields
        content = None
        title_from_file = None
        url_from_file = None
        
        # Get title from file_info if available (for fallback)
        if 'file_info' in data and isinstance(data['file_info'], dict) and 'title' in data['file_info']:
            title_from_file = data['file_info']['title']
        
        # Get URL from file_info if available (for fallback)
        if 'file_info' in data and isinstance(data['file_info'], dict) and 'a_attr' in data['file_info']:
            if 'href' in data['file_info']['a_attr']:
                href = data['file_info']['a_attr']['href']
                url_from_file = "https://ato.gov.au" + href if href.startswith('/') else href
        
        # Option 1: Try 'content' field
        if 'content' in data and data['content']:
            content = data['content']
            logging.info("Found content in 'content' field")
        
        # Option 2: Try 'page_md' field (common in ATO reference files)
        elif 'page_md' in data and data['page_md']:
            content = data['page_md']
            # Also include file_info if available
            if title_from_file:
                content += f"\n\nTitle: {title_from_file}"
            if url_from_file:
                content += f"\nURL: {url_from_file}"
            logging.info("Found content in 'page_md' field")
        
        # Option 3: Try 'text' field
        elif 'text' in data and data['text']:
            content = data['text']
            logging.info("Found content in 'text' field")
            
        # Option 4: Try 'file_info' -> 'content'
        elif 'file_info' in data and isinstance(data['file_info'], dict):
            if 'content' in data['file_info'] and data['file_info']['content']:
                content = data['file_info']['content']
                logging.info("Found content in 'file_info.content' field")
        
        # Option 5: If we have HTML, try to extract text
        elif 'html' in data and data['html']:
            content = data['html']
            content = re.sub(r'<[^>]+>', ' ', content)  # Simple HTML tag removal
            logging.info("Found and cleaned HTML content")
        
        # Handle minimal content case - construct content from available metadata
        if not content or len(content.strip()) < 50:  # Very minimal content
            logging.warning(f"Content is minimal or empty. Constructing from metadata.")
            content = "ATO Document Reference\n\n"
            
            if title_from_file:
                content += f"Title: {title_from_file}\n"
            if doc_id_from_file:
                content += f"Document ID: {doc_id_from_file}\n"
            if url_from_file:
                content += f"URL: {url_from_file}\n"
                
            if 'path_info' in data and isinstance(data['path_info'], list):
                content += "\nCategories:\n"
                for item in data['path_info']:
                    if isinstance(item, dict) and 'title' in item:
                        content += f"- {item['title']}\n"
            
            logging.info(f"Constructed content from metadata: {len(content)} chars")
        
        file_size = len(content)
        logging.info(f"Content length: {file_size} characters")
        
        # Call LLM to extract and structure information
        structured_content = call_llm(content, single_chunk_prompt)
        
        # Extract metadata from the structured content
        extracted_metadata = parse_metadata(structured_content)
        
        # Use pre-extracted doc_id if LLM extraction failed
        extracted_doc_id = extracted_metadata.get("Doc ID")
        if not extracted_doc_id or extracted_doc_id == "Not available":
            extracted_doc_id = doc_id_from_file
            logging.info(f"Using pre-extracted doc_id: {extracted_doc_id}")
        
        # Sanitize filename to replace invalid characters
        output_filename = f"ATO_{sanitize_filename(extracted_doc_id)}.json"
        
        # Create the output JSON structure with metadata and chunk
        output_data = {
            "title": extracted_metadata.get("Title", title_from_file or "Unknown Title"),
            "url": extracted_metadata.get("URL", url_from_file or ""),
            "text": structured_content,
            "metadata": {
                "doc_id": extracted_doc_id,
                "full_reference": f"{extracted_doc_id} {extracted_metadata.get('Title', title_from_file or 'Unknown Title')}",
                "date_info": extracted_metadata.get("Date", ""),
                "document_type": "ato_ruling",
                "source_file": file_name,
                "creation_date": datetime.datetime.now().isoformat()
            }
        }
        
        # Save the processed file
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        # Calculate processing time
        file_duration = time.time() - file_start_time
        
        return {
            "status": "success",
            "file_name": file_name,
            "output_filename": output_filename,
            "file_size": file_size,
            "duration": file_duration
        }
        
    except Exception as e:
        error_msg = f"Error processing file {file_name}: {e}"
        logging.error(error_msg)
        file_duration = time.time() - file_start_time
        
        # Additional debugging for failed files
        try:
            with open(local_file_path, 'r', encoding='utf-8') as f:
                sample_data = json.load(f)
                logging.error(f"File structure: Keys at root level: {list(sample_data.keys())}")
                logging.error(f"File size: {os.path.getsize(local_file_path)} bytes")
                
                # Save a copy of the problematic file for examination
                debug_file = os.path.join(LOG_DIR, f"debug_{file_name}")
                with open(debug_file, 'w', encoding='utf-8') as df:
                    json.dump(sample_data, df, indent=2)
                logging.error(f"Saved debug copy to {debug_file}")
        except Exception as debug_err:
            logging.error(f"Error during debugging: {debug_err}")
        
        return {
            "status": "failed",
            "file_name": file_name,
            "error": str(e),
            "file_size": len(content) if 'content' in locals() else 0,
            "duration": file_duration
        }

def main():
    # Track overall start time
    overall_start_time = time.time()
    
    # Print welcome banner
    print("\n" + "=" * 80)
    print(" " * 30 + "ATO DOCUMENT CHUNKING PIPELINE")
    print("=" * 80 + "\n")
    
    logging.info("Starting the ATO document chunking pipeline.")
    
    # Directory containing the JSON files and checkpoint file
    checkpoint_path = os.path.join(LOG_DIR, "chunking_checkpoint.txt")  # Store checkpoint in log directory
    processed_files = []  # Track successfully processed files
    failed_files = []     # Track failed files
    time_per_file = {}    # Track processing time for each file
    
    # Ask user if they want to reset the checkpoint
    reset_checkpoint = input("Do you want to reset the checkpoint? (yes/no): ").strip().lower()
    if reset_checkpoint == 'yes':
        start_index = 0
        logging.info("Checkpoint reset to 0.")
    else:
        # Read checkpoint if it exists
        start_index = 0
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, "r") as cp:
                    start_index = int(cp.read().strip())
                    logging.info(f"Resuming from checkpoint: file index {start_index}")
            except Exception as e:
                logging.error(f"Error reading checkpoint file: {e}")
    
    file_counter = start_index  # Set file counter to the starting index
    
    try:
        # Sort files alphabetically to ensure a stable processing order
        files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.json')])
        if not files:
            logging.error("No JSON files found in the input directory.")
            return

        total_files = len(files)
        logging.info(f"Found {total_files} JSON files in {INPUT_DIR}")
        
        # Determine how many files to process
        files_to_process = files[start_index:]
        if MAX_FILES_TO_PROCESS > 0:
            files_to_process = files_to_process[:MAX_FILES_TO_PROCESS]
            logging.info(f"Will process up to {len(files_to_process)} files (MAX_FILES_TO_PROCESS={MAX_FILES_TO_PROCESS})")
        else:
            logging.info(f"Will process all {len(files_to_process)} remaining files")

        # Prepare file data for parallel processing
        file_data_list = [(file_name, i, len(files_to_process), start_index) 
                          for i, file_name in enumerate(files_to_process, 1)]
        
        # Process files in parallel
        print(f"\nProcessing with {NUM_WORKERS} parallel workers")
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            results = list(executor.map(process_single_file, file_data_list))
            
            # Process results
            for result in results:
                if result["status"] == "success":
                    processed_files.append(result["file_name"])
                    time_per_file[result["file_name"]] = result["duration"]
                    
                    # Create individual summary log
                    dt_string = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    summary_filename = f"{result['file_name']}_summary_{dt_string}.txt"
                    summary_log_path = os.path.join(LOG_DIR, summary_filename)
                    write_individual_summary_log(summary_log_path, result["file_name"], 
                                                result["file_size"], result["duration"])
                else:
                    failed_files.append((result["file_name"], result["error"]))
                    
                    # Create summary log for failed file
                    dt_string = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    summary_filename = f"{result['file_name']}_error_summary_{dt_string}.txt"
                    summary_log_path = os.path.join(LOG_DIR, summary_filename)
                    write_individual_summary_log(summary_log_path, result["file_name"], 
                                                 result["file_size"], result["duration"], 
                                                 False, result["error"])
        
        # Update checkpoint to the end of all processed files
        file_counter = start_index + len(files_to_process)
        try:
            with open(checkpoint_path, "w") as cp:
                cp.write(str(file_counter))
        except Exception as e:
            logging.error(f"Error writing to checkpoint file: {e}")

    except Exception as e:
        logging.error(f"Error accessing the directory: {e}")
        import traceback
        traceback.print_exc()

    # Final stats
    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time
    total_processed = len(processed_files)
    total_failed = len(failed_files)
    
    # Convert elapsed time to hours, minutes, and seconds
    hours, rem = divmod(total_duration, 3600)
    minutes, seconds = divmod(rem, 60)
    
    # Print final stats
    print("\n" + "=" * 80)
    print(" " * 30 + "PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total files processed: {total_processed}")
    print(f"Successfully processed: {total_processed}")
    print(f"Failed: {total_failed}")
    print(f"Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    if total_processed > 0:
        avg_time = sum(time_per_file.values()) / total_processed
        avg_hours, avg_rem = divmod(avg_time, 3600)
        avg_minutes, avg_seconds = divmod(avg_rem, 60)
        print(f"Average processing time per file: {int(avg_hours)}h {int(avg_minutes)}m {int(avg_seconds)}s")
    
    # Generate summary report
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"ato_chunking_report_{timestamp}.txt"
    report_path = os.path.join(LOG_DIR, report_filename)  # Save in log directory
    with open(report_path, "w", encoding="utf-8") as report_file:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_file.write(f"ATO Chunking Report - Generated on {current_time}\n")
        report_file.write("=" * 50 + "\n\n")
        
        report_file.write(f"Files Processed: {file_counter - start_index}\n")
        report_file.write(f"Successfully Processed: {total_processed}\n")
        report_file.write(f"Failed: {total_failed}\n")
        
        # Report processing time
        report_file.write(f"Total Processing Time: {int(hours)}h {int(minutes)}m {int(seconds)}s\n\n")
        
        if total_processed > 0:
            report_file.write(f"Average Processing Time Per File: {int(avg_hours)}h {int(avg_minutes)}m {int(avg_seconds)}s\n\n")
        
        # List successfully processed files
        if processed_files:
            report_file.write("Successfully Processed Files:\n")
            report_file.write("-" * 30 + "\n")
            for file in processed_files:
                duration = time_per_file.get(file, 0)
                f_hours, f_rem = divmod(duration, 3600)
                f_minutes, f_seconds = divmod(f_rem, 60)
                report_file.write(f"{file} - {int(f_hours)}h {int(f_minutes)}m {int(f_seconds)}s\n")
            report_file.write("\n")
        
        # Report failed files
        if failed_files:
            report_file.write("Failed Files:\n")
            report_file.write("-" * 30 + "\n")
            for file, error in failed_files:
                report_file.write(f"{file}: {error}\n")
        else:
            report_file.write("All files processed successfully.\n")
            
    print(f"Chunking completed. Report generated and saved as '{report_path}'.")

if __name__ == "__main__":
    main() 
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

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
# Set up logging with timestamp
timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE_NAME = f"log_chunking_ato_{timestamp_str}.txt"

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

# Max files to process in one run
MAX_FILES_TO_PROCESS = int(os.getenv("MAX_FILES_TO_PROCESS", "30"))

# Updated prompt for a single-chunk process with detailed instructions:
single_chunk_prompt = """Please process the attached document to extract and summarize the following, while retaining the important text and structure. Do not invent new statements unless it is necessary for clarity or to fulfill the required format.

Header/Metadata & References:
- Doc ID: [If available]
- Title: [Title of the document]
- URL: [Any URL present]
- Date: [Any relevant date]
- Legislative references, if any.

Issue & Decision:
- Identify and summarize the main issue addressed by the document and the decision or outcome. Provide additional detail and clarity in your explanation.

Facts & Reasoning:
- Extract and detail the relevant facts and background information. Describe the reasoning steps and include key examples where applicable, ensuring ample detail.

Present your answer as a single comprehensive chunk with clearly labeled sections.

---
Document content:
"""

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
    messages = [
        {"role": "system", "content": "You are an AI assistant specializing in Australian tax law and ATO rulings. Extract and organize the key information from ATO documents."},
        {"role": "user", "content": prompt + text}
    ]
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            start_time = time.time()  # Start timing
            response = openai.ChatCompletion.create(
                model="gpt-4o",  # Use the appropriate model name
                messages=messages,
                temperature=0,
                timeout=30
            )
            end_time = time.time()  # End timing
            processing_time = end_time - start_time
            logging.info(f"LLM processing completed in {processing_time:.2f} seconds")
            
            return response.choices[0].message.content
        except openai.error.OpenAIError as e:
            logging.error(f"Error communicating with OpenAI: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logging.info(f"Retrying in {wait_time} seconds... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise

def parse_metadata(chunk_text: str):
    """
    Attempt to extract metadata such as doc_id, title, URL, and date from the processed chunk.
    Fallback to defaults if not found.
    """
    doc_id = None
    title = None
    url = None
    date_info = None
    lines = chunk_text.splitlines()
    for line in lines:
        line_stripped = remove_markdown(line.strip())
        line_stripped = re.sub(r'^[\-\*\s]+', '', line_stripped).strip()
        if line_stripped.lower().startswith("doc id:"):
            doc_id = line_stripped.split(":", 1)[1].strip()
        elif line_stripped.lower().startswith("title:"):
            title = line_stripped.split(":", 1)[1].strip()
        elif line_stripped.lower().startswith("url:") or line_stripped.lower().startswith("href:"):
            url = line_stripped.split(":", 1)[1].strip()
            if url and not url.lower().startswith("http"):
                url = "https://ato.gov.au/" + url.lstrip("/")
        elif line_stripped.lower().startswith("date:") or line_stripped.lower().startswith("date of decision:"):
            date_info = line_stripped.split(":", 1)[1].strip()
    if not doc_id:
        doc_id = None  # Will use fallback file name later
    if not title:
        title = "Untitled Document"
    if not url:
        url = "No URL Provided"
    if not date_info:
        date_info = "No date info provided"
    return doc_id, title, url, date_info

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

def main():
    # Track overall start time
    overall_start_time = time.time()
    
    # Print welcome banner
    print("\n" + "=" * 80)
    print(" " * 30 + "ATO DOCUMENT CHUNKING PIPELINE")
    print("=" * 80 + "\n")
    
    logging.info("Starting the ATO document chunking pipeline.")
    
    # Directory containing the JSON files and checkpoint file
    checkpoint_path = "chunking_checkpoint.txt"  # Stored in the current working directory
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

        # Process files
        for i, file_name in enumerate(files_to_process, 1):
            file_counter = start_index + i
            file_start_time = time.time()  # Track time for this file
            
            progress_str = f"Processing file {i}/{len(files_to_process)} [{file_counter}/{total_files}]: {file_name}"
            print("\n" + "-" * len(progress_str))
            print(progress_str)
            print("-" * len(progress_str))
            
            local_file_path = os.path.join(INPUT_DIR, file_name)
            logging.info(f"Processing file: {local_file_path}")
            
            # Calculate elapsed time since start of run
            elapsed_since_start = time.time() - overall_start_time
            hours, rem = divmod(elapsed_since_start, 3600)
            minutes, seconds = divmod(rem, 60)
            print(f"Time elapsed since start: {int(hours)}h {int(minutes)}m {int(seconds)}s")
            
            file_size = os.path.getsize(local_file_path)
            
            try:
                with open(local_file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                error_msg = f"Error reading JSON file {file_name}: {e}"
                logging.error(error_msg)
                failed_files.append((file_name, error_msg))
                
                # Create summary log for failed file
                dt_string = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                summary_filename = f"{file_name}_error_summary_{dt_string}.txt"
                summary_log_path = os.path.join(INPUT_DIR, summary_filename)
                file_duration = time.time() - file_start_time
                write_individual_summary_log(summary_log_path, file_name, file_size, file_duration, False, error_msg)
                
                # Update checkpoint even on failure to skip problematic file next time
                with open(checkpoint_path, "w") as cp:
                    cp.write(str(file_counter))
                continue
            
            # Extract title and URL from the JSON file
            file_info = data.get("file_info", {})
            file_title = file_info.get("title", "Untitled Document")
            file_href = file_info.get("a_attr", {}).get("href", "")
            if file_href:
                if not file_href.lower().startswith("http"):
                    # Remove any leading slash from href and join with base URL
                    full_url = "https://ato.gov.au" + file_href if file_href.startswith("/") else "https://ato.gov.au/" + file_href
                else:
                    full_url = file_href
            else:
                full_url = "No URL Provided"
            
            # Convert the entire JSON content to a string
            json_content = json.dumps(data, indent=2)
            
            # Call the LLM with the entire JSON content
            try:
                logging.info(f"Sending document to LLM for processing...")
                llm_response = call_llm(json_content, single_chunk_prompt)
            except Exception as e:
                error_msg = f"LLM call failed for file {file_name}: {e}"
                logging.error(error_msg)
                failed_files.append((file_name, error_msg))
                
                # Create summary log for failed file
                dt_string = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                summary_filename = f"{file_name}_error_summary_{dt_string}.txt"
                summary_log_path = os.path.join(INPUT_DIR, summary_filename)
                file_duration = time.time() - file_start_time
                write_individual_summary_log(summary_log_path, file_name, file_size, file_duration, False, error_msg)
                
                with open(checkpoint_path, "w") as cp:
                    cp.write(str(file_counter))
                continue
            
            logging.info(f"Received response from LLM for single-chunk processing.")
            processed_chunk = llm_response.strip()
            
            debug_print("Processed Chunk (first 200 chars):\n" + processed_chunk[:200] + "...")
            
            # Parse metadata from the processed chunk
            extracted_doc_id, extracted_title, extracted_url, extracted_date_info = parse_metadata(processed_chunk)
            logging.info(f"Extracted metadata: doc_id={extracted_doc_id}, title={extracted_title}")
            
            # Use fallback file-level metadata if LLM extraction returned defaults or if the URL is relative/missing
            if not extracted_doc_id:
                extracted_doc_id = file_name.rsplit(".", 1)[0]
            if extracted_title == "Untitled Document":
                extracted_title = file_title
            if extracted_url == "No URL Provided" or not extracted_url:
                extracted_url = full_url
            elif not extracted_url.lower().startswith("http"):
                extracted_url = "https://ato.gov.au" + extracted_url
            
            # Create output JSON structure
            output_data = {
                "doc_id": extracted_doc_id,
                "chunk_text": processed_chunk,
                "date_info": extracted_date_info,
                "title": extracted_title,
                "url": extracted_url,
                "document_type": "ato_ruling",  # Adding document type for namespace determination
                "processing_date": datetime.datetime.now().isoformat()
            }
            
            # Save processed JSON
            output_filename = f"ATO_{extracted_doc_id}.json"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            try:
                with open(output_path, "w", encoding="utf-8") as outf:
                    json.dump(output_data, outf, indent=2)
                logging.info(f"Saved processed file to {output_path}")
                processed_files.append(file_name)
                
                # Calculate processing time for this file
                file_end_time = time.time()
                file_duration = file_end_time - file_start_time
                time_per_file[file_name] = file_duration
                
                # Convert time to display format
                hours, rem = divmod(file_duration, 3600)
                minutes, seconds = divmod(rem, 60)
                print(f"Completed processing {file_name} in {int(hours)}h {int(minutes)}m {int(seconds)}s")
                
                # Create individual summary log
                dt_string = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                summary_filename = f"{file_name}_summary_{dt_string}.txt"
                summary_log_path = os.path.join(INPUT_DIR, summary_filename)
                write_individual_summary_log(summary_log_path, file_name, file_size, file_duration)
                
            except Exception as e:
                error_msg = f"Error saving processed file {output_filename}: {e}"
                logging.error(error_msg)
                failed_files.append((file_name, error_msg))
                
                # Create summary log for failed file
                dt_string = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                summary_filename = f"{file_name}_error_summary_{dt_string}.txt"
                summary_log_path = os.path.join(INPUT_DIR, summary_filename)
                file_duration = time.time() - file_start_time
                write_individual_summary_log(summary_log_path, file_name, file_size, file_duration, False, error_msg)
            
            # Update checkpoint file after processing each file (successful or not)
            try:
                with open(checkpoint_path, "w") as cp:
                    cp.write(str(file_counter))
            except Exception as e:
                logging.error(f"Error writing to checkpoint file: {e}")
            
            # Show elapsed time since start of run
            total_elapsed = time.time() - overall_start_time
            t_hours, t_rem = divmod(total_elapsed, 3600)
            t_minutes, t_seconds = divmod(t_rem, 60)
            print(f"Total time elapsed since start: {int(t_hours)}h {int(t_minutes)}m {int(t_seconds)}s")
    
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
    with open(report_filename, "w", encoding="utf-8") as report_file:
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
            
    print(f"Chunking completed. Report generated and saved as '{report_filename}'.")

if __name__ == "__main__":
    main() 
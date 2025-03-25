#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ATO Document Processor with Structured Extraction and Keyword Generation

- Loads JSON files containing ATO rulings from a specified directory.
- Extracts key information directly from the JSON structure.
- Identifies metadata like Doc ID, Title, URL, and Date.
- Uses GPT-4o to extract relevant keywords and categories from tax taxonomy.
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
import concurrent.futures

# Load environment variables
load_dotenv()

# Set OpenAI API key from environment variable
openai.api_key = os.environ.get("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable not found. Please check your .env file.")

# Load keyword library from the same file used by the law chunking script
try:
    with open('tax_keywords.json', 'r', encoding='utf-8') as kw_file:
        keyword_dict = json.load(kw_file)
        
    # Create mappings for case-insensitive matching
    KEYWORD_TO_CATEGORY = {}
    KEYWORD_ORIGINAL_CASE = {}

    for category, keywords in keyword_dict.items():
        category_lower = category.lower()
        KEYWORD_ORIGINAL_CASE[category_lower] = category
        KEYWORD_TO_CATEGORY[category_lower] = category
        for keyword in keywords:
            keyword_lower = keyword.lower()
            KEYWORD_ORIGINAL_CASE[keyword_lower] = keyword
            KEYWORD_TO_CATEGORY[keyword_lower] = category

    # Define TAX_KEYWORDS as a flat list of all keywords and categories from the keyword library
    TAX_KEYWORDS = list(KEYWORD_ORIGINAL_CASE.values())
    
    logging.info(f"Successfully loaded tax keyword library with {len(TAX_KEYWORDS)} total terms")
except Exception as e:
    logging.error(f"Error loading tax_keywords.json: {e}")
    TAX_KEYWORDS = []
    keyword_dict = {}
    KEYWORD_TO_CATEGORY = {}
    KEYWORD_ORIGINAL_CASE = {}

# --- CONFIGURATION ---
# Create local log directory
LOG_DIR = "/Users/kenmacpro/pinecone-upsert/logs/chunking"
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

# Directory paths
INPUT_DIR = os.getenv("ATO_INPUT_DIR", "/Users/kenmacpro/pinecone-upsert/testfiles_cgt/input")
OUTPUT_DIR = os.getenv("JSON_OUTPUT_DIR", "/Users/kenmacpro/pinecone-upsert/testfiles_cgt/output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Max files to process in one run (0 = no limit)
MAX_FILES_TO_PROCESS = int(os.getenv("MAX_FILES_TO_PROCESS", "0"))

# Number of parallel workers (adjust based on your API rate limits)
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "5"))

def extract_keywords_with_llm(text, keyword_library, doc_id=None, source_file=None):
    """
    Extract relevant keywords and categories from text using LLM.
    This function is aligned with the approach in Step1_chunking_law.py.
    """
    # Log section and source information
    if doc_id and source_file:
        logging.info(f"\nProcessing ATO ID {doc_id} from {source_file}")
    
    prompt = f"""
    Extract relevant keywords from the following text based on the provided keyword library.

    Text:
    {text}

    Keyword Library:
    {', '.join(keyword_library)}

    Return keywords as a comma-separated list.
    """
    
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            # Attempt to call the OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100,
                request_timeout=60  # Set a timeout of 60 seconds
            )
            
            # Extract keywords from the response with error handling
            try:
                if isinstance(response, dict) and 'choices' in response and len(response['choices']) > 0:
                    keywords_text = response['choices'][0]['message']['content'].strip()
                else:
                    logging.warning("Unexpected response structure")
                    return [], []
                
                # Log the raw keywords response for debugging
                logging.info(f"Raw keywords response: {keywords_text}")
                
                # Parse the comma-separated keywords
                raw_keywords = [k.strip() for k in keywords_text.split(',') if k.strip()]
                logging.info(f"Raw keywords after splitting: {raw_keywords}")
                
                # Case-insensitive keyword matching
                valid_keywords = set()
                matched_categories = set()
                
                for keyword in raw_keywords:
                    keyword_lower = keyword.lower()

                    if keyword_lower in KEYWORD_TO_CATEGORY:
                        original_keyword = KEYWORD_ORIGINAL_CASE[keyword_lower]
                        category = KEYWORD_TO_CATEGORY[keyword_lower]
                        valid_keywords.add(original_keyword)
                        matched_categories.add(category)
                        logging.info(f"Matched keyword '{original_keyword}' under category '{category}'")
                    else:
                        # Partial matching fallback
                        for lib_keyword_lower in KEYWORD_TO_CATEGORY:
                            if keyword_lower in lib_keyword_lower or lib_keyword_lower in keyword_lower:
                                original_keyword = KEYWORD_ORIGINAL_CASE[lib_keyword_lower]
                                category = KEYWORD_TO_CATEGORY[lib_keyword_lower]
                                valid_keywords.add(original_keyword)
                                matched_categories.add(category)
                                logging.info(f"Partial match '{keyword}' → '{original_keyword}' under category '{category}'")
                                break
                
                # Convert sets to sorted lists for consistent ordering
                valid_keywords = sorted(valid_keywords)
                matched_categories = sorted(matched_categories)
                
                # Add doc_id to keywords if it's not already there
                if doc_id and doc_id not in valid_keywords:
                    valid_keywords.append(doc_id)
                    logging.info(f"Added doc_id '{doc_id}' to keywords")
                
                logging.info(f"FINAL KEYWORDS: {len(valid_keywords)} keywords: {valid_keywords}")
                return valid_keywords, matched_categories
                
            except Exception as parsing_error:
                logging.error(f"Error parsing OpenAI response: {str(parsing_error)}")
                logging.error(f"Response content: {str(response)[:500]}...")
                return [], []
            
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(f"OpenAI API error (attempt {attempt+1}/{max_retries}): {str(e)}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                logging.error(f"Failed to extract keywords after {max_retries} attempts: {str(e)}")
                return [], []

def extract_data_from_json(file_path):
    """
    Extract document data directly from the JSON structure.
    Focus on extracting title, href (URL), page_md, file_name.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        result = {
            "doc_id": None,
            "title": None,
            "url": None,
            "content": None,
            "file_name": None,
            "date_info": None
        }
        
        # Extract file_name
        if 'file_name' in data:
            result["file_name"] = data['file_name']
        else:
            result["file_name"] = os.path.basename(file_path)
        
        # Try to extract doc_id from various sources
        if 'file_info' in data and isinstance(data['file_info'], dict) and 'title' in data['file_info']:
            # Try to extract doc ID from title before " -" 
            # Handles formats like:
            # - "ATO ID 2002/239 - ..."
            # - "TD 60 - ..."
            # - "CR 2020/6 - ..."
            # - "TD 2000/14 - ..."
            # - "TR 2010/4 - ..."
            title = data['file_info']['title']
            if ' - ' in title:
                doc_id = title.split(' - ')[0].strip()
                result["doc_id"] = doc_id
                logging.info(f"Extracted doc_id '{result['doc_id']}' from title")

        # Try file_name if doc_id not found
        if not result["doc_id"] and 'file_name' in data and data['file_name']:
            # Try patterns like "AID|AID201126|00001.json"
            file_parts = data['file_name'].split('|')
            if len(file_parts) >= 2:
                result["doc_id"] = file_parts[1]
                logging.info(f"Extracted doc_id '{result['doc_id']}' from file_name field")
        
        # Try URL in file_info if still not found
        if not result["doc_id"] and 'file_info' in data and isinstance(data['file_info'], dict):
            if 'a_attr' in data['file_info'] and 'href' in data['file_info']['a_attr']:
                href = data['file_info']['a_attr']['href']
                # Extract from patterns like "/law/view/document?docid=AID/AID201126/00001"
                if 'docid=' in href:
                    docid_part = href.split('docid=')[1]
                    doc_parts = docid_part.split('/')
                    if len(doc_parts) >= 2:
                        result["doc_id"] = doc_parts[1]
                        logging.info(f"Extracted doc_id '{result['doc_id']}' from URL")
        
        # If doc_id still not found, use filename as fallback
        if not result["doc_id"]:
            result["doc_id"] = os.path.basename(file_path).replace('.json', '')
            logging.info(f"Using filename '{result['doc_id']}' as doc_id fallback")
        
        # Extract title
        if 'file_info' in data and isinstance(data['file_info'], dict) and 'title' in data['file_info']:
            result["title"] = data['file_info']['title']
        
        # Extract URL from href
        if 'file_info' in data and isinstance(data['file_info'], dict) and 'a_attr' in data['file_info']:
            if 'href' in data['file_info']['a_attr']:
                href = data['file_info']['a_attr']['href']
                result["url"] = "https://ato.gov.au" + href if href.startswith('/') else href
        
        # Extract content for keyword analysis
        # Try different fields in order of preference
        if 'page_md' in data and data['page_md']:
            result["content"] = data['page_md']
        elif 'content' in data and data['content']:
            result["content"] = data['content']
        elif 'text' in data and data['text']:
            result["content"] = data['text']
        elif 'file_info' in data and isinstance(data['file_info'], dict):
            if 'content' in data['file_info'] and data['file_info']['content']:
                result["content"] = data['file_info']['content']
        elif 'html' in data and data['html']:
            content = data['html']
            result["content"] = re.sub(r'<[^>]+>', ' ', content)  # Simple HTML tag removal
        
        # Try to extract date information, if available
        if 'date' in data:
            result["date_info"] = data['date']
        elif 'metadata' in data and 'date' in data['metadata']:
            result["date_info"] = data['metadata']['date']
        
        # If content is too small, enrich with metadata
        if not result["content"] or len(result["content"].strip()) < 50:
            enriched_content = "ATO Document Reference\n\n"
            if result["title"]:
                enriched_content += f"Title: {result['title']}\n"
            if result["doc_id"]:
                enriched_content += f"Document ID: {result['doc_id']}\n"
            if result["url"]:
                enriched_content += f"URL: {result['url']}\n"
            if 'path_info' in data and isinstance(data['path_info'], list):
                enriched_content += "\nCategories:\n"
                for item in data['path_info']:
                    if isinstance(item, dict) and 'title' in item:
                        enriched_content += f"- {item['title']}\n"
            result["content"] = enriched_content
        
        return result
    
    except Exception as e:
        logging.error(f"Error extracting data from JSON: {e}")
        raise

def write_individual_summary_log(log_file, filename, file_size, processing_time, success=True, error_message=None):
    """Write an individual summary log for an ATO file."""
    # Convert time_taken to hours, minutes, and seconds
    hours, rem = divmod(processing_time, 3600)
    minutes, seconds = divmod(rem, 60)
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Processing Summary for {filename}\n")
        f.write("=" * 40 + "\n")
        f.write(f"Processing Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input File Size: {file_size} bytes\n")
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
    """Process a single file."""
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
        # Extract structured data from the JSON file
        file_size = os.path.getsize(local_file_path)
        extracted_data = extract_data_from_json(local_file_path)
        
        # Extract keywords and categories using LLM
        start_llm_time = time.time()
        keywords, categories = extract_keywords_with_llm(
            extracted_data["content"], 
            TAX_KEYWORDS, 
            extracted_data["doc_id"], 
            file_name
        )
        llm_processing_time = time.time() - start_llm_time
        print(f"LLM processing time for keywords extraction: {llm_processing_time:.2f} seconds")
        
        # Format the document ID for output
        if not extracted_data["doc_id"].upper().startswith("ATO"):
            doc_id_formatted = f"ATO ID {extracted_data['doc_id']}"
        else:
            doc_id_formatted = extracted_data["doc_id"]
        
        # Create the output JSON structure
        output_data = {
            "title": extracted_data["title"] or "Unknown Title",
            "url": extracted_data["url"] or "",
            "text": extracted_data["content"],
            "doc_id": doc_id_formatted,
            "document_type": "ato",
            "metadata": {
                "keywords": keywords,
                "categories": categories,
                "full_reference": f"ATO ID {extracted_data['doc_id'].replace('ATO ID ', '')}" + f" {extracted_data['title'] or 'Unknown Title'}",
                "source_file": extracted_data["file_name"] or file_name,
                "creation_date": datetime.datetime.now().isoformat()
            }
        }
        
        # Create a sanitized output filename
        output_filename = f"ATO_{sanitize_filename(extracted_data['doc_id'])}.json"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Save the processed file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logging.info(f"Successfully saved JSON file: {output_path}")
        print(f"Output saved to: {output_filename}")
        
        # Print debug information for keywords in test mode
        if DEBUG:
            print(f"Keywords for this document: {keywords}")
            print(f"Categories for this document: {categories}")
        
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
        import traceback
        traceback.print_exc()
        file_duration = time.time() - file_start_time
        
        return {
            "status": "failed",
            "file_name": file_name,
            "error": str(e),
            "file_size": os.path.getsize(local_file_path) if os.path.exists(local_file_path) else 0,
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
    logging.info(f"Tax Keywords library loaded with {len(TAX_KEYWORDS)} terms")
    
    # Directory containing the JSON files and checkpoint file
    checkpoint_path = os.path.join(LOG_DIR, "chunking_ato_checkpoint.txt")  # Store checkpoint in log directory
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
    
    # Ask if user wants to run in test mode
    test_mode = input("Run in test mode (process only first 10 files)? (yes/no): ").strip().lower() == 'yes'
    if test_mode:
        MAX_FILES_TO_PROCESS = 10
        logging.info("TEST MODE ACTIVE: Processing only the first 10 files.")
        print("\n*** TEST MODE ENABLED - Only processing first 10 files ***\n")
    
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
    print(f"Total files processed: {total_processed + total_failed}")
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
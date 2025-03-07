import os
import re
import json
import openai
import datetime
import logging
import time
from dotenv import load_dotenv

load_dotenv()

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION ---
openai.api_key = os.getenv("OPENAI_API_KEY")

# Directory for input and output files
INPUT_DIR = "/Users/kenmacpro/pinecone-upsert/testfiles_cgt"  # Directory with raw ATO files
OUTPUT_DIR = "/Users/kenmacpro/pinecone-upsert/testfiles_law/json"  # Where to save processed JSON
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt + text}
    ]
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",  # Use the appropriate model name
                messages=messages,
                temperature=0,
                timeout=30
            )
            return response.choices[0].message.content
        except openai.error.OpenAIError as e:
            logging.error(f"Error communicating with OpenAI: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logging.info(f"Retrying in {wait_time} seconds...")
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

def main():
    logging.info("Starting the ATO document chunking pipeline.")
    
    # Directory containing the JSON files and checkpoint file
    checkpoint_path = "chunking_checkpoint.txt"  # Stored in the current working directory
    failed_files = []  # List to store file names that failed processing
    
    # Ask user if they want to reset the checkpoint
    reset_checkpoint = input("Do you want to reset the checkpoint? (yes/no): ").strip().lower()
    if reset_checkpoint == 'yes':
        start_index = 0
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

        # Process up to 30 files starting from the checkpoint
        for file_name in files[start_index : start_index + 30]:
            file_counter += 1
            print(f"Processing file {file_counter}/{start_index + 30}: {file_name}")
            local_file_path = os.path.join(INPUT_DIR, file_name)
            logging.info(f"Processing file: {local_file_path}")
            
            try:
                with open(local_file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logging.error(f"Error reading JSON file {file_name}: {e}")
                failed_files.append(file_name)
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
                llm_response = call_llm(json_content, single_chunk_prompt)
            except Exception as e:
                logging.error(f"LLM call failed for file {file_name}: {e}")
                failed_files.append(file_name)
                with open(checkpoint_path, "w") as cp:
                    cp.write(str(file_counter))
                continue
            logging.info("Received response from LLM for single-chunk processing.")
            processed_chunk = llm_response.strip()
            print("Processed Chunk:\n", processed_chunk)
            
            # Parse metadata from the processed chunk
            extracted_doc_id, extracted_title, extracted_url, extracted_date_info = parse_metadata(processed_chunk)
            logging.info(f"Extracted metadata for file {file_name}: doc_id={extracted_doc_id}, title={extracted_title}, url={extracted_url}, date_info={extracted_date_info}")
            
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
            except Exception as e:
                logging.error(f"Error saving processed file {output_filename}: {e}")
                failed_files.append(file_name)
            
            # Update checkpoint file after processing each file (successful or not)
            try:
                with open(checkpoint_path, "w") as cp:
                    cp.write(str(file_counter))
            except Exception as e:
                logging.error(f"Error writing to checkpoint file: {e}")
    
    except Exception as e:
        logging.error(f"Error accessing the directory: {e}")

    # Generate summary report
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"ato_chunking_report_{timestamp}.txt"
    with open(report_filename, "w", encoding="utf-8") as report_file:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_file.write(f"ATO Chunking Report - Generated on {current_time}\n")
        report_file.write(f"Files Processed: {file_counter - start_index}\n")
        report_file.write("=" * 50 + "\n\n")
        
        # Report failed files
        if failed_files:
            report_file.write("Failed Files:\n")
            for failed in failed_files:
                report_file.write(f"{failed}\n")
        else:
            report_file.write("All files processed successfully.\n")
            
    print(f"Chunking completed. Report generated and saved as '{report_filename}'.")

if __name__ == "__main__":
    main() 
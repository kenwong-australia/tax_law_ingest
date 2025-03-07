import os
import re
import json
import openai
import datetime
import logging
import time
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()


# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION ---
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
index_name = os.getenv("PINECONE_INDEX")
VECTOR_DIMENSION = 1536  # For the embedding vector

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

def init_pinecone(api_key: str) -> Pinecone:
    """Initialize Pinecone with the given API key."""
    return Pinecone(api_key=api_key)

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

def get_embedding(text: str, model="text-embedding-ada-002") -> list:
    """
    Obtain an embedding vector for the given text using OpenAI's Embeddings API.
    """
    response = openai.Embedding.create(input=[text], model=model)
    return response['data'][0]['embedding']

def main():
    logging.info("Starting the single-chunk processing pipeline.")
    
    # Initialize Pinecone and create index if it doesn't exist
    try:
        pc = init_pinecone(pinecone_api_key)
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=VECTOR_DIMENSION,
                metric='euclidean',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
        index = pc.Index(index_name)
        logging.info("Initialized Pinecone.")
    except Exception as e:
        logging.error(f"Error initializing Pinecone: {e}")
        return

    # Directory containing the JSON files and checkpoint file
    test_files_directory = "/Users/kenmacpro/pinecone-upsert/testfiles_cgt"
    checkpoint_path = "checkpoint.txt"  # Stored in the current working directory
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
        files = sorted([f for f in os.listdir(test_files_directory) if f.lower().endswith('.json')])
        if not files:
            logging.error("No JSON files found in the test files directory.")
            return

        # Process up to 30 files starting from the checkpoint
        for file_name in files[start_index : start_index + 30]:
            file_counter += 1
            print(f"Processing file {file_counter}/{start_index + 30}: {file_name}")
            local_file_path = os.path.join(test_files_directory, file_name)
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
            
            try:
                embedding = get_embedding(processed_chunk)
                logging.info(f"Successfully obtained embedding for file {file_name}.")
            except Exception as e:
                logging.error(f"Error obtaining embedding for file {file_name}: {e}")
                failed_files.append(file_name)
                with open(checkpoint_path, "w") as cp:
                    cp.write(str(file_counter))
                continue
            
            metadata = {
                "doc_id": extracted_doc_id,
                "chunk_text": processed_chunk,
                "upsert_date": datetime.datetime.now().isoformat(),
                "date_info": extracted_date_info,
                "title": extracted_title,
                "url": extracted_url
            }
            
            vector = {
                "id": f"{extracted_doc_id}",
                "values": embedding,
                "metadata": metadata
            }
            
            try:
                upsert_response = index.upsert(vectors=[vector])
                logging.info(f"Upserted vector for file {file_name} to Pinecone.")
                print("Upsert response:", upsert_response)
            except Exception as e:
                logging.error(f"Error during upsert for file {file_name}: {e}")
                failed_files.append(file_name)
            
            # Update checkpoint file after processing each file (successful or not)
            try:
                with open(checkpoint_path, "w") as cp:
                    cp.write(str(file_counter))
            except Exception as e:
                logging.error(f"Error writing to checkpoint file: {e}")
    
    except Exception as e:
        logging.error(f"Error accessing the test files directory: {e}")

    # --- QUERY & REPORT GENERATION ---
    metadata_filter = {
        "$or": [
            {"source_url": {"$ne": ""}},
            {"section_url": {"$ne": ""}}
        ]
    }
    dummy_vector = [0.0] * VECTOR_DIMENSION
    query_response = index.query(
        vector=dummy_vector,
        filter=metadata_filter,
        include_metadata=True,
        include_values=False,
        top_k=10000
    )
    matches = query_response.get("matches", [])
    if not matches:
        print("No matching records found.")
    else:
        print(f"Total matches found: {len(matches)}")
        # Limit reporting to the first 30 matches
        for match in matches[:30]:
            metadata = match.get('metadata', {})
            print(f"Match ID: {match.get('id', 'N/A')}, Document ID: {metadata.get('doc_id', 'N/A')}")
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"pinecone_report_ato_{timestamp}.txt"
        with open(report_filename, "w", encoding="utf-8") as report_file:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            report_file.write(f"Pinecone Report - Generated on {current_time}\n")
            report_file.write(f"Total Records: {len(matches)}\n")
            report_file.write("=" * 50 + "\n\n")
            for match in matches[:30]:
                metadata = match.get("metadata", {})
                report_file.write("--------------------------------------------------\n")
                report_file.write(f"ID: {match.get('id', 'N/A')}\n")
                report_file.write(f"Score: {match.get('score', 'N/A')}\n")
                report_file.write(f"Chunk Text: {metadata.get('chunk_text', 'N/A')}\n")
                report_file.write("-- Metadata --\n")
                report_file.write(f"Date Info: {metadata.get('date_info', 'N/A')}\n")
                report_file.write(f"Document ID: {metadata.get('doc_id', 'N/A')}\n")
                report_file.write(f"Title: {metadata.get('title', 'N/A')}\n")
                report_file.write(f"URL: {metadata.get('url', 'N/A')}\n")
                report_file.write(f"Upsert Date: {metadata.get('upsert_date', 'N/A')}\n")
                report_file.write("\n")
            # Append failed files list to the report
            if failed_files:
                report_file.write("--------------------------------------------------\n")
                report_file.write("Failed Files:\n")
                for failed in failed_files:
                    report_file.write(f"{failed}\n")
                report_file.write("\n")
        print(f"Report generated and saved as '{report_filename}'.")
    

if __name__ == "__main__":
    main()

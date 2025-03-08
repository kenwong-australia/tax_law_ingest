import openai
from pinecone import Pinecone, ServerlessSpec
import logging
import datetime
import time
import re
from dotenv import load_dotenv
import os
load_dotenv()


# --- CONFIGURATION ---
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
index_name = os.getenv("PINECONE_INDEX")
VECTOR_DIMENSION = 3072  # For the embedding vector - text-embedding-3-large uses 3072 dimensions

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_embedding(text: str, model="text-embedding-3-large") -> list:
    """Obtain an embedding vector for the given text."""
    response = openai.Embedding.create(input=[text], model=model)
    return response['data'][0]['embedding']

def init_pinecone(api_key: str):
    """Initialize Pinecone and return the index instance."""
    # Create Pinecone client
    pc = Pinecone(api_key=api_key)
    
    # Get the index
    if index_name not in pc.list_indexes().names():
        logging.warning(f"Index {index_name} not found. Creating new index...")
        pc.create_index(
            name=index_name,
            dimension=VECTOR_DIMENSION,
            metric='cosine'
        )
    
    # Return the index
    return pc.Index(index_name)

def query_pinecone(index, query_vector, top_k=10, namespace=None):
    """Query the Pinecone index with the provided vector and return matches with metadata."""
    response = index.query(
        vector=query_vector, 
        top_k=top_k, 
        include_metadata=True,
        namespace=namespace
    )
    return response.get("matches", [])

def call_llm_with_context(query: str, context: str) -> str:
    """
    Calls the LLM with the provided user query and context.
    The context is built by concatenating the retrieved chunk texts.
    Logs how long the GPT call took.
    """
    prompt = (
        f"Using the following context, answer the question:\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context}"
    )
    messages = [
        {"role": "system", "content": "You are a helpful AI Australian tax advisor."},
        {"role": "user", "content": prompt}
    ]
    max_retries = 5
    for attempt in range(max_retries):
        try:
            start_time = time.time()  # START timing
            response = openai.ChatCompletion.create(
                model="gpt-4o",  # Adjust model name as needed
                messages=messages,
                temperature=0,
                timeout=30
            )
            end_time = time.time()    # END timing
            duration = end_time - start_time
            logging.info(f"GPT primary call took {duration:.2f} seconds.")

            return response.choices[0].message.content.strip()
        except openai.error.OpenAIError as e:
            logging.error(f"Error communicating with OpenAI: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logging.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise

def identify_used_chunks_gpt(answer: str, matches: list) -> list:
    """
    Makes a second GPT call to determine which retrieved chunks
    were most relevant/used in the final answer.
    Logs how long the GPT call took.
    Returns a list of chunk indices or metadata references.
    """
    # Build a readable list of chunks for GPT to examine
    chunk_list_str = ""
    for i, match in enumerate(matches, start=1):
        metadata = match.get("metadata", {})
        chunk_text = metadata.get("chunk_text", "")
        doc_id = metadata.get("doc_id", f"chunk_{i}")
        chunk_list_str += f"{i}. (Doc ID: {doc_id}) {chunk_text}\n\n"

    # Now we ask GPT which items from the chunk list were used
    prompt = f"""
Here is the final answer to a tax question:
\"\"\"{answer}\"\"\"

Below are the retrieved chunks that were made available as context:

{chunk_list_str}

Which of the above chunks contributed directly to the final answer? 
Please respond with the *numbers or doc_ids of the chunks* that were essential in forming the answer. 
Only include the ones that are truly relevant.
"""
    messages = [
        {"role": "system", "content": "You are an assistant helping to identify the references used."},
        {"role": "user", "content": prompt}
    ]

    max_retries = 5
    used_str = ""  # Keep track of GPT's response
    for attempt in range(max_retries):
        try:
            start_time = time.time()  # START timing
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=messages,
                temperature=0,
                timeout=30
            )
            end_time = time.time()    # END timing
            duration = end_time - start_time
            logging.info(f"GPT chunk-identification call took {duration:.2f} seconds.")

            used_str = response.choices[0].message.content.strip()
            break
        except openai.error.OpenAIError as e:
            logging.error(f"Error communicating with OpenAI for chunk identification: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logging.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise

    # Parse GPT's response for chunk references
    used_indices = []
    for i, match in enumerate(matches, start=1):
        doc_id = match.get("metadata", {}).get("doc_id", f"chunk_{i}")
        # If GPT mentions either the numeric index or the doc_id, consider it "used"
        if str(i) in used_str or doc_id in used_str:
            used_indices.append(i)

    return used_indices

def safe_print(text):
    """Print text safely, escaping any special shell characters."""
    # Replace any potentially problematic characters
    text = text.replace('?', '\\?')
    text = text.replace('*', '\\*')
    text = text.replace('[', '\\[')
    text = text.replace(']', '\\]')
    text = text.replace('-', '\\-')
    print(text)

def main():
    # Initialize Pinecone
    index = init_pinecone(pinecone_api_key)
    
    # Ask user which namespace to query
    print("\nWhich data source would you like to query?")
    print("1. Legislation (default)")
    print("2. ATO Rulings")
    print("3. Both (combined results)")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    # Set namespace based on user choice
    namespace = None  # Default to legislation (no namespace)
    if choice == "2":
        namespace = "ato"
        print("Querying ATO Rulings...")
    elif choice == "3":
        print("Querying both Legislation and ATO Rulings...")
    else:  # Default or choice "1"
        print("Querying Legislation...")
    
    # Read user query from console
    user_query = input("\nEnter your query: ").strip()
    if not user_query:
        print("No query provided.")
        return

    # Get embedding for the query
    query_vector = get_embedding(user_query)
    
    # Handle the combined query case
    if choice == "3":
        # Query both namespaces and combine results
        legislation_matches = query_pinecone(index, query_vector, top_k=5)  # No namespace for legislation
        ato_matches = query_pinecone(index, query_vector, top_k=5, namespace="ato")
        
        # Combine matches, sorting by score
        matches = legislation_matches + ato_matches
        matches.sort(key=lambda x: x.get("score", 0), reverse=True)
        matches = matches[:10]  # Keep top 10 overall
    else:
        # Query Pinecone for matching chunks with specified namespace
        matches = query_pinecone(index, query_vector, top_k=10, namespace=namespace)
    
    if not matches:
        print("No matching records found.")
        return

    # Build a context string by concatenating chunk texts
    context = "\n".join(match["metadata"].get("chunk_text", "") for match in matches)
    
    # 1) Primary GPT call - generate final answer
    answer = call_llm_with_context(user_query, context)
    
    # 2) Secondary GPT call - identify which chunks were used
    used_chunk_indices = identify_used_chunks_gpt(answer, matches)

    # Print the final answer
    print("\nAnswer:")
    # Split answer into lines and print each line separately to avoid shell interpretation
    answer_lines = answer.split('\n')
    for line in answer_lines:
        safe_print(line)
    
    # Print only a simple list of sections used as references
    print("\nRelevant Sections:")
    for i in used_chunk_indices:
        # 'i' is 1-based; matches is 0-based
        match = matches[i-1]
        metadata = match.get("metadata", {})
        is_ato = choice == "2" or (choice == "3" and "ato" in str(metadata.get('doc_id', '')))
        source = "ATO Ruling" if is_ato else "Legislation"
        
        # Get the section reference - prefer full_reference, fall back to section, then doc_id
        section_ref = metadata.get("full_reference", metadata.get("section", metadata.get("doc_id", "Unknown")))
        
        # Include URL for ATO rulings
        if is_ato:
            url = metadata.get("url", "No URL available")
            safe_print(f"* {source}: {section_ref}")  # Using asterisk instead of dash or bullet
            safe_print(f"  URL: {url}")
        else:
            safe_print(f"* {source}: {section_ref}")  # Using asterisk instead of dash or bullet

if __name__ == "__main__":
    main()

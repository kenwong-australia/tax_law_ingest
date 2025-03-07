import openai
import pinecone
import logging
import datetime
import time
from dotenv import load_dotenv
import os
load_dotenv()


# --- CONFIGURATION ---
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
index_name = os.getenv("PINECONE_INDEX")
VECTOR_DIMENSION = 1536  # For the embedding vector

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_embedding(text: str, model="text-embedding-ada-002") -> list:
    """Obtain an embedding vector for the given text."""
    response = openai.Embedding.create(input=[text], model=model)
    return response['data'][0]['embedding']

def init_pinecone(api_key: str):
    """Initialize Pinecone and return the index instance."""
    pinecone.init(api_key=api_key)
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=VECTOR_DIMENSION)
    return pinecone.Index(index_name)

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

def main():
    # Initialize Pinecone
    index = init_pinecone(pinecone_api_key)
    
    # Read user query from console
    user_query = input("Enter your query: ").strip()
    if not user_query:
        print("No query provided.")
        return

    # Get embedding for the query
    query_vector = get_embedding(user_query)
    
    # Query Pinecone for matching chunks
    matches = query_pinecone(index, query_vector, top_k=10, namespace="ato")
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
    print("\nAnswer:\n", answer)

    # Print only the used chunks as references
    print("\nUsed/Relevant Chunks:")
    for i in used_chunk_indices:
        # 'i' is 1-based; matches is 0-based
        match = matches[i-1]
        metadata = match.get("metadata", {})
        print("-----")
        print(f"Doc ID: {metadata.get('doc_id', 'N/A')}")
        print(f"Title: {metadata.get('title', 'N/A')}")
        print(f"URL: {metadata.get('url', 'N/A')}")
        print(f"Chunk Text: {metadata.get('chunk_text', 'N/A')}\n")

if __name__ == "__main__":
    main()

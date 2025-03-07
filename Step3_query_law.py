import openai
import pinecone
import logging
import time
import os
from dotenv import load_dotenv

load_dotenv()


# --- CONFIGURATION ---
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
index_name = os.getenv("PINECONE_INDEX")
VECTOR_DIMENSION = 3072  # For the "text-embedding-3-large" model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------------------
# Embedding Function
# -----------------------------
def get_embedding(text: str, model="text-embedding-3-large") -> list:
    """Obtain an embedding vector for the given text."""
    response = openai.Embedding.create(input=[text], model=model)
    return response['data'][0]['embedding']

# -----------------------------
# Pinecone Init + Query
# -----------------------------
def init_pinecone(api_key: str, env: str, index_name: str) -> pinecone.Index:
    """
    Initialize Pinecone and return the index instance.
    """
    pinecone.init(api_key=api_key, environment=env)
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=VECTOR_DIMENSION)
    return pinecone.Index(index_name)

def query_pinecone(index, query_vector, top_k=5):
    """
    Query the Pinecone index with the provided vector and return
    the top_k matches with metadata.
    """
    response = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    return response.get("matches", [])

# -----------------------------
# GPT Calls
# -----------------------------
def call_llm_with_context(query: str, context: str) -> str:
    """
    Primary GPT call:
    Uses the chunked text 'context' to produce a final answer.
    """
    prompt = (
        f"Answer the following question using the context provided.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context}\n\n"
        "Please provide a concise, direct response referencing the context where relevant."
    )
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant for Australian tax legislation."},
        {"role": "user", "content": prompt}
    ]
    
    start_time = time.time()
    response = openai.ChatCompletion.create(
        model="gpt-4o",  # Using your specialized GPT model
        messages=messages,
        temperature=0,
        timeout=30
    )
    end_time = time.time()
    logging.info(f"Primary GPT call took {end_time - start_time:.2f} seconds.")
    
    return response.choices[0].message.content.strip()

def identify_used_chunks_gpt(final_answer: str, matches: list) -> list:
    """
    Secondary GPT call:
    - Provide GPT the final answer + a list of chunk references.
    - GPT identifies which chunks were actually used in formulating the answer.
    Returns a list of chunk indices (1-based).
    """
    # Build a readable list of chunks for GPT to examine
    chunk_list_str = ""
    for i, match in enumerate(matches, start=1):
        metadata = match.get("metadata", {})
        sec = metadata.get("section", "No Section Provided")
        chunk_preview = metadata.get("chunk_text", "")[:200]  # just a snippet
        chunk_list_str += f"{i}. [Section: {sec}]\nSnippet: {chunk_preview}...\n\n"
    
    # Ask GPT which chunks contributed to the final answer
    prompt = f"""
Here is the final answer:
\"\"\"{final_answer}\"\"\"

Below are the retrieved chunks that were available as context:

{chunk_list_str}

Which of these chunks contributed directly to the final answer? 
Please respond with the *numbers or the 'section' text* that were essential.
Only list the ones truly relevant.
"""
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant identifying references used."},
        {"role": "user", "content": prompt}
    ]

    start_time = time.time()
    response = openai.ChatCompletion.create(
        model="gpt-4o",  # Using your specialized GPT model
        messages=messages,
        temperature=0,
        timeout=30
    )
    end_time = time.time()
    logging.info(f"Reference-identification GPT call took {end_time - start_time:.2f} seconds.")

    used_str = response.choices[0].message.content.strip()
    
    # Parse the GPT response to see which chunks it mentions
    used_indices = []
    for i, match in enumerate(matches, start=1):
        sec = match["metadata"].get("section", "No Section Provided")
        # If GPT mentions the numeric index or the 'section' text, consider it used
        if str(i) in used_str or sec in used_str:
            used_indices.append(i)
    return used_indices

# -----------------------------
# Main Query Function
# -----------------------------
def main():
    # 1) Initialize Pinecone
    index = init_pinecone(pinecone_api_key, pinecone_env, index_name)

    # 2) Ask user for a question
    user_query = input("Enter your legislation query: ").strip()
    if not user_query:
        print("No query provided.")
        return
    
    # 3) Embed user query
    query_vector = get_embedding(user_query)

    # 4) Retrieve top-k matches
    matches = query_pinecone(index, query_vector, top_k=5)
    if not matches:
        print("No matching records found in Pinecone.")
        return

    # 5) Build a context string from the chunk_text
    context = ""
    for i, match in enumerate(matches, start=1):
        meta = match.get("metadata", {})
        sec = meta.get("section", "No Section Provided")
        chunk_text = meta.get("chunk_text", "")
        context += f"--- Chunk {i} (Section: {sec}) ---\n{chunk_text}\n\n"

    # 6) Primary GPT call => final answer
    final_answer = call_llm_with_context(user_query, context)

    # 7) Secondary GPT call => identify used chunks
    used_indices = identify_used_chunks_gpt(final_answer, matches)

    # 8) Print final answer
    print("\n=== FINAL ANSWER ===")
    print(final_answer)

    # 9) Print used chunks: only "section" + snippet of "chunk_text"
    print("\n=== RELEVANT CHUNKS ===")
    if not used_indices:
        print("GPT did not list any chunk references.")
    else:
        for i in used_indices:
            match = matches[i - 1]  # i is 1-based
            meta = match.get("metadata", {})
            section_text = meta.get("section", "No Section Provided")
            snippet = meta.get("chunk_text", "")[:200]  # first 200 chars
            print("-----")
            print(f"Section: {section_text}")
            print(f"Snippet: {snippet}...")
            print("-----")

if __name__ == "__main__":
    main()

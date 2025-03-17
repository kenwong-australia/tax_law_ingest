import openai
import pinecone
import logging
import time
import os
import json
from dotenv import load_dotenv

load_dotenv()


# --- CONFIGURATION ---
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
index_name = os.getenv("PINECONE_INDEX")
VECTOR_DIMENSION = 3072  # For the "text-embedding-3-large" model

# Path to the tax keyword library
TAX_KEYWORD_PATH = os.getenv("TAX_KEYWORD_PATH", "/Users/kenmacpro/pinecone-upsert/tax_keywords.json")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------------------
# Load Keywords and Categories Library
# -----------------------------
def load_keyword_library():
    """
    Load keywords and categories from the tax keyword JSON file.
    The JSON structure has categories as keys and arrays of keywords as values.
    
    Returns:
        tuple: (all_keywords, all_categories, category_to_keywords) - sets of unique keywords 
               and categories, plus a dictionary mapping categories to their keywords
    """
    all_keywords = set()
    all_categories = set()
    category_to_keywords = {}
    
    try:
        # Load the specific tax_keyword.json file
        if os.path.exists(TAX_KEYWORD_PATH):
            with open(TAX_KEYWORD_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # In this structure, keys are categories and values are arrays of keywords
                for category, keywords in data.items():
                    if isinstance(keywords, list):
                        # Add the category
                        all_categories.add(category)
                        
                        # Add all keywords from this category
                        all_keywords.update(keywords)
                        
                        # Store the mapping of category to its keywords
                        category_to_keywords[category] = keywords
                
            logging.info(f"Loaded {len(all_categories)} categories with a total of {len(all_keywords)} unique keywords from {TAX_KEYWORD_PATH}")
        else:
            logging.warning(f"Tax keyword file not found at {TAX_KEYWORD_PATH}")
    except Exception as e:
        logging.error(f"Error loading tax keyword file: {e}")
    
    return all_keywords, all_categories, category_to_keywords

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
    # New Pinecone initialization for v2.x
    from pinecone import Pinecone
    
    pc = Pinecone(api_key=api_key)
    return pc.Index(index_name)

def query_pinecone(index, query_vector, top_k=5, filter_dict=None):
    """
    Query the Pinecone index with the provided vector and return
    the top_k matches with metadata.
    
    Args:
        index: The Pinecone index to query
        query_vector: The embedded vector representation of the query
        top_k: Number of results to return
        filter_dict: Optional dictionary for metadata filtering (e.g., {"categories": "Income tax"})
    """
    response = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict
    )
    return response.get("matches", [])

def extract_relevant_terms(query: str, all_keywords: set, all_categories: set, category_to_keywords: dict) -> dict:
    """
    Extract relevant keywords and categories from the query using the predefined library.
    Uses a single GPT call to identify both categories and their respective keywords.
    
    Args:
        query: The user's query
        all_keywords: Set of all available keywords from the library
        all_categories: Set of all available categories from the library
        category_to_keywords: Dictionary mapping categories to their keywords
        
    Returns:
        dict: Contains 'keywords' and 'categories' lists with matching terms
    """
    # Format the available categories to send to GPT
    categories_str = ", ".join(sorted(all_categories))
    
    # Create a formatted list of categories with their keywords
    categories_with_keywords = []
    for category in sorted(all_categories):
        keywords = category_to_keywords.get(category, [])
        if keywords:
            keyword_str = ", ".join(keywords)
            categories_with_keywords.append(f"{category}: [{keyword_str}]")
    
    # Join all category-keyword mappings with line breaks for readability
    mapping_str = "\n".join(categories_with_keywords)
    
    prompt = f"""
    I need to match a tax legislation query against our controlled vocabulary of tax categories and keywords.
    
    QUERY: "{query}"
    
    Below is our taxonomy with categories and their respective keywords:
    
    {mapping_str}
    
    Please analyze the query and:
    1. Identify up to 2 of the most relevant categories from our taxonomy
    2. For each identified category, select up to 3 of the most relevant keywords from that category's list
    
    Format your response as a JSON object with this structure:
    {{
        "categories": ["Category1", "Category2"],
        "selected_keywords": {{
            "Category1": ["Keyword1", "Keyword2", "Keyword3"],
            "Category2": ["Keyword1", "Keyword2"]
        }}
    }}
    
    ONLY include exact matches from the provided categories and keywords.
    """
    
    messages = [
        {"role": "system", "content": "You are a legal taxonomy specialist. You help match queries to a controlled vocabulary."},
        {"role": "user", "content": prompt}
    ]
    
    # Single GPT call to get all information at once
    start_time = time.time()
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.1,
        response_format={"type": "json_object"},
        timeout=25  # Slightly longer timeout for the more complex task
    )
    end_time = time.time()
    logging.info(f"Taxonomy matching took {end_time - start_time:.2f} seconds")
    
    try:
        content = response.choices[0].message.content.strip()
        result = json.loads(content)
        
        # Validate that the returned categories are actually in our library
        valid_categories = [cat for cat in result.get("categories", []) if cat in all_categories]
        
        # Collect all keywords from the selected categories
        all_suggested_keywords = []
        selected_keywords = result.get("selected_keywords", {})
        
        for category, keywords in selected_keywords.items():
            if category in valid_categories:
                # Get available keywords for this category
                available_keywords = category_to_keywords.get(category, [])
                
                # Only accept keywords that are actually in this category
                valid_category_keywords = [kw for kw in keywords if kw in available_keywords]
                all_suggested_keywords.extend(valid_category_keywords)
        
        return {
            "keywords": all_suggested_keywords,
            "categories": valid_categories
        }
    except Exception as e:
        logging.warning(f"Error parsing terms from GPT response: {e}")
        return {"keywords": [], "categories": []}

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
        "Please provide a concise, direct response referencing the context where relevant. "
        "Cite specific sections of legislation when applicable."
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
        keywords = metadata.get("keywords", [])
        categories = metadata.get("categories", [])
        
        # Add keywords and categories to the chunk information
        keywords_str = ", ".join(keywords) if keywords else "None"
        categories_str = ", ".join(categories) if categories else "None"
        
        chunk_preview = metadata.get("chunk_text", "")[:150]  # just a snippet
        chunk_list_str += f"{i}. [Section: {sec}]\nKeywords: {keywords_str}\nCategories: {categories_str}\nSnippet: {chunk_preview}...\n\n"
    
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
    
    # 2) Load the keyword and category library
    all_keywords, all_categories, category_to_keywords = load_keyword_library()
    
    if not all_keywords and not all_categories:
        logging.warning("No keywords or categories found in the library. Proceeding without controlled vocabulary matching.")

    # 3) Ask user for a question
    user_query = input("Enter your legislation query: ").strip()
    if not user_query:
        print("No query provided.")
        return
    
    # 4) Ask if user wants to filter by specific category or keyword
    use_filters = input("Would you like to apply metadata filters? (yes/no): ").strip().lower()
    filter_dict = None
    
    if use_filters in ['yes', 'y']:
        filter_mode = input("Filter by: (1) category, (2) keyword, (3) auto-extract from controlled vocabulary: ").strip()
        
        if filter_mode == '1':
            # Show available categories if we have them
            if all_categories:
                print(f"Available categories: {', '.join(sorted(all_categories))}")
            category = input("Enter category to filter by: ").strip()
            if category:
                filter_dict = {"categories": category}
        
        elif filter_mode == '2':
            # First select a category
            if all_categories:
                print(f"Available categories: {', '.join(sorted(all_categories))}")
                selected_category = input("First select a category: ").strip()
                
                if selected_category in all_categories:
                    # Show keywords for the selected category
                    category_keywords = category_to_keywords.get(selected_category, [])
                    if category_keywords:
                        print(f"Keywords for {selected_category}:")
                        print(", ".join(category_keywords))
                        
                        keyword = input("Enter keyword to filter by: ").strip()
                        if keyword and keyword in category_keywords:
                            filter_dict = {"keywords": keyword}
                        else:
                            print("Invalid keyword. Proceeding without keyword filter.")
                else:
                    print("Invalid category. Proceeding without category filter.")
        
        elif filter_mode == '3' and (all_keywords or all_categories):
            # Auto-extract using controlled vocabulary
            print("Extracting relevant terms from your query using our controlled vocabulary...")
            terms = extract_relevant_terms(user_query, all_keywords, all_categories, category_to_keywords)
            
            if terms["keywords"]:
                print(f"Suggested keywords: {', '.join(terms['keywords'])}")
            
            if terms["categories"]:
                print(f"Suggested categories: {', '.join(terms['categories'])}")
            
            if terms["keywords"] or terms["categories"]:
                filter_type = input("Filter by (1) keywords, (2) categories, (3) both: ").strip()
                
                if filter_type == '1' and terms["keywords"]:
                    filter_dict = {"keywords": {"$in": terms["keywords"]}}
                elif filter_type == '2' and terms["categories"]:
                    filter_dict = {"categories": {"$in": terms["categories"]}}
                elif filter_type == '3' and (terms["keywords"] or terms["categories"]):
                    # We need to build a composite filter with $and operator
                    filter_dict = {"$and": []}
                    
                    if terms["keywords"]:
                        filter_dict["$and"].append({"keywords": {"$in": terms["keywords"]}})
                    
                    if terms["categories"]:
                        filter_dict["$and"].append({"categories": {"$in": terms["categories"]}})
                        
                    # If we only have one condition, simplify the filter
                    if len(filter_dict["$and"]) == 1:
                        filter_dict = filter_dict["$and"][0]
            else:
                print("No matching terms found in the controlled vocabulary. Proceeding without filters.")
    
    # 5) Embed user query
    query_vector = get_embedding(user_query)

    # 6) Retrieve top-k matches
    matches = query_pinecone(index, query_vector, top_k=15, filter_dict=filter_dict)
    if not matches:
        print("No matching records found in Pinecone.")
        return
    
    # Display all retrieved sections
    print("\n=== RETRIEVED CHUNKS ===")
    print("Listing all sections retrieved from Pinecone:")
    for i, match in enumerate(matches, start=1):
        sec = match.get("metadata", {}).get("section", "No Section Provided")
        print(f"{i}. Section: {sec}")
    print("========================\n")

    # 7) Build a context string from the chunk_text, now including keywords and categories
    context = ""
    for i, match in enumerate(matches, start=1):
        meta = match.get("metadata", {})
        sec = meta.get("section", "No Section Provided")
        keywords = meta.get("keywords", [])
        categories = meta.get("categories", [])
        
        # Format keywords and categories for inclusion
        keywords_str = ", ".join(keywords) if keywords else "None"
        categories_str = ", ".join(categories) if categories else "None"
        
        chunk_text = meta.get("chunk_text", "")
        context += f"--- Chunk {i} ---\n"
        context += f"Section: {sec}\n"
        context += f"Keywords: {keywords_str}\n"
        context += f"Categories: {categories_str}\n"
        context += f"Content: {chunk_text}\n\n"

    # 8) Primary GPT call => final answer
    final_answer = call_llm_with_context(user_query, context)

    # 9) Secondary GPT call => identify used chunks
    used_indices = identify_used_chunks_gpt(final_answer, matches)

    # 10) Print final answer
    print("\n=== FINAL ANSWER ===")
    print(final_answer)

    # 11) Print used chunks with enhanced metadata
    print("\n=== RELEVANT CHUNKS ===")
    if not used_indices:
        print("GPT did not identify any specific chunk references.")
    else:
        for i in used_indices:
            match = matches[i - 1]  # i is 1-based
            meta = match.get("metadata", {})
            section_text = meta.get("section", "No Section Provided")
            keywords = meta.get("keywords", [])
            categories = meta.get("categories", [])
            
            keywords_str = ", ".join(keywords) if keywords else "None"
            categories_str = ", ".join(categories) if categories else "None"
            
            snippet = meta.get("chunk_text", "")[:200]  # first 200 chars
            print("-----")
            print(f"Section: {section_text}")
            print(f"Keywords: {keywords_str}")
            print(f"Categories: {categories_str}")
            print(f"Snippet: {snippet}...")
            print("-----")

if __name__ == "__main__":
    main()

import openai
import pinecone
import logging
import time
import os
import json
import re
from collections import Counter
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

# Hybrid search configuration
HYBRID_VECTOR_WEIGHT = 0.7  # Weight for vector scores (0.0-1.0)
HYBRID_KEYWORD_WEIGHT = 0.3  # Weight for keyword scores (0.0-1.0)
HYBRID_CANDIDATES = 50  # Number of initial candidates for hybrid search

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
# Text Processing for Hybrid Search
# -----------------------------
def extract_keywords(text, min_length=3, max_keywords=12):
    """
    Extract potential keywords from text for keyword-based matching.
    Uses a tax domain-aware approach with improved financial term handling.
    
    Args:
        text: Input text to extract keywords from
        min_length: Minimum length of words to consider
        max_keywords: Maximum number of keywords to return
    
    Returns:
        list: Top keywords extracted from the text
    """
    # Preserve section numbers (pattern: digits-digits)
    section_numbers = []
    section_pattern = re.compile(r'\b\d+[-‑]\d+\b')  # Match patterns like "102-5"
    for match in section_pattern.finditer(text):
        section_numbers.append(match.group(0))
    
    # Preserve dollar amounts for proper handling
    dollar_amounts = []
    dollar_pattern = re.compile(r'\$\s*[\d,]+(?:\.\d+)?')  # Match patterns like "$1,000" or "$200,000"
    for match in dollar_pattern.finditer(text):
        amount = match.group(0).replace('$', '').replace(',', '')
        dollar_amounts.append('dollars_' + amount)  # Add prefix to identify as money
    
    # Preserve dates in various formats
    dates = []
    date_patterns = [
        re.compile(r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b', re.IGNORECASE),
        re.compile(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', re.IGNORECASE)
    ]
    for pattern in date_patterns:
        for match in pattern.finditer(text):
            dates.append(match.group(0).lower().replace(' ', '_'))
    
    # Clean text and apply standard tokenization
    processed_text = text.lower()
    # Replace special chars with spaces but keep hyphens for section numbers
    processed_text = re.sub(r'[^\w\s-]', ' ', processed_text)
    # Extra space around hyphens that aren't in section numbers
    processed_text = re.sub(r'(?<!\d)-(?!\d)', ' ', processed_text)
    # Ensure spaces around numbers
    processed_text = re.sub(r'(\d+)', r' \1 ', processed_text)
    
    # Split into words and remove extra spaces
    words = [w.strip() for w in processed_text.split()]
    words = [w for w in words if w]
    
    # Add preserved tokens back
    words.extend(section_numbers)
    words.extend(dollar_amounts)
    words.extend(dates)
    
    # Filter short words but keep section numbers
    words = [word for word in words if (len(word) >= min_length or re.match(r'\d+-\d+', word))]
    
    # Remove common stop words (tax-domain specific)
    stop_words = {
        'and', 'the', 'is', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
        'by', 'as', 'that', 'this', 'are', 'was', 'were', 'been', 'being',
        'have', 'has', 'had', 'not', 'but', 'what', 'which', 'who', 'whom',
        'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
        'more', 'most', 'some', 'such', 'nor', 'from', 'you', 'your', 'yours',
        'can', 'will', 'would', 'should', 'could', 'may', 'might',
        # Filter purely numeric tokens except for section numbers
        # Section numbers have format like '102-5'
    }
    
    filtered_words = []
    for word in words:
        # Keep section numbers
        if re.match(r'\d+-\d+', word):
            filtered_words.append(word)
        # Keep dollar amounts
        elif word.startswith('dollars_'):
            filtered_words.append(word)
        # Filter numeric-only terms
        elif word.isdigit():
            # Keep years (4-digit numbers between 1900-2100)
            if len(word) == 4 and 1900 <= int(word) <= 2100:
                filtered_words.append(word)
        # Apply regular stop word filtering
        elif word not in stop_words:
            filtered_words.append(word)
    
    # Add domain specific contextual terms
    tax_terms = ["capital", "gains", "income", "tax", "loss", "asset", "exemption", 
                 "discount", "deduction", "section", "calculation", 
                 "taxable", "financial", "year", "individual", "corporate", "trust"]
    
    # Count word frequencies
    word_counts = Counter(filtered_words)
    
    # Boost certain tax-related terms
    for term in tax_terms:
        if term in word_counts:
            word_counts[term] *= 1.5  # Boost tax-related terms
    
    # Get top keywords by frequency
    top_keywords = [word for word, _ in word_counts.most_common(max_keywords)]
    
    # Clean up dollar amount prefixes for display
    top_keywords = [w.replace('dollars_', '$') if w.startswith('dollars_') else w for w in top_keywords]
    
    # Remove any remaining purely numeric terms that aren't years or section numbers
    top_keywords = [w for w in top_keywords if not (w.isdigit() and not (len(w) == 4 and 1900 <= int(w) <= 2100))]
    
    return top_keywords

def calculate_keyword_score(query_keywords, text):
    """
    Calculate a score based on keyword matching between query keywords and text.
    
    Args:
        query_keywords: List of keywords extracted from the query
        text: Text to match against
        
    Returns:
        float: Score between 0-1 indicating keyword match quality
    """
    if not query_keywords or not text:
        return 0.0
    
    text_lower = text.lower()
    
    # Count matches
    matches = sum(1 for keyword in query_keywords if keyword.lower() in text_lower)
    
    # Calculate score based on proportion of matches and their positions
    base_score = matches / len(query_keywords) if len(query_keywords) > 0 else 0
    
    # Bonus for keyword density (matches relative to text length)
    density_bonus = min(0.2, matches / (len(text) / 100)) if len(text) > 0 else 0
    
    return min(1.0, base_score + density_bonus)

# -----------------------------
# Hybrid Search Implementation
# -----------------------------
def hybrid_search(index, query, query_vector, top_k=15, filter_dict=None, 
                 vector_weight=HYBRID_VECTOR_WEIGHT, 
                 keyword_weight=HYBRID_KEYWORD_WEIGHT,
                 candidates=HYBRID_CANDIDATES):
    """
    Perform a hybrid search combining vector similarity with keyword matching.
    
    Args:
        index: Pinecone index
        query: Original text query
        query_vector: Vector embedding of the query
        top_k: Number of results to return
        filter_dict: Optional metadata filters
        vector_weight: Weight to apply to vector scores (0-1)
        keyword_weight: Weight to apply to keyword scores (0-1)
        candidates: Number of initial candidates to retrieve
        
    Returns:
        list: Reranked matches
    """
    # 1. Extract keywords from the query
    query_keywords = extract_keywords(query)
    logging.info(f"Extracted keywords for hybrid search: {', '.join(query_keywords)}")
    
    try:
        # 2. Get initial candidates with vector search (more than we need)
        response = index.query(
            vector=query_vector,
            top_k=candidates,  # Get more candidates than we need
            include_metadata=True,
            filter=filter_dict
        )
        
        initial_matches = response.get("matches", [])
        
        if not initial_matches:
            logging.warning("No initial matches found in Pinecone.")
            return []
        
        # 3. Calculate hybrid scores
        hybrid_results = []
        
        for match in initial_matches:
            # Create a fresh result dictionary - will work with any match object type
            result = {}
            
            # Get the vector similarity score - will work with both dict and ScoredVector
            vector_score = 0
            try:
                if hasattr(match, "score"):
                    vector_score = float(match.score)
                elif isinstance(match, dict) and "score" in match:
                    vector_score = float(match["score"])
            except (ValueError, TypeError):
                pass
                
            # Get id
            match_id = None
            try:
                if hasattr(match, "id"):
                    match_id = match.id
                elif isinstance(match, dict) and "id" in match:
                    match_id = match["id"]
            except Exception:
                pass
                
            # Get metadata safely
            metadata = {}
            try:
                if hasattr(match, "metadata") and match.metadata is not None:
                    metadata = match.metadata
                elif isinstance(match, dict) and "metadata" in match:
                    metadata = match["metadata"]
            except Exception:
                pass
            
            # Copy over standard Pinecone fields
            if match_id:
                result["id"] = match_id
            
            # Use a copy of metadata to avoid reference issues
            if metadata:
                result["metadata"] = dict(metadata)
                
            # Get text content and metadata fields safely
            text_content = ""
            section = ""
            categories = []
            keywords = []
            
            if metadata:
                # Extract fields safely
                if isinstance(metadata, dict):
                    text_content = metadata.get("chunk_text", "")
                    section = metadata.get("section", "")
                    
                    # Handle potential None values for lists
                    categories = metadata.get("categories", [])
                    if categories is None:
                        categories = []
                    elif isinstance(categories, str):
                        categories = [categories]
                        
                    keywords = metadata.get("keywords", [])
                    if keywords is None:
                        keywords = []
                    elif isinstance(keywords, str):
                        keywords = [keywords]
                else:
                    # Try attribute access for non-dict metadata
                    try:
                        if hasattr(metadata, "chunk_text"):
                            text_content = metadata.chunk_text or ""
                        if hasattr(metadata, "section"):
                            section = metadata.section or ""
                        if hasattr(metadata, "categories"):
                            categories = metadata.categories or []
                        if hasattr(metadata, "keywords"):
                            keywords = metadata.keywords or []
                    except Exception:
                        pass
            
            # Convert any non-list categories/keywords to lists
            if not isinstance(categories, list):
                categories = [categories] if categories else []
            if not isinstance(keywords, list):
                keywords = [keywords] if keywords else []
            
            # Create combined text for keyword matching
            combined_text = f"{text_content} {section}"
            if categories:
                combined_text += f" {' '.join(str(c) for c in categories)}"
            if keywords:
                combined_text += f" {' '.join(str(k) for k in keywords)}"
            
            # Calculate keyword matching score
            keyword_score = calculate_keyword_score(query_keywords, combined_text)
            
            # Combine scores with weights
            hybrid_score = (vector_score * vector_weight) + (keyword_score * keyword_weight)
            
            # Add score fields
            result["original_score"] = vector_score
            result["keyword_score"] = keyword_score
            result["score"] = hybrid_score
            
            hybrid_results.append(result)
        
        # Log success message
        logging.info(f"Processed {len(hybrid_results)} matches for hybrid search")
        
        # 4. Sort by hybrid score and take top_k
        hybrid_results.sort(key=lambda x: x["score"], reverse=True)
        return hybrid_results[:top_k]
        
    except Exception as e:
        logging.error(f"Error in hybrid search: {e}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        # Fall back to regular vector search
        logging.info("Falling back to standard vector search")
        return query_pinecone(index, query_vector, top_k=top_k, filter_dict=filter_dict)

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

def query_pinecone(index, query_vector, top_k=15, filter_dict=None):
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
    
    # 4) Ask if user wants to use filters and select search method
    search_mode = input("Select search method: (1) vector search, (2) hybrid search: ").strip()
    use_hybrid = search_mode == '2'
    
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

    # 6) Retrieve matches based on search mode
    if use_hybrid:
        print(f"\nUsing hybrid search (vector weight: {HYBRID_VECTOR_WEIGHT}, keyword weight: {HYBRID_KEYWORD_WEIGHT})")
        matches = hybrid_search(
            index=index, 
            query=user_query,
            query_vector=query_vector, 
            top_k=15, 
            filter_dict=filter_dict
        )
    else:
        matches = query_pinecone(index, query_vector, top_k=15, filter_dict=filter_dict)
    
    if not matches:
        print("No matching records found in Pinecone.")
        return
    
    # Display all retrieved sections
    print("\n=== RETRIEVED CHUNKS ===")
    print("Listing all sections retrieved from Pinecone:")
    for i, match in enumerate(matches, start=1):
        sec = match.get("metadata", {}).get("section", "No Section Provided")
        score_info = ""
        if use_hybrid:
            vector_score = match.get("original_score", 0)
            keyword_score = match.get("keyword_score", 0)
            hybrid_score = match.get("score", 0)
            score_info = f" [V: {vector_score:.3f}, K: {keyword_score:.3f}, H: {hybrid_score:.3f}]"
        print(f"{i}. Section: {sec}{score_info}")
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
        
        # Add score information for hybrid search
        score_info = ""
        if use_hybrid:
            vector_score = match.get("original_score", 0)
            keyword_score = match.get("keyword_score", 0)
            hybrid_score = match.get("score", 0)
            score_info = f"Scores [Vector: {vector_score:.3f}, Keyword: {keyword_score:.3f}, Hybrid: {hybrid_score:.3f}]\n"
        
        chunk_text = meta.get("chunk_text", "")
        context += f"--- Chunk {i} ---\n"
        context += f"Section: {sec}\n"
        context += f"Keywords: {keywords_str}\n"
        context += f"Categories: {categories_str}\n"
        if score_info:
            context += score_info
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
            
            # Add score information for hybrid search
            score_info = ""
            if use_hybrid:
                vector_score = match.get("original_score", 0)
                keyword_score = match.get("keyword_score", 0)
                hybrid_score = match.get("score", 0)
                score_info = f"Scores [Vector: {vector_score:.3f}, Keyword: {keyword_score:.3f}, Hybrid: {hybrid_score:.3f}]"
            
            snippet = meta.get("chunk_text", "")[:200]  # first 200 chars
            print("-----")
            print(f"Section: {section_text}")
            print(f"Keywords: {keywords_str}")
            print(f"Categories: {categories_str}")
            if score_info:
                print(score_info)
            print(f"Snippet: {snippet}...")
            print("-----")
    
    # 12) Follow-up query for additional sections
    additional_query = input("\nWould you like to add specific sections to improve the answer? (yes/no): ").strip().lower()
    
    if additional_query in ['yes', 'y']:
        # Get the section numbers or keywords from the user
        section_input = input("Enter section numbers or keywords (comma separated, e.g., '102-5, capital gains'): ").strip()
        
        if section_input:
            additional_matches = []
            sections_added = set()
            
            # Parse the input into a list of section numbers and search terms
            search_terms = [term.strip() for term in section_input.split(',')]
            
            print("\nRetrieving additional sections...")
            
            # First, try to retrieve by exact section number
            section_terms = [term for term in search_terms if '-' in term or term.isdigit()]
            if section_terms:
                for term in section_terms:
                    # Build a filter for exact section number match
                    section_filter = {"section": term}
                    
                    # Query Pinecone with this filter
                    section_matches = query_pinecone(index, query_vector, top_k=5, filter_dict=section_filter)
                    
                    for match in section_matches:
                        section = match.get("metadata", {}).get("section", "")
                        if section and section not in sections_added:
                            additional_matches.append(match)
                            sections_added.add(section)
                            print(f"Added section: {section}")
            
            # Then try keyword search for any remaining terms
            keyword_terms = [term for term in search_terms if term not in section_terms and len(term) >= 3]
            if keyword_terms:
                # Use our keywords to filter by metadata
                for term in keyword_terms:
                    # Try searching in keywords field
                    keyword_filter = {"keywords": {"$in": [term]}}
                    keyword_matches = query_pinecone(index, query_vector, top_k=3, filter_dict=keyword_filter)
                    
                    for match in keyword_matches:
                        section = match.get("metadata", {}).get("section", "")
                        if section and section not in sections_added:
                            additional_matches.append(match)
                            sections_added.add(section)
                            print(f"Added section: {section} (matched keyword: {term})")
            
            # If we found additional matches, regenerate the answer
            if additional_matches:
                print(f"\nAdded {len(additional_matches)} new sections. Regenerating answer...")
                
                # Combine the original used chunks with the new ones
                combined_matches = []
                
                # First add the chunks that were used in the original answer
                for i in used_indices:
                    combined_matches.append(matches[i - 1])
                
                # Then add the new chunks
                combined_matches.extend(additional_matches)
                
                # Build a new context string with all chunks
                new_context = ""
                for i, match in enumerate(combined_matches, start=1):
                    meta = match.get("metadata", {})
                    sec = meta.get("section", "No Section Provided")
                    keywords = meta.get("keywords", [])
                    categories = meta.get("categories", [])
                    
                    # Format keywords and categories for inclusion
                    keywords_str = ", ".join(keywords) if keywords else "None"
                    categories_str = ", ".join(categories) if categories else "None"
                    
                    chunk_text = meta.get("chunk_text", "")
                    new_context += f"--- Chunk {i} ---\n"
                    new_context += f"Section: {sec}\n"
                    new_context += f"Keywords: {keywords_str}\n"
                    new_context += f"Categories: {categories_str}\n"
                    new_context += f"Content: {chunk_text}\n\n"
                
                # Generate a new answer with the expanded context
                improved_answer = call_llm_with_context(user_query, new_context)
                
                # Print the improved answer
                print("\n=== IMPROVED ANSWER WITH ADDITIONAL SECTIONS ===")
                print(improved_answer)
                
                # Identify which chunks were used in the improved answer
                improved_used_indices = identify_used_chunks_gpt(improved_answer, combined_matches)
                
                # Print used chunks for the improved answer
                print("\n=== RELEVANT CHUNKS FOR IMPROVED ANSWER ===")
                if not improved_used_indices:
                    print("GPT did not identify any specific chunk references.")
                else:
                    for i in improved_used_indices:
                        match = combined_matches[i - 1]  # i is 1-based
                        meta = match.get("metadata", {})
                        section_text = meta.get("section", "No Section Provided")
                        snippet = meta.get("chunk_text", "")[:200]  # first 200 chars
                        print("-----")
                        print(f"Section: {section_text}")
                        print(f"Snippet: {snippet}...")
                        print("-----")
            else:
                print("No additional sections found matching your criteria.")
        else:
            print("No input provided. Keeping the original answer.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ato_extraction.py

A refactored module for extracting law data from the ATO website.
This script:
  - Sends requests to the ATO API to download a hierarchical structure of legal documents.
  - Recursively extracts and saves data from folders.
  - Builds a file index from the extracted hierarchy.
  - Contains utilities to extract page content and convert HTML to Markdown.
  
Make sure to update the PROJECT_PATH variable below.
"""

import os
import json
import requests
from random import choice
from time import sleep
from tqdm import tqdm
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
from markdownify import markdownify as md
import datetime

# --- Global Configuration ---
# Update this path to point to your local project directory.
PROJECT_PATH = "/Users/kenmacpro/pinecone-upsert/testfiles_ato"  
BASE_URL = "https://www.ato.gov.au/API/v1/law/lawservices/browse-content/"
DELAY_CHOICES = [14, 17, 18, 20, 19, 21, 22, 23]

# Create a global fake user agent instance.
ua = UserAgent()

# --- Utility Functions ---

def read_json(data_path):
    with open(data_path, 'r') as file:
        return json.load(file)

def write_json(data_path, data):
    with open(data_path, 'w') as f:
        json.dump(data, f)

def read_txt(txt_path):
    with open(txt_path, "r") as f:
        return f.read()

def write_json_lines(file_name, dict_data):
    json_string = json.dumps(dict_data)
    with open(file_name, 'a') as f:
        f.write(json_string + "\n")

def read_json_lines(file_name):
    lines = []
    with open(file_name) as file_in:
        for line in file_in:
            try:
                lines.append(json.loads(line))
            except Exception:
                continue
    return lines

def save_dict_list(file_name, dicts_data):
    for dict_data in dicts_data:
        write_json_lines(file_name, dict_data)

# --- Extraction Functions ---

def send_request(main_url, sub_url):
    headers = {
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Host": "www.ato.gov.au",
        "Referer": "https://www.ato.gov.au/single-page-applications/legaldatabase",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "TE": "trailers",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0"
    }
    url_ = main_url + "?" + sub_url
    headers['fake_ua'] = ua.random
    response = requests.get(url_, headers=headers)
    return response.json()

def rec_extract(parent_link, save_dir, level, parent):
    if 'folder' in parent_link and parent_link['folder']:
        title = parent_link['title'].replace('/', '|')
        url = parent_link['data']['url']
        save_dir_all = os.path.join(save_dir, f'level_{level}', parent)
        os.makedirs(save_dir_all, exist_ok=True)
        save_path = os.path.join(save_dir_all, title + '.jsonl')
        if os.path.exists(save_path):
            out_data = read_json_lines(save_path)
        else:
            out_data = send_request(BASE_URL, url)
            save_dict_list(save_path, out_data)
            sleep(choice(DELAY_CHOICES))
        for link in out_data:
            rec_extract(link, save_dir, level + 1, parent_link['title'].replace('/', '|'))

def recursive_extraction_for_topic(save_dir, save_file, topic):
    os.makedirs(save_dir, exist_ok=True)
    save_dir_all = os.path.join(save_dir, 'level_1')
    os.makedirs(save_dir_all, exist_ok=True)
    save_path = os.path.join(save_dir_all, save_file)
    if os.path.exists(save_path):
        out_data = read_json_lines(save_path)
    else:
        out_data = send_request(BASE_URL, "Mode=topic")
        save_dict_list(save_path, out_data)
    
    # Filter for the specific topic
    topic_data = [link for link in out_data if topic.lower() in link.get('title', '').lower()]
    
    level = 2
    for link in tqdm(topic_data, desc=f"Extracting {topic} hierarchy"):
        rec_extract(link, save_dir, level, 'main')
    return True

def extract_file_index(parent_links, save_dir, level, parent, save_path):
    if isinstance(parent_links, list):
        parent_link = parent_links[-1]
        if 'folder' in parent_link and parent_link['folder']:
            title = parent_link['title'].replace('/', '|')
            save_dir_all = os.path.join(save_dir, f'level_{level}', parent)
            hierarchy_path = os.path.join(save_dir_all, title + '.jsonl')
            out_data = read_json_lines(hierarchy_path)
            for link in out_data:
                new_links = parent_links + [link]
                extract_file_index(new_links, save_dir, level + 1, parent_link['title'].replace('/', '|'), save_path)
        elif 'a_attr' in parent_link:
            data_ = {
                'path_info': parent_links[:-1],
                'file_info': parent_links[-1]
            }
            write_json_lines(save_path, data_)

def recursive_file_index(save_dir, save_file, save_path):
    save_dir_all = os.path.join(save_dir, 'level_1')
    hierarchy_path = os.path.join(save_dir_all, save_file)
    out_data = read_json_lines(hierarchy_path)
    level = 2
    for link in tqdm(out_data, desc="Building file index"):
        extract_file_index([link], save_dir, level, 'main', save_path)
    return True

# --- Page Extraction Functions ---

def extract_page(url):
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "utf-8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Host": "www.ato.gov.au",
        "Priority": "u=0, i",
        "Referer": "https://colab.research.google.com/",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "cross-site",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": '1',
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    html_content = soup.find("div", {"id": "lawContents"})
    if html_content is None:
        html_content = soup.find("div", {"id": "LawContent"})
    if html_content is None:
        html_content = soup.find("div", {"id": "lawContent"})
    if html_content is None:
        return None
    return md(str(html_content))

def extract_pages(pages_index, save_dir):
    """
    Converts each page in 'pages_index' to a single .json file that your upsert code can handle.
    
    The resulting JSON shape is:
    {
      "text": "...",
      "section": null,
      "title": "...",
      "metadata": {
        "source_file": "...",
        "full_reference": "...",
        "url": "..."
      }
    }
    """
    os.makedirs(save_dir, exist_ok=True)
    main_page_url = "https://www.ato.gov.au"
    
    for page_info in tqdm(pages_index, desc="Extracting pages"):
        file_info = page_info.get('file_info', {})
        file_title = file_info.get('title', "No Title Provided")
        file_href  = file_info.get('a_attr', {}).get('href', '')
        
        if not file_href:
            continue
        
        # Build a fully-qualified URL
        file_url = main_page_url + file_href
        
        # Use the docid or sanitized title to form a local filename
        file_name = file_url.split('docid=')[-1].replace('/', '|') + '.json'
        file_path = os.path.join(save_dir, file_name)
        
        # Convert page HTML to markdown
        md_page = extract_page(url=file_url)
        if md_page is None:
            continue
        
        # If we haven't saved this page yet, create a final doc in upsert-friendly shape
        if not os.path.exists(file_path):
            # Get current date/time for metadata
            current_date = datetime.datetime.now().isoformat()
            
            # Build the final doc
            doc = {
                "text": md_page,           # full page content in markdown
                "section": None,          # ATO docs don't use 'section'
                "title": file_title,      
                "metadata": {
                    "source_file": "ATO_extraction",   # or some dynamic ref
                    "full_reference": "",              # optional or set as needed
                    "url": file_url,                   # store ATO doc's URL
                    "creation_date": current_date      # add creation date
                }
            }
            
            # Write out a single .json file for this doc
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)
            
            # Random sleep to avoid hitting server too quickly
            sleep(choice([4, 5, 6, 8, 9, 10, 12]))

# --- Main Execution ---

def main():
    # Create the base directory if it doesn't exist
    if not os.path.exists(PROJECT_PATH):
        os.makedirs(PROJECT_PATH, exist_ok=True)
        print(f"Created directory: {PROJECT_PATH}")
        
    topic_dir = os.path.join(PROJECT_PATH, 'topic')
    if not os.path.exists(topic_dir):
        os.makedirs(topic_dir, exist_ok=True)
        print(f"Created directory: {topic_dir}")
    
    # Start hierarchical extraction for a specific topic.
    topic = "Capital Gains Tax"
    print(f"Starting hierarchical extraction for {topic}...")
    if recursive_extraction_for_topic(save_dir=topic_dir, save_file='main.jsonl', topic=topic):
        print(f"Hierarchy extraction for {topic} completed.")
    
    # Build file index from the extracted hierarchy.
    file_index_path = os.path.join(topic_dir, 'file_index.jsonl')
    if recursive_file_index(save_dir=topic_dir, save_file='main.jsonl', save_path=file_index_path):
        print("File index creation completed.")
    
    # Extract ONLY 10 pages for testing
    pages_index = read_json_lines(file_index_path)
    cgt_pages_dir = os.path.join(topic_dir, f"{topic.replace(' ', '_')}_pages_json")
    
    # Limit to first 10 pages
    test_pages = pages_index[:10]
    extract_pages(test_pages, cgt_pages_dir)
    print(f"Test extraction completed. 10 JSON files saved to: {cgt_pages_dir}")

if __name__ == '__main__':
    main()

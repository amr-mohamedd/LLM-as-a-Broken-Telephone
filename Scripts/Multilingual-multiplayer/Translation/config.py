"""
Configuration file for translation pipeline settings and prompts.
"""
import logging

# Prompts
prompt1 = "Translate the following text from {} to {}: {}"
prompt2 = "Respond only with a valid translation. Do not write an introduction or a summary. Translate the following text from {} to {}: {}"
prompt3 = """
You are a translation expert. Given a passage, a source language, and a target language, translate the passage from the source language to the target language while preserving all the original meaning and without losing any context. 
Do not write an introduction or a summary. Return only the translated passage.
Translate the following text from {} to {}: {}
"""
prompt4 = """
You are about to participate in a translation game. Please follow these instructions carefully:
    
- **Task**: You will receive a paragraph in **{}**.
- **Objective**: Translate the paragraph into **{}**.
- **Guidelines**:
  - Do not write an introduction or a summary. 
  - Preserve the original meaning entirely; ensure no information is lost or altered.
  - Do not add, omit, or modify any details from the original paragraph.
  - Maintain the tone and style as closely as possible.
      
- **Paragraph**: {}
"""

# Hugging Face API token
hf_auth_token = 'YOUR_HF_TOKEN'

# setting configs
num_texts = 150
num_translations = 100

# Language mapping dictionary
lang_dict_map = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "pr": "Portuguese",
    "it": "Italian",
    "th": "Thai",
    "hi": "Hindi",
    "ar": "Arabic",
    "nl": "Dutch",
    "vn": "Vietnamese",
    "zh": "Chinese"
}

# Default file paths and directories
default_file_path = './../../../Datasets/all_data.csv'
log_dir = './logs'
output_dir = './results'

# Logging configurations
logging_format = '%(asctime)s - %(levelname)s - %(message)s'
logging_level = logging.INFO

# Shuffle setting
default_shuffle = False

# Decoding parameter configurations for Gemma and Llama
def get_gemma_config(max_new_tokens=None):
    if max_new_tokens is None:
        max_new_tokens = 8000
    return {
        'max_new_tokens': max_new_tokens,
        'use_cache': False,
        'do_sample': True,
        'temperature': 1.0
    }

def get_llama_config(max_new_tokens=None):
    if max_new_tokens is None:
        max_new_tokens = 8000
    return {
        'temperature': 0.6,
        'top_p': 0.9,    
        "max_new_tokens": max_new_tokens
        }

def get_mistral_config(max_new_tokens=None):
    if max_new_tokens is None:
        max_new_tokens = 8000
    return {
        'max_new_tokens': max_new_tokens,
        'use_cache': False,
        'do_sample': True,
    }
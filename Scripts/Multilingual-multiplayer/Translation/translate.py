import argparse
import pandas as pd
import json
from tqdm import tqdm
import datetime
import logging
import random
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import prompt3 as prompt, hf_auth_token, num_texts, num_translations, lang_dict_map, log_dir, output_dir, logging_format, logging_level, get_gemma_config, get_llama_config, get_mistral_config
import numpy as np

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def run_translation_pipeline(model_names, dataset_name, cuda_device, language_sequence, shuffle, default_file_path):
    """
    Run the translation pipeline to translate texts through multiple languages.

    Args:
        model_names (list): List of model names to use for translation.
        dataset_name (str): Name of the dataset column to translate.
        language_sequence (list): Sequence of language codes to translate through.
        shuffle (bool): Whether to shuffle the language sequence each iteration.
    """
    # Set device
    if cuda_device is not None and torch.cuda.is_available():
        device = f"cuda:{cuda_device}"
    else:
        device = "auto" if torch.cuda.is_available() else "cpu"
    logging.info("Using device: %s", device)

    if shuffle is None:
        shuffle = False

    gemma_tokenizer = None
    llama_tokenizer = None
    mistral_tokenizer = None


    if "meta-llama/Llama-3.1-8B-Instruct" in (model_names):
        # Load Model and Tokenizer
        llama_model_name = "meta-llama/Llama-3.1-8B-Instruct"
        logging.info("Loading tokenizer for model: %s", llama_model_name)
        llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name, token=hf_auth_token)
        logging.info("Loading model: %s", llama_model_name)
        llama_model = AutoModelForCausalLM.from_pretrained(
            llama_model_name,
            device_map="auto", #if device == "auto" else {"": device},
            torch_dtype=torch.bfloat16,
            token=hf_auth_token,
            use_cache=False,
            trust_remote_code=True
        )
        llama_model.generation_config.pad_token_id = llama_tokenizer.pad_token_id

    if "google/gemma-2-9b-it" in (model_names):
        gemma_model_name = "google/gemma-2-9b-it"
        logging.info("Loading tokenizer for model: %s", gemma_model_name)
        gemma_tokenizer = AutoTokenizer.from_pretrained(gemma_model_name)
        logging.info("Loading model: %s", gemma_model_name)
        gemma_model = AutoModelForCausalLM.from_pretrained(
            gemma_model_name,
            device_map="auto", #if device == "auto" else {"": device},
            torch_dtype=torch.bfloat16,
            use_cache=False,
            trust_remote_code=True
        )
        gemma_model.generation_config.pad_token_id = gemma_tokenizer.pad_token_id
    
    if "mistralai/Mistral-7B-Instruct-v0.2" in (model_names):
        mistral_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        logging.info("Loading tokenizer for model: %s", mistral_model_name)
        mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model_name, token=hf_auth_token)
        logging.info("Loading model: %s", mistral_model_name)
        mistral_model = AutoModelForCausalLM.from_pretrained(
            mistral_model_name,
            device_map="auto", #if device == "auto" else {"": device},
            torch_dtype=torch.bfloat16,
            token=hf_auth_token,
            use_cache=False,
            trust_remote_code=True
        )
        mistral_model.generation_config.pad_token_id = mistral_tokenizer.pad_token_id
        
    def get_response_llama(messages, model , tokenizer):
        """Get response from Llama model."""
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        llama_config = get_llama_config()
        outputs = model.generate(
            input_ids=input_ids,
            **llama_config
        )
        response = outputs[0][input_ids.shape[-1]:]
        output = tokenizer.decode(response, skip_special_tokens=True)
        return output
    
    def get_response_gemma(messages, model, tokenizer):
        """Get response from Gemma model."""
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)#.to(model.device)
        inputs = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt").to(model.device)
        gemma_config = get_gemma_config()
        outputs = model.generate(
            input_ids=inputs.to(model.device),
            **gemma_config
        )
        output = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return output

    def get_response_mistral(messages):
        """Get response from Mistral model."""
        input_ids = mistral_tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(mistral_model.device)
        mistral_config = get_mistral_config()
        outputs = mistral_model.generate(input_ids, **mistral_config)
        output = mistral_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output.split('[/INST]')[-1].strip()
    
    def translate_message(text, src_lang, tgt_lang):
        """Translate text from source to target language using a random model."""
        logging.info("Translating from %s to %s", src_lang, tgt_lang)
        selected_model = random.choice(model_names)
        logging.info("Selected model: %s", selected_model)

        # Validate input text
        if not text or text.isspace():
            logging.error("Empty or whitespace-only text received")
            raise ValueError("Cannot translate empty text")

        # Perform translation based on the selected model
        if "meta-llama/Llama-3.1-8B-Instruct" in selected_model:
            messages = [{"role": "user", "content": prompt.format(src_lang, tgt_lang, text.strip())}]
            translated_text_passage = get_response_llama(messages, model=llama_model, tokenizer=llama_tokenizer)
        elif "mistralai/Mistral-7B-Instruct-v0.2" in selected_model:
            messages = [{"role": "user", "content": prompt.format(src_lang, tgt_lang, text.strip())}]
            translated_text_passage = get_response_mistral(messages)
        elif "google/gemma-2-9b-it" in selected_model:
            messages = [{"role": "user", "content": prompt.format(src_lang, tgt_lang, text.strip())}]
            translated_text_passage = get_response_gemma(messages, model=gemma_model, tokenizer=gemma_tokenizer)
        else:
            raise ValueError(f'No model inference function found for the model: {selected_model}')

        # Validate output
        if not translated_text_passage or translated_text_passage.isspace():
            logging.error("Empty translation received from model")
            raise ValueError(f"Model {selected_model} returned empty translation")

        # Clean up the output
        translated_text_passage = translated_text_passage.strip()
        
        # Remove any model references from the output
        if 'model' in translated_text_passage.lower():
            parts = translated_text_passage.lower().split('model')
            translated_text_passage = parts[-1].strip()

        # Split the output when 'model' appears
        while 'model' in translated_text_passage:
            translated_text_passage = translated_text_passage.split('model', 1)[1].strip()

        # Remove the prompt and clean up the output
        return translated_text_passage.replace(prompt, "").strip(), selected_model

    def iterative_translation(passage, n, language_sequence, shuffle=False):
        """Perform iterative translation through multiple languages."""
        translated_passage = passage
        sequence_length = len(language_sequence) + 2
        translated_passages = {"prompt": prompt, "0_en": passage, "model_sequence": []}
        forced_next_lang = "en"

        for i in range(1, n + 1):
            if shuffle:
                shuffled_language_sequence = [k for k in language_sequence if k != forced_next_lang]
                random.shuffle(shuffled_language_sequence)
                language_sequence = ["en"] + shuffled_language_sequence + ["en"]
            
            if sequence_length < 2:
                raise ValueError("Language sequence must contain at least 2 languages after shuffling.")
            
            for j in range(0, sequence_length-1):
                src_lang = language_sequence[j]
                tgt_lang = language_sequence[j+1]
                logging.info("Iteration %d/%d: Translating from %s to %s", i, n, src_lang, tgt_lang)
                translated_passage, selected_model = translate_message(translated_passage, lang_dict_map[src_lang], lang_dict_map[tgt_lang])
                translated_passages[f'{i}_{tgt_lang}'] = translated_passage
                translated_passages["model_sequence"].append(selected_model)
            forced_next_lang = tgt_lang

        return translated_passages


    def validate_language_sequence(language_sequence):
        """Validate that all languages in the sequence exist in lang_dict_map"""
        invalid_langs = [lang for lang in language_sequence if lang not in lang_dict_map]
        if invalid_langs:
            raise ValueError(f"Invalid language codes found: {invalid_langs}")
    
    # Validate the language sequence
    validate_language_sequence(language_sequence)

    # Load and clean passages
    logging.info("Loading and cleaning passages")
    passages = pd.read_csv("./../../../Datasets/" + default_file_path)

    # Validate dataset columns
    if 'id' not in passages.columns:
        raise ValueError("DataFrame is missing required 'id' column")
    if dataset_name not in passages.columns:
        raise ValueError(f"DataFrame is missing required '{dataset_name}' column")
    
    # Ensure num_texts doesn't exceed DataFrame size
    actual_num_texts = min(num_texts, len(passages))
    if actual_num_texts < num_texts:
        logging.warning(f"num_texts ({num_texts}) exceeds available passages ({actual_num_texts}). Using {actual_num_texts} instead.")
    
    # Create output filename at the start
    file_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_path = f'{output_dir}/translated_passages_{dataset_name}_{"_".join(model_names).replace("/", "_")}_trans_{"_".join(language_sequence)}_{file_timestamp}.json'
    logging.info("Will save translated passages to %s", output_file_path)

    # Translate passages
    logging.info("Starting translation for %d texts.", actual_num_texts)

    # Check if output file exists and load existing results if so
    if os.path.exists(output_file_path):
        with open(output_file_path, 'r', encoding='utf-8') as json_file:
            results = json.load(json_file)
    else:
        results = {}

    for i in tqdm(range(actual_num_texts)):
        passage_id = str(passages['id'].iloc[i])
        if passage_id in results:
            logging.info("Skipping passage with id '%s' as it is already processed.", passage_id)
            continue

        content = passages[dataset_name].iloc[i]
        logging.info("Translating passage %d/%d with id '%s'", i + 1, actual_num_texts, passage_id)

        translated_passages = iterative_translation(
            content, 
            num_translations, 
            language_sequence, 
            shuffle=shuffle
        )

        results[passage_id] = translated_passages

        # Append new results incrementally
        with open(output_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(results, json_file, ensure_ascii=False, indent=2)
            
    logging.info("Translation pipeline completed successfully.")

if __name__ == "__main__":
    os.environ["TRANSFORMERS_NO_TF"] = "1"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_names", type=str, nargs='+', required=True, help="The list of model names to use for translation.")
    parser.add_argument("--dataset_name", type=str, required=True, help="The dataset name to use for translation.")
    parser.add_argument("--language_sequence", type=str, nargs='+', required=True, help="Sequence of languages to translate through.")
    parser.add_argument("--cuda_device", type=int, required=False, help="The CUDA device to use for translation.")
    parser.add_argument("--shuffle", action="store_true", help="Enable shuffling of the language sequence.")
    parser.add_argument("--no-shuffle", action="store_false", dest="shuffle", help="Disable shuffling of the language sequence.")
    parser.add_argument("--default_file_path", type=str, required=True, help="The path to the default file to use for translation.")
    parser.set_defaults(shuffle=False)
    args = parser.parse_args()

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Set up logging configuration with a dynamic log filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_dir}/translation_{'_'.join(args.model_names).replace('/', '_')}_trans_{args.dataset_name}_{'_'.join(args.language_sequence)}_{timestamp}.log"

    logging.basicConfig(filename=log_filename,
                        level=logging_level, format=logging_format)

    logging.info("Script arguments: %s", args)

    if len(args.language_sequence) < 2:
        raise ValueError("Language sequence must contain at least 2 languages")

    run_translation_pipeline(
        args.model_names,
        args.dataset_name,
        args.cuda_device,
        args.language_sequence,
        args.shuffle,
        args.default_file_path
    )
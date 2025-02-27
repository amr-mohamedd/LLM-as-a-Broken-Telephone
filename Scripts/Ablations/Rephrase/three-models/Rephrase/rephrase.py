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
from config import prompt3 as prompt, hf_auth_token, num_texts, log_dir, output_dir, logging_format, logging_level, get_gemma_config, get_llama_config, get_mistral_config
import time
import numpy as np

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def run_rephrasing_pipeline(model_names, dataset_name, cuda_device, num_rephrasings, default_file_path):
    """
    Run the rephrasing pipeline to rephrase texts iteratively.

    Args:
        model_names (list): List of model names to use for rephrasing.
        dataset_name (str): Name of the dataset column to rephrase.
        num_rephrasings (int): Number of times to rephrase each text.
    """

    # Set device
    if cuda_device is not None and torch.cuda.is_available():
        device = f"cuda:{cuda_device}"
    else:
        device = "auto" if torch.cuda.is_available() else "cpu"
    logging.info("Using device: %s", device)

    # Load Models and Tokenizers
    llama_model_name = "meta-llama/Llama-3.1-8B-Instruct"
    mistral_model_name = "mistral.mistral-7b-instruct-v0:2"
    gemma_model_name = "google/gemma-2-9b-it"

    logging.info("Loading tokenizer for model: %s", llama_model_name)
    llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name, token=hf_auth_token)
    logging.info("Loading model: %s", llama_model_name)
    llama_model = AutoModelForCausalLM.from_pretrained(
        llama_model_name,
        device_map="auto" if device == "auto" else {"": device},
        torch_dtype=torch.bfloat16,
        token=hf_auth_token,
        use_cache=False,
        trust_remote_code=True
    )
    llama_model.generation_config.pad_token_id = llama_tokenizer.pad_token_id


    logging.info("Loading tokenizer for model: %s", mistral_model_name)
    mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model_name, token=hf_auth_token)
    logging.info("Loading model: %s", mistral_model_name)
    mistral_model = AutoModelForCausalLM.from_pretrained(
        mistral_model_name,
        device_map="auto" if device == "auto" else {"": device},
        torch_dtype=torch.bfloat16,
        token=hf_auth_token,
        use_cache=False,
        trust_remote_code=True
    )
    mistral_model.generation_config.pad_token_id = mistral_tokenizer.pad_token_id
    

    logging.info("Loading tokenizer for model: %s", gemma_model_name)
    gemma_tokenizer = AutoTokenizer.from_pretrained(gemma_model_name, token=hf_auth_token)
    logging.info("Loading model: %s", gemma_model_name)
    gemma_model = AutoModelForCausalLM.from_pretrained(
        gemma_model_name,
        device_map="auto" if device == "auto" else {"": device},
        torch_dtype=torch.bfloat16,
        token=hf_auth_token,
        use_cache=False,
        trust_remote_code=True
    )
    gemma_model.generation_config.pad_token_id = gemma_tokenizer.pad_token_id

    def get_response_llama(messages):
        """Get response from Llama model."""
        input_ids = llama_tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(llama_model.device)
        llama_config = get_llama_config()
        outputs = llama_model.generate(
            input_ids=input_ids,
            **llama_config
        )
        response = outputs[0][input_ids.shape[-1]:]
        output = llama_tokenizer.decode(response, skip_special_tokens=True)
        return output

    def get_response_mistral(messages):
        """Get response from Mistral model."""
        input_ids = mistral_tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(mistral_model.device)
        mistral_config = get_mistral_config()
        outputs = mistral_model.generate(input_ids, **mistral_config)
        output = mistral_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output.split('[/INST]')[-1].strip()


    def get_response_gemma(messages):
        """Get response from Gemma model."""
        prompt_text = gemma_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = gemma_tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
        gemma_config = get_gemma_config()
        outputs = gemma_model.generate(
            input_ids=inputs.to(gemma_model.device),
            **gemma_config
        )
        output = gemma_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return output

    def rephrase_message(text):
        """Rephrase text using a randomly selected model."""
        logging.info("Rephrasing text")
        selected_model = random.choice(model_names)
        logging.info("Selected model: %s", selected_model)
        
        if "llama" in selected_model.lower():
            message = [{
                "role": "user",
                "content": prompt.format(text.strip())
            }]
            rephrased_text = get_response_llama(message)
        elif "mistral" in selected_model.lower():
            message = [{
                "role": "user",
                "content": prompt.format(text.strip())
            }]
            rephrased_text = get_response_mistral(message)
        elif "gemma" in selected_model.lower():
            message = [{
                "role": "user",
                "content": prompt.format(text.strip())
            }]
            rephrased_text = get_response_gemma(message)
        else:
            rephrased_text = 'No model inference function found for the specified model.'
        
        if text in rephrased_text:
            rephrased_text = rephrased_text.split(text)[1:]
            if isinstance(rephrased_text, list):
                rephrased_text = " ".join(rephrased_text)
        if "model\n" in rephrased_text:
            rephrased_text = rephrased_text.split("model\n")[1:]
            # join the list into a string if it's a list
            if isinstance(rephrased_text, list):
                rephrased_text = " ".join(rephrased_text)
        return rephrased_text.replace(prompt, "").strip(), selected_model

    def iterative_rephrasing(passage, n):
        """Perform iterative rephrasing."""
        rephrased_passage = passage
        rephrased_passages = {"prompt": prompt, "0_en": passage, "model_sequence": []}
        for i in range(1, n+1):
            logging.info("Iteration %d/%d: Rephrasing text", i, n)
            rephrased_passage, selected_model = rephrase_message(rephrased_passage)
            rephrased_passages[f'{i}_en'] = rephrased_passage
            rephrased_passages["model_sequence"].append(selected_model)

        return rephrased_passages

    # Load and clean passages
    logging.info("Loading and cleaning passages")
    passages = pd.read_csv("./../../../../../Datasets/" + default_file_path)

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
    output_file_path = f'{output_dir}/rephrased_passages_{dataset_name}_{"_".join(model_names).replace("/", "_")}_rephrased_{file_timestamp}_{default_file_path}.json'
    logging.info("Will save rephrased passages to %s", output_file_path)

    # Rephrase passages
    logging.info("Starting rephrasing for %d texts.", actual_num_texts)

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
        logging.info("Rephrasing passage %d/%d with id '%s'", i + 1, actual_num_texts, passage_id)

        rephrased_passages = iterative_rephrasing(
            content, 
            num_rephrasings
        )

        results[passage_id] = rephrased_passages

        # Append new results incrementally
        with open(output_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(results, json_file, ensure_ascii=False, indent=2)
            
    logging.info("Rephrasing pipeline completed successfully.")

if __name__ == "__main__":
    os.environ["TRANSFORMERS_NO_TF"] = "1"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_names", type=str, nargs='+', required=True, help="The list of model names to use for rephrasing.")
    parser.add_argument("--dataset_name", type=str, required=True, help="The dataset name to use for rephrasing.")
    parser.add_argument("--cuda_device", type=int, required=False, help="The CUDA device to use for rephrasing.")
    parser.add_argument("--num_rephrasings", type=int, default=1, help="Number of times to rephrase each text.")
    parser.add_argument("--default_file_path", type=str, required=True, help="The path to the default file to use for rephrasing.")
    args = parser.parse_args()

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Set up logging configuration with a dynamic log filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_dir}/rephrasing_{'_'.join(args.model_names).replace('/', '_')}_rephrased_{args.dataset_name}_{timestamp}_{args.default_file_path}.log"

    logging.basicConfig(filename=log_filename,
                        level=logging_level, format=logging_format)

    logging.info("Script arguments: %s", args)

    run_rephrasing_pipeline(
        args.model_names,
        args.dataset_name,
        args.cuda_device,
        args.num_rephrasings,
        args.default_file_path
    )
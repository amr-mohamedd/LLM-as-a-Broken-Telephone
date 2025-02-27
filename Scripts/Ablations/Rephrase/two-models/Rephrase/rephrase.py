import argparse
import torch
import pandas as pd
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch 
import datetime
import logging
import random
import time
import os
from config import prompt3 as prompt, hf_auth_token, log_dir, output_dir, logging_format, logging_level, get_mistral_config, get_llama_config
import numpy as np

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def load_data(file_path, chunk_size=None):
    """Load data with optional chunking."""
    if chunk_size:
        return pd.read_csv(file_path, chunksize=chunk_size)
    else:
        return pd.read_csv(file_path)

def save_results_incrementally(results, output_file_path):
    """Save results incrementally to avoid memory issues."""
    with open(output_file_path, 'a', encoding='utf-8') as json_file:
        for result in results:
            json.dump(result, json_file, ensure_ascii=False, indent=2)
            json_file.write('\n')

def run_rephrasing_pipeline(model_names, dataset_name, cuda_device, num_rephrasings, default_file_path):
    """
    Run the rephrasing pipeline to rephrase texts iteratively.

    Args:
        model_names (list): List of model names to use for rephrasing.
        dataset_name (str): Name of the dataset column to rephrase.
        cuda_device (int): CUDA device ID to use, or None for CPU.
        num_rephrasings (int): Number of times to rephrase each text.
    """
    # Set device
    if cuda_device is not None and torch.cuda.is_available():
        device = f"cuda:{cuda_device}"
    else:
        device = "auto" if torch.cuda.is_available() else "cpu"
    logging.info("Using device: %s", device)

    # Load Model and Tokenizer
    llama_model_name = "meta-llama/Llama-3.1-8B-Instruct"
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
    torch.cuda.empty_cache()
    llama_model.generation_config.pad_token_id = llama_tokenizer.pad_token_id
    torch.cuda.empty_cache()

    mistral_model_name = "mistralai/Mistral-7B-v0.2"
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
    torch.cuda.empty_cache()
    mistral_model.generation_config.pad_token_id = mistral_tokenizer.pad_token_id
    torch.cuda.empty_cache()

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
            torch.cuda.empty_cache()
            rephrased_text = get_response_llama(message)
            torch.cuda.empty_cache()
        elif "mistral" in selected_model.lower():
            message = [{
                "role": "user",
                "content": prompt.format(text.strip())
            }]
            torch.cuda.empty_cache()
            rephrased_text = get_response_mistral(message)
            torch.cuda.empty_cache()
        else:
            rephrased_text = 'No model inference function found for the specified model.'
        
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

    # Load passages in chunks
    logging.info("Loading and cleaning passages")
    passages = load_data('./../../../../../Datasets/' + default_file_path)

    file_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_path = os.path.join(
        output_dir,
        f'{dataset_name}/rephrased_passages_{dataset_name}_{model_names[0].split("/")[-1]}_rephrase_{file_timestamp}.json'
    )

    # Rephrase passages
    total_passages = len(passages)
    logging.info(f"Starting rephrasing for {total_passages} texts.")

    def process_passage(index):
        passage_id = str(passages['id'][index])
        content = passages[dataset_name][index]
        logging.info(f"Rephrasing passage {index + 1}/{total_passages} with id '{passage_id}'")

        rephrased_passages = iterative_rephrasing(
            content,
            num_rephrasings
        )
        return {passage_id: rephrased_passages}

    # Rephrase passages sequentially
    results = []
    for index in tqdm(range(total_passages), desc="Rephrasing passages"):
        passage_id = str(passages['id'][index])
        content = passages[dataset_name][index]
        logging.info(f"Rephrasing passage {index + 1}/{total_passages} with id '{passage_id}'")
    
        rephrased_passages = iterative_rephrasing(
            content,
            num_rephrasings
        )
        result = {passage_id: rephrased_passages}
        results.append(result)
        save_results_incrementally([result], output_file_path)


    logging.info(f"Rephrasing pipeline completed successfully. Output saved to {output_file_path}")

if __name__ == "__main__":
    os.environ["TRANSFORMERS_NO_TF"] = "1"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_names", type=str, nargs='+', required=True, help="The list of model names to use for rephrasing.")
    parser.add_argument("--dataset_name", type=str, required=True, help="The dataset name to use for translation.")
    parser.add_argument("--cuda_device", type=int, required=False, help="Which CUDA device to use (optional).")
    parser.add_argument("--num_rephrasings", type=int, default=1, help="Number of times to rephrase each text.")
    parser.add_argument("--default_file_path", type=str, default="all_data.csv", help="The default file path to use for translation.")
    parser.add_argument("--random_seed", type=int, default=42, help="The random seed to use for translation.")
    args = parser.parse_args()

    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(f'./results/{args.dataset_name}'), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Set up logging configuration with a dynamic log filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_dir}/rephrase_{args.model_names[0].replace('/', '_')}_rephrase_{args.dataset_name}_{timestamp}_{args.random_seed}.log"
    logging.basicConfig(filename=log_filename,
                        level=logging_level, format=logging_format)

    logging.info("Script arguments: %s", args)

    start_time = time.time()
    run_rephrasing_pipeline(
        args.model_names,
        args.dataset_name,
        args.cuda_device,
        args.num_rephrasings,
        args.default_file_path
    )
    end_time = time.time()
    logging.info(f"Rephrasing pipeline completed successfully. Time taken: {end_time - start_time} seconds")
    print(f"Rephrasing pipeline completed successfully. Time taken: {end_time - start_time} seconds")
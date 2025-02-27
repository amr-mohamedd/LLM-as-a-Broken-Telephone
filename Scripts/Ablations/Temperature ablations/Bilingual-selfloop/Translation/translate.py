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
from config import prompt1 as prompt, hf_auth_token, num_translations, lang_dict_map, log_dir, output_dir, logging_format, logging_level, get_gemma_config, get_llama_config
from concurrent.futures import ThreadPoolExecutor
import numpy as np



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

def run_translation_pipeline(model_name, dataset_name, cuda_device, language_sequence, shuffle, default_file_path, temperature, seed):
    """
    Run the translation pipeline to translate texts through multiple languages.

    Args:
        model_name (str): Name of the model to use for translation
        dataset_name (str): Name of the dataset column to translate
        cuda_device (int): CUDA device ID to use, or None for CPU
        language_sequence (list): Sequence of language codes to translate through
        shuffle (bool): Whether to shuffle the language sequence each iteration
    """

    # Set device
    if cuda_device is not None and torch.cuda.is_available():
        device = f"cuda:{cuda_device}"
    else:
        device = "auto" if torch.cuda.is_available() else "cpu"
    logging.info("Using device: %s", device)

    if shuffle is None:
        shuffle = False

    # Load Model and Tokenizer
    logging.info("Loading tokenizer for model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_auth_token)
    logging.info("Loading model: %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device == "auto" else {"": device},
        torch_dtype=torch.bfloat16,
        token=hf_auth_token,
        use_cache=False,
        trust_remote_code=True
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    def get_response_llama(messages):
        """Get response from Llama model."""
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=8000,
            eos_token_id=[128001, 128008, 128009],
            use_cache=False,
            do_sample=True,
            temperature=temperature,
            top_p=0.9
        )
        response = outputs[0][input_ids.shape[-1]:]
        output = tokenizer.decode(response, skip_special_tokens=True)
        return output

    def translate_message(text, src_lang, tgt_lang):
        """Translate text from source to target language."""
        logging.info("Translating from %s to %s", src_lang, tgt_lang)
        torch.cuda.empty_cache()
        message = [{
            "role": "user",
            "content": prompt.format(src_lang, tgt_lang, text.strip())
        }]
        if "llama" in model_name.lower():
            translated_text_passage = get_response_llama(message)
        else:
            translated_text_passage = 'No model inference function found for the specified model.'
        torch.cuda.empty_cache()
        return translated_text_passage.replace(prompt, "").strip()

    def iterative_translation(passage, n, language_sequence, shuffle=False):
        """Perform iterative translation through multiple languages."""
        translated_passage = passage
        sequence_length = len(language_sequence)
        translated_passages = {"prompt": prompt, "0_en": passage}
        forced_next_lang = "en"
        for i in range(1, n+1):
            if shuffle:
                shuffled_language_sequence = [k for k in language_sequence if k != forced_next_lang]
                random.shuffle(shuffled_language_sequence)
                language_sequence = [forced_next_lang] + shuffled_language_sequence
            for j in range(sequence_length - 1):
                src_lang = language_sequence[j]
                tgt_lang = language_sequence[j + 1]
                logging.info("Iteration %d/%d: Translating from %s to %s", i, n, src_lang, tgt_lang)
                translated_passage = translate_message(translated_passage, lang_dict_map[src_lang], lang_dict_map[tgt_lang])
                translated_passages[f'{i}_{tgt_lang}'] = translated_passage
            forced_next_lang = tgt_lang

        return translated_passages

    # Load passages in chunks
    logging.info("Loading and cleaning passages")
    passages = load_data('./../../../../../Datasets/30_passages/' + default_file_path)

    file_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_path = os.path.join(
        output_dir,
        f'{dataset_name}/translated_passages_{dataset_name}_{model_name.split("/")[-1]}_trans_{"_".join(language_sequence)}_{file_timestamp}_{temperature}_{seed}.json'
    )

    # Translate passages
    total_passages = len(passages)
    logging.info(f"Starting translation for {total_passages} texts.")

    def process_passage(index):
        passage_id = str(passages['id'][index])
        content = passages[dataset_name][index]
        logging.info(f"Translating passage {index + 1}/{total_passages} with id '{passage_id}'")

        translated_passages = iterative_translation(
            content,
            num_translations,
            language_sequence,
            shuffle=shuffle
        )
        return {passage_id: translated_passages}

    # Translate passages sequentially
    results = []
    for index in tqdm(range(total_passages), desc="Translating passages"):
        passage_id = str(passages['id'][index])
        content = passages[dataset_name][index]
        logging.info(f"Translating passage {index + 1}/{total_passages} with id '{passage_id}'")
    
        translated_passages = iterative_translation(
            content,
            num_translations,
            language_sequence,
            shuffle=shuffle
        )
        result = {passage_id: translated_passages}
        results.append(result)
        save_results_incrementally([result], output_file_path)


    logging.info(f"Translation pipeline completed successfully. Output saved to {output_file_path}")

if __name__ == "__main__":
    os.environ["TRANSFORMERS_NO_TF"] = "1"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="The model name to use for translation.")
    parser.add_argument("--dataset_name", type=str, required=True, help="The dataset name to use for translation.")
    parser.add_argument("--cuda_device", type=int, required=False, help="Which CUDA device to use (optional).")
    parser.add_argument("--language_sequence", type=str, nargs='+', required=True, help="Sequence of languages to translate through.")
    parser.add_argument("--shuffle", action="store_true", help="Enable shuffling of the language sequence.")
    parser.add_argument("--no-shuffle", action="store_false", dest="shuffle", help="Disable shuffling of the language sequence.")
    parser.set_defaults(shuffle=False)
    parser.add_argument("--default_file_path", type=str, default="all_data.csv", help="The default file path to use for translation.")
    parser.add_argument("--temperature", type=float, default=0, required=True, help="The temperature to use for translation.")
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
    log_filename = f"{log_dir}/translation_{args.model_name.replace('/', '_')}_trans_{args.dataset_name}_{args.language_sequence}_{timestamp}_{args.temperature}_{args.random_seed}.log"
    logging.basicConfig(filename=log_filename,
                        level=logging_level, format=logging_format)

    logging.info("Script arguments: %s", args)

    start_time = time.time()
    run_translation_pipeline(
        args.model_name,
        args.dataset_name,
        args.cuda_device,
        args.language_sequence,
        args.shuffle,
        args.default_file_path,
        args.temperature,
        args.random_seed
    )
    end_time = time.time()
    logging.info(f"Translation pipeline completed successfully. Time taken: {end_time - start_time} seconds")
    print(f"Translation pipeline completed successfully. Time taken: {end_time - start_time} seconds")
import argparse
import pandas as pd
import os
import re
import json
from factscorer.factscore.factscorer import FactScorer
import nltk
from nltk.translate.meteor_score import meteor_score
from torchmetrics.text import BLEUScore, ROUGEScore
from nltk.translate.chrf_score import sentence_chrf
from bert_score import score
from transformers import logging
import sys
import datetime
import en_core_web_sm
nlp = en_core_web_sm.load()
nltk.download('punkt_tab')
nltk.download('wordnet')
logging.set_verbosity_error()
LOG_DIR = './../../../Results/Evaluation/Logs/'
os.makedirs(LOG_DIR, exist_ok=True)  # Create the logs directory if it doesn't exist

def get_log_filename(input_path):
    """Generate a single log filename for the entire JSON processing."""
    # Create all parent directories
    os.makedirs(os.path.dirname(LOG_DIR), exist_ok=True)
    
    base_name = os.path.basename(input_path).split('.')[0]
    log_file = os.path.join(LOG_DIR, f"{base_name}.log")
    
    # Create parent directory for the log file if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    return log_file

def log_message(message, log_file):
    """Log a message with timestamp to the log file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} - {message}\n")

# Metric calculation functions
def compute_meteor(reference, candidate):
    return meteor_score([nltk.word_tokenize(reference)], nltk.word_tokenize(candidate))

def compute_bertscore(reference, candidate):
    P, R, F1 = score([candidate], [reference], lang='en')
    return P, R, F1

def compute_all_metrics(reference, candidate):
    if type(candidate) == float or not candidate.strip():
        return {}
    P, R, F1 = compute_bertscore(reference, candidate)
    metrics = {
        'BLEU': float(BLEUScore()([reference], [[candidate]])),
        'ROUGE-1': float(ROUGEScore()(reference, candidate)['rouge1_fmeasure']),
        'ROUGE-L': float(ROUGEScore()(reference, candidate)['rougeL_fmeasure']),
        'METEOR': compute_meteor(reference, candidate),
        'BERTScore Precision': P.mean().item(),
        'BERTScore Recall': R.mean().item(),
        'BERTScore F1': F1.mean().item(),
        'CHR-F': sentence_chrf(candidate, reference)
    }
    return metrics

def get_titles(dataset_name, results_path):
    knowledge_source_path = f'./../../../Datasets/Knowledge sources/{dataset_name}.jsonl'
    if not os.path.exists(knowledge_source_path):
        raise FileNotFoundError(f"Knowledge source file not found: {knowledge_source_path}")
    knowledge_source = pd.read_json(knowledge_source_path, lines=True)
    results_df = pd.read_json(results_path)

    merged = pd.merge(
    left=results_df.T[['0_en']],
    right=knowledge_source,
    left_on='0_en',
    right_on='text',
    how='inner')[['0_en' , 'title']]
    return dict(zip(merged['0_en'], merged['title']))

def extract_and_save(file_path, nested_results):
    file_name = file_path.split('/')[-1]
    dataset_name = file_path.split('/')[-2]
    model_name = file_path.split('/')[-3]
    setting_name = file_path.split('/')[-4]

    output_dir = f"./../../../Results/Evaluation/{setting_name}/{model_name}/{dataset_name}/"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"{file_name}_results.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(nested_results, f, ensure_ascii=False, indent=4)

    print(f"Results saved to {output_file}")
    return output_file

def sanitize_text(text):
    """Sanitize text by replacing double quotes with single quotes and handling NaN."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    return text.replace('"', "'").strip()

def process_file_to_nested_dict(
    file_path, 
    metrics_type='factscore',   # <-- Default to factscore
    anthropic_model_version='anthropic.claude-3-5-sonnet-20241022-v2:0'
):
    """
    Reads a JSON file of generation results, evaluates them EITHER with FactScorer 
    OR with other metrics from `compute_all_metrics`, then saves incremental 
    nested JSON results keyed by title and generation number.
    metrics_type:
        - 'factscore': compute FactScorer only
        - 'additional': compute only the metrics from compute_all_metrics
    """

    # -------------------------------------------------------------------------
    # 1. Parse file_path to get setting/dataset/model/file names
    # -------------------------------------------------------------------------
    file_name     = file_path.split('/')[-1]
    if metrics_type == 'additional':
        output_file = 'additionalmetrics_' + file_name

    metric_dir = 'Factuality' if metrics_type == 'factscore' else 'Other metrics'

    dataset_name  = file_path.split('/')[-2]
    model_name    = file_path.split('/')[-3]
    setting_name  = file_path.split('/')[-4]
    output_dir    = f"./../../../Results/Evaluation/{setting_name}/{metric_dir}/{model_name}/{dataset_name}/"

    output_file   = os.path.join(output_dir, file_name)

    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Prepare known (int) generations to evaluate
    # -------------------------------------------------------------------------
    original_gens = [0, 2, 4, 6, 8, 10, 13, 16, 20, 25, 30, 40, 50, 65, 80, 100]

    # -------------------------------------------------------------------------
    # 3. Load and filter the DataFrame
    # -------------------------------------------------------------------------
    log_file = get_log_filename(file_path)

    results_df = pd.read_json(file_path)
    results_df = results_df.reindex(sorted(results_df.columns), axis=1).reset_index().rename(columns={'index': 'gen'})
    # Replace NaN with empty string and sanitize all text columns
    results_df = results_df.fillna("")
    # Sanitize all non-index columns (assuming first column is 'gen')
    for col in results_df.columns[1:]:
        results_df[col] = results_df[col].apply(sanitize_text)
    results_df = results_df[results_df['gen'].str.contains('en', na=False)].copy()
    results_df['gen_num'] = results_df['gen'].apply(lambda x: int(x.split('_')[0]))
    results_df = results_df[results_df['gen_num'].isin(original_gens)].copy()
    results_df = results_df.sort_values(by='gen_num').reset_index(drop=True)
    log_message(f"Filtered to {len(results_df)} English rows", log_file)

    # -------------------------------------------------------------------------
    # 4. Titles, number of passages, and FactScorer
    # -------------------------------------------------------------------------
    titles = get_titles(dataset_name, file_path)
    if metrics_type == 'factscore':
        fs = FactScorer(anthropic_model_version = anthropic_model_version)

    num_passages = results_df.shape[1] - 2  # subtract 'gen' and 'gen_num' columns
    log_message(f"Number of passages being processed: {num_passages}", log_file)

    # -------------------------------------------------------------------------
    # 5. Load or create nested_results from the output JSON
    # -------------------------------------------------------------------------
    nested_results = {}
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            nested_results = json.load(f)

    # -------------------------------------------------------------------------
    # 6. Main loop over each passage (i.e., each text column)
    # -------------------------------------------------------------------------
    for i in range(num_passages):
        try:
            reference_text = sanitize_text(results_df.iloc[0, i + 1])
            title = titles[reference_text]

            if not reference_text:
                log_message(f"Empty reference text found for passage {i}, skipping", log_file)
                continue

            # If we already evaluated gen=100 for this title, skip
            if title in nested_results:
                # Only check for gen 100 if we didn't create a new title
                if '100' in nested_results[title]:
                    log_message(f"Title '{title}' is already fully evaluated (has gen=100). Skipping.", log_file)
                    continue
                existing_keys = set(nested_results[title].keys())
                local_gens = [g for g in original_gens if str(g) not in existing_keys]
            else:
                nested_results[title] = {}
                local_gens = original_gens
                

            # ---------------------------------------------------------------------
            # 7. Inner loop: find the row(s) for each required gen_val
            # ---------------------------------------------------------------------
            for gen_val in local_gens:
                #print('gen_val: ', gen_val)
                subset_df = results_df[results_df['gen_num'] == gen_val]
                if subset_df.empty:
                    log_message(f"No row found for gen={gen_val} in DataFrame. Skipping.", log_file)
                    continue

                text = sanitize_text(subset_df.iloc[0, i + 1])
                
                if text:
                    print(title)
                    print('\n\n\n\n\n\n')
                    print(text)
                    if metrics_type == 'factscore':
                        factscore_out = fs.get_score([title], [text], knowledge_source=dataset_name)
                        metrics_dict = {
                            "text": text,
                            "factscore": float(factscore_out["score"]) if not pd.isna(factscore_out["score"]) else 0.0,
                            "factscore_init_score": float(factscore_out["init_score"]) if not pd.isna(factscore_out["init_score"]) else 0.0,
                            "factscore_respond_ratio": float(factscore_out["respond_ratio"]) if not pd.isna(factscore_out["respond_ratio"]) else 0.0,
                            "factscore_num_facts_per_response": float(factscore_out["num_facts_per_response"]) if not pd.isna(factscore_out["num_facts_per_response"]) else 0.0,
                        }
                    else:
                        additional_metrics = compute_all_metrics(reference_text, text)
                        metrics_dict = {
                            "text": text,
                            **{k: float(v) if not pd.isna(v) else 0.0 for k, v in additional_metrics.items()}
                        }
                else:
                    metrics_dict = {
                        "text": "",
                        **({"factscore": 0.0, "factscore_init_score": 0.0, "factscore_respond_ratio": 0.0, "factscore_num_facts_per_response": 0.0} 
                            if metrics_type == 'factscore' else 
                            {"BLEU": 0.0, "ROUGE-1": 0.0, "ROUGE-L": 0.0, "METEOR": 0.0, 
                            "BERTScore Precision": 0.0, "BERTScore Recall": 0.0, "BERTScore F1": 0.0, "CHR-F": 0.0})
                    }

                nested_results[title][str(gen_val)] = metrics_dict

                # Incrementally save after each generation
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(nested_results, f, ensure_ascii=False, indent=4)

                log_message(f"Saved passage {i} (title='{title}'), generation {gen_val} to {output_file}", log_file)

        except Exception as e:
            log_message(f"Error processing passage {i}: {str(e)}", log_file)
            continue

    # -------------------------------------------------------------------------
    # 10. Return the final nested dictionary
    # -------------------------------------------------------------------------
    return nested_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process evaluation files and calculate metrics.")
    parser.add_argument("file_path", type=str, help="Path to the file to process.")
    parser.add_argument(
        "--metrics_type", 
        type=str,
        choices=["factscore", "additional"],
        default="factscore",
        help="Choose which metrics to compute: 'factscore' or 'additional'."
    )
    parser.add_argument(
        "--anthropic_model_version", 
        type=str, 
        default="anthropic.claude-3-5-sonnet-20241022-v2:0",
        help="Which Claude model version to use if metrics_type=factscore."
    )

    args = parser.parse_args()

    file_path = args.file_path
    metrics_type = args.metrics_type
    anthropic_model_version = args.anthropic_model_version

    output = process_file_to_nested_dict(
        file_path=file_path,
        metrics_type=metrics_type,
        anthropic_model_version=anthropic_model_version)
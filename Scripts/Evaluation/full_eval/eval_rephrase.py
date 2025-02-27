import argparse
import pandas as pd
import os
import json
from factscorer.factscore.factscorer import FactScorer
import nltk
from nltk.translate.meteor_score import meteor_score
from torchmetrics.text import BLEUScore, ROUGEScore
from nltk.translate.chrf_score import sentence_chrf
from bert_score import score
from transformers import logging
import datetime
import en_core_web_sm

# Load Spacy model
nlp = en_core_web_sm.load()

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

logging.set_verbosity_error()
LOG_DIR = './../../../Results/Evaluation/Logs/'
os.makedirs(LOG_DIR, exist_ok=True)  # Create the logs directory if it doesn't exist

def get_log_filename(input_path):
    """Generate a single log filename for the entire JSON processing."""
    base_name = os.path.basename(input_path).split('.')[0]
    log_file = os.path.join(LOG_DIR, f"{base_name}.log")
    return log_file

def log_message(message, log_file):
    """Log a message with timestamp to the log file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} - {message}\n")

def sanitize_text(text):
    """Sanitize text by replacing double quotes with single quotes and handling NaN."""
    if not isinstance(text, str):
        return ""
    return text.replace('"', "'").strip()

def compute_meteor(reference, candidate):
    return meteor_score([nltk.word_tokenize(reference)], nltk.word_tokenize(candidate))

def compute_bertscore(reference, candidate):
    P, R, F1 = score([candidate], [reference], lang='en')
    return P, R, F1

def compute_all_metrics(reference, candidate):
    if isinstance(candidate, float) or not str(candidate).strip():
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

def get_titles(language):
    knowledge_source_path = f'./../../../Datasets/Knowledge sources/news2024.jsonl'
    results_path = f"./../../../Results/Ablations/Rephrase/"

    if not os.path.exists(knowledge_source_path):
        raise FileNotFoundError(f"Knowledge source file not found: {knowledge_source_path}")
    knowledge_source = pd.read_json(knowledge_source_path, lines=True, encoding="utf-8")

    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results directory not found: {results_path}")
    result_files = [f for f in os.listdir(results_path) if f[0] != '.']
    print(result_files)
    if not result_files:
        raise FileNotFoundError(f"No result files found in: {results_path}")
    results_df = pd.read_json(os.path.join(results_path, result_files[0]), encoding="utf-8").drop(['model_sequence'], axis=0)

    merged = pd.merge(
        left=pd.DataFrame(results_df.iloc[1:3]).T,
        right=knowledge_source,
        left_on='0_en',
        right_on='text',
        how='inner'
    ).drop_duplicates()

    if merged.empty:
        raise ValueError("No matching titles found between knowledge source and results.")

    return merged.title.values if 'title' in merged else []

def process_file_to_nested_dict(
    file_path, 
    metrics_type='factscore',
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

    file_name = os.path.basename(file_path)
    output_dir = f"./../../../Results/Evaluation/Ablations/Rephrase/"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, file_name[:-5] + '_' + metrics_type + '.json')

    original_gens = [0, 2, 4, 6, 8, 10, 13, 16, 20, 25, 30, 40, 50, 65, 80, 100]

    log_file = get_log_filename(file_path)

    try:
        results_df = pd.read_json(file_path, encoding="utf-8").reset_index().rename(columns={'index': 'gen'})
    except Exception as e:
        log_message(f"Error reading JSON file: {str(e)}", log_file)
        return {}

    results_df = results_df.fillna("")

    if 'gen' not in results_df.columns:
        raise KeyError("Column 'gen' not found in results file.")

    results_df = results_df[results_df['gen'].astype(str).str.contains('_en', na=False)].copy()

    results_df['gen_num'] = results_df['gen'].apply(lambda x: int(x.split('_')[0]))
    results_df = results_df[results_df['gen_num'].isin(original_gens)].copy()
    results_df = results_df.sort_values(by='gen_num').reset_index(drop=True)
    log_message(f"Filtered to {len(results_df)} English rows", log_file)

    titles = get_titles("")

    if metrics_type == 'factscore':
        try:
            fs = FactScorer(anthropic_model_version=anthropic_model_version)
        except Exception as e:
            log_message(f"Error initializing FactScorer: {str(e)}", log_file)
            return {}

    num_passages = results_df.shape[1] - 2
    log_message(f"Number of passages being processed: {num_passages}", log_file)

    nested_results = {}
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            nested_results = json.load(f)

    for i in range(num_passages):
        try:
            title = sanitize_text(titles[i] if i < len(titles) else "")
            reference_text = sanitize_text(results_df.iloc[0, i + 1])

            if not reference_text:
                log_message(f"Empty reference text found for passage {i}, skipping", log_file)
                continue

            if title in nested_results and '100' in nested_results[title]:
                log_message(f"Title '{title}' is already fully evaluated (has gen=100). Skipping.", log_file)
                continue

            local_gens = [g for g in original_gens if str(g) not in nested_results.get(title, {})]

            for gen_val in local_gens:
                subset_df = results_df[results_df['gen_num'] == gen_val]
                if subset_df.empty:
                    log_message(f"No row found for gen={gen_val} in DataFrame. Skipping.", log_file)
                    continue

                text = sanitize_text(subset_df.iloc[0, i + 1])

                if metrics_type == 'factscore':
                    factscore_out = fs.get_score([title], [text], knowledge_source="news2024")
                    metrics_dict = {
                        "text": text,
                        "factscore": float(factscore_out["score"] or 0.0),
                        "factscore_init_score": float(factscore_out["init_score"] or 0.0),
                        "factscore_respond_ratio": float(factscore_out["respond_ratio"] or 0.0),
                        "factscore_num_facts_per_response": float(factscore_out["num_facts_per_response"] or 0.0),
                    }
                else:
                    metrics_dict = {
                        "text": text,
                        **compute_all_metrics(reference_text, text)
                    }

                nested_results.setdefault(title, {})[str(gen_val)] = metrics_dict

                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(nested_results, f, ensure_ascii=False, indent=4)

                log_message(f"Saved passage {i} (title='{title}'), generation {gen_val} to {output_file}", log_file)

        except Exception as e:
            log_message(f"Error processing passage {i}: {str(e)}", log_file)
            continue

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

    output = process_file_to_nested_dict(
        file_path=args.file_path,
        metrics_type=args.metrics_type,
        anthropic_model_version=args.anthropic_model_version)

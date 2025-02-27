import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(context='notebook', style='whitegrid', font_scale=1.2)


results_path = "./../../Results/"

json_files = glob.glob(os.path.join(results_path, '**', '*.json'), recursive=True)
json_files = [file for file in json_files if 'news2024' in file or 'booksum' in file or 'scriptbase' in file]

multilingual_files = {
    "en_fr_bmp": {
        'factscore': './../../Results/Evaluation/Bilingual multiplayer/French/translated_passages_news2024_meta-llama_Llama-3.1-8B-Instruct_mistralai_Mistral-7B-Instruct-v0.2_trans_en_fr_en_20250117_232407_factscore.json',
        'additional': './../../Results/Evaluation/Bilingual multiplayer/French/translated_passages_news2024_meta-llama_Llama-3.1-8B-Instruct_mistralai_Mistral-7B-Instruct-v0.2_trans_en_fr_en_20250117_232407_additional.json'
    },
    "en_th_bmp": {
        'factscore': './../../Results/Evaluation/Bilingual multiplayer/Thai/translated_passages_news2024_meta-llama_Llama-3.1-8B-Instruct_mistralai_Mistral-7B-Instruct-v0.2_factscore.json',
        'additional': './../../Results/Evaluation/Bilingual multiplayer/Thai/translated_passages_news2024_meta-llama_Llama-3.1-8B-Instruct_mistralai_Mistral-7B-Instruct-v0.2_additional.json'
    },
    "en_fr_bsl_llama3.1_8b": {
        'factscore': './../../Results/Evaluation/Bilingual self-loop/Factuality/llama3.1_8b/news2024/translated_passages_news2024_Llama-3.1-8B-Instruct_trans_en_fr_en_20250107_155326.json',
        'additional': './../../Results/Evaluation/Bilingual self-loop/Other metrics/llama3.1_8b/news2024/translated_passages_news2024_Llama-3.1-8B-Instruct_trans_en_fr_en_20250107_155326.json'
    },
    "en_th_bsl_llama3.1_8b": {
        'factscore': './../../Results/Evaluation/Bilingual self-loop/Factuality/llama3.1_8b/news2024/translated_passages_news2024_Llama-3.1-8B-Instruct_trans_en_th_en_20250119_042506.json',
        'additional': './../../Results/Evaluation/Bilingual self-loop/Other metrics/llama3.1_8b/news2024/translated_passages_news2024_Llama-3.1-8B-Instruct_trans_en_th_en_20250119_042506.json'
    },
     "en_fr_bsl_mistral7bv2": {
        'factscore': './../../Results/Evaluation/Bilingual self-loop/Factuality/mistral7bv2/news2024/translated_passages_news2024_Mistral-7B-Instruct-v0.2_trans_en_fr_en_20250107_114523.json',
        'additional': './../../Results/Evaluation/Bilingual self-loop/Other metrics/mistral7bv2/news2024/translated_passages_news2024_Mistral-7B-Instruct-v0.2_trans_en_fr_en_20250107_114523.json'
    },
    "en_th_bsl_mistral7bv2": {
        'factscore': './../../Results/Evaluation/Bilingual self-loop/Factuality/mistral7bv2/news2024/translated_passages_news2024_Mistral-7B-Instruct-v0.2_trans_en_th_en_20250107_080922.json',
        'additional': './../../Results/Evaluation/Bilingual self-loop/Other metrics/mistral7bv2/news2024/translated_passages_news2024_Mistral-7B-Instruct-v0.2_trans_en_th_en_20250107_080922.json'
    },
}

metrics = ['ROUGE-1', 'METEOR', 'BLEU', 'BERTScore F1', 'CHR-F', 'factscore']
num_generations = [0, 2, 4, 6, 8, 10, 13, 16, 20, 25, 30, 40, 50, 65, 80, 100]
error_bar_indices = {4, 10, 20, 30, 40, 50, 65, 80, 100}  # Error bars at specific points


def get_average_values(key):
    """ Compute mean & standard deviation for each metric across generations. """
    metric_data_avg = {metric: {} for metric in metrics}
    metric_data_std = {metric: {} for metric in metrics}

    for file in ['factscore', 'additional']:
        df = pd.read_json(multilingual_files[key][file])
        passage_titles = df.columns.tolist()

        for gen_idx in num_generations:
            for metric in metrics:
                values = []

                for title in passage_titles:
                    try:
                        value = df[title][gen_idx][metric]
                        if pd.notna(value):
                            values.append(value)
                    except (KeyError, TypeError, IndexError):
                        continue

                if values:
                    metric_data_avg[metric].setdefault(key, []).append(sum(values) / len(values))
                    metric_data_std[metric].setdefault(key, []).append(pd.Series(values).std())

    return metric_data_avg, metric_data_std


def merge_dicts(dict1, dict2):
    """ Merge mean & std dictionaries, concatenating lists where keys match. """
    merged = {}

    for metric in set(dict1.keys()).union(dict2.keys()):
        merged[metric] = {}

        if metric in dict1:
            for lang, data in dict1[metric].items():
                merged[metric][lang] = data[:]

        if metric in dict2:
            for lang, data in dict2[metric].items():
                merged[metric][lang] = merged[metric].get(lang, []) + data

    return merged


metric_dicts_avg, metric_dicts_std = {}, {}

for key in multilingual_files.keys():
    avg, std = get_average_values(key)

    if not metric_dicts_avg:
        metric_dicts_avg = avg
        metric_dicts_std = std
    else:
        metric_dicts_avg = merge_dicts(metric_dicts_avg, avg)
        metric_dicts_std = merge_dicts(metric_dicts_std, std)

COLOR_SCHEME = {
    'en_fr_bmp': '#1f77b4', 'en_th_bmp': '#2ca02c',
    'en_fr_bsl_llama3.1_8b': '#ff7f0e', 'en_th_bsl_llama3.1_8b': '#8c564b',
    'en_fr_bsl_mistral7bv2': '#d62728', 'en_th_bsl_mistral7bv2': '#e377c2',
    
}
LEGEND_MAP = {
    'en_fr_bmp': 'EN ↔ FR Llama+Mistral', 'en_th_bmp': 'EN ↔ TH Llama+Mistral',
    'en_fr_bsl_llama3.1_8b': 'EN ↔ FR Llama', 'en_th_bsl_llama3.1_8b': 'EN ↔ TH Llama',
    'en_fr_bsl_mistral7bv2': 'EN ↔ FR Mistral', 'en_th_bsl_mistral7bv2': 'EN ↔ TH Mistral'
}

output_path = "./../../Results/Visualizations/BMP"
os.makedirs(output_path, exist_ok=True)
        
fig, axs = plt.subplots(2, 3, figsize=(18, 5), sharex=True)

for idx, metric in enumerate(metrics):
    ax = axs[idx // 3, idx % 3]

    for lang_str in metric_dicts_avg[metric]:
        values = metric_dicts_avg[metric][lang_str]
        std_values = metric_dicts_std[metric][lang_str]

        if len(values) == len(num_generations):
            # Extract error bar values
            filtered_generations = [g for g in num_generations if g in error_bar_indices]
            filtered_values = [values[num_generations.index(g)] for g in filtered_generations]
            filtered_errors = [std_values[num_generations.index(g)] for g in filtered_generations]

            # Plot the main line without error bars
            ax.plot(num_generations, values, marker='o', linestyle='-',
                    label=LEGEND_MAP[lang_str],
                    color=COLOR_SCHEME[lang_str],
                    linewidth=1.5,
                    alpha=0.7)  # Keep the line fully visible

            # Overlay error bars **only** at the selected indices
            ax.errorbar(filtered_generations, filtered_values, yerr=filtered_errors,
                        fmt='o',  # Marker style only (no connecting line)
                        color=COLOR_SCHEME[lang_str],  # Match line color
                        elinewidth=1,  # Thinner error bars
                        capsize=4,      # Small caps on error bars
                        capthick=1,     # Cap thickness
                        alpha=0.6)      # More transparent error bars


    if idx in [3 , 4, 5]:
        ax.set_xlabel('Iteration', fontsize=16, fontweight='bold')
        ax.tick_params(axis='x' , labelsize=14)
        filtered_ticks = [x for x in num_generations if x not in [2, 6, 10 , 16, 25]]
        ax.set_xticks(filtered_ticks)
    else:
        ax.set_xlabel('')
        ax.tick_params(axis='x' , labelsize=14)
    ax.set_ylabel(f'{metric}', fontsize=14, fontweight='bold')

    # Adjust y-axis limits
    if metric != "BERTScore F1":
        ax.set_ylim(bottom=-0.05)
    else:
        ax.set_ylim(bottom=0.7)

    ax.tick_params(axis='y', labelsize=14)
    ax.grid(True)

    # Get handles and labels for the global legend
    handles, labels = axs[0, 0].get_legend_handles_labels()

    # Place the legend only for BLEU
    if idx == 2 and metric == 'BLEU':
        fig.legend(
            handles, labels,
            loc='upper right',
            bbox_to_anchor=(0.992, 0.93),
            ncol=2,
            fontsize=11
        )

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f"{output_path}/BMP_all.pdf", bbox_inches='tight', dpi=300, facecolor='white')
plt.close()
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(context='notebook', style='whitegrid', font_scale=1.2)

results_path = "./../../Results/Evaluation/Ablations/temperature variation/"

json_files = glob.glob(os.path.join(results_path, '**', '*.json'), recursive=True)
json_files = [file for file in json_files if 'news2024' in file or 'booksum' in file or 'scriptbase' in file]

temperature_files = {
    "temperature_0": {
        "factscore": "./../../Results/Evaluation/Ablations/temperature variation/Factuality/translated_passages_news2024_Llama-3.1-8B-Instruct_trans_en_fr_en_20250128_223831_1e-16_42.json",
        "additional": "./../../Results/Evaluation/Ablations/temperature variation/Other metrics/translated_passages_news2024_Llama-3.1-8B-Instruct_trans_en_fr_en_20250128_223831_1e-16_42.json"
    },
    "temperature_0.25": {
        "factscore": "./../../Results/Evaluation/Ablations/temperature variation/Factuality/translated_passages_news2024_Llama-3.1-8B-Instruct_trans_en_fr_en_20250128_223831_0.25_42.json",
        "additional": "./../../Results/Evaluation/Ablations/temperature variation/Other metrics/translated_passages_news2024_Llama-3.1-8B-Instruct_trans_en_fr_en_20250128_223831_0.25_42.json"
    },
    "temperature_0.5": {
        "factscore": "./../../Results/Evaluation/Ablations/temperature variation/Factuality/translated_passages_news2024_Llama-3.1-8B-Instruct_trans_en_fr_en_20250128_223831_0.5_42.json",
        "additional": "./../../Results/Evaluation/Ablations/temperature variation/Other metrics/translated_passages_news2024_Llama-3.1-8B-Instruct_trans_en_fr_en_20250128_223831_0.5_42.json"
    },
    "temperature_0.75": {
        "factscore": "./../../Results/Evaluation/Ablations/temperature variation/Factuality/translated_passages_news2024_Llama-3.1-8B-Instruct_trans_en_fr_en_20250128_223831_0.75_42.json",
        "additional": "./../../Results/Evaluation/Ablations/temperature variation/Other metrics/translated_passages_news2024_Llama-3.1-8B-Instruct_trans_en_fr_en_20250128_223831_0.75_42.json"
    },
    "temperature_1": {
        "factscore": "./../../Results/Evaluation/Ablations/temperature variation/Factuality/translated_passages_news2024_Llama-3.1-8B-Instruct_trans_en_fr_en_20250131_135751_1.0_42.json",
        "additional": "./../../Results/Evaluation/Ablations/temperature variation/Other metrics/translated_passages_news2024_Llama-3.1-8B-Instruct_trans_en_fr_en_20250131_135751_1.0_42.json"
    }
}

metrics = ['ROUGE-1', 'METEOR', 'BLEU', 'BERTScore F1', 'CHR-F', 'factscore']
num_generations = [0, 2, 4, 6, 8, 10, 13, 16, 20, 25, 30, 40, 50, 65, 80, 100]
error_bar_indices = {4, 10, 20, 30, 40, 50, 65, 80, 100}  # Error bars at specific points


def get_average_values(key):
    """ Compute mean & standard deviation for each metric across generations. """
    metric_data_avg = {metric: {} for metric in metrics}
    metric_data_std = {metric: {} for metric in metrics}

    for file in ['factscore', 'additional']:
        df = pd.read_json(temperature_files[key][file])
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


# 🏗️ Compute metrics for all experiments
metric_dicts_avg, metric_dicts_std = {}, {}

for key in temperature_files.keys():
    avg, std = get_average_values(key)

    if not metric_dicts_avg:
        metric_dicts_avg = avg
        metric_dicts_std = std
    else:
        for metric in metrics:
            for temp in avg[metric]:
                metric_dicts_avg[metric].setdefault(temp, []).extend(avg[metric][temp])
                metric_dicts_std[metric].setdefault(temp, []).extend(std[metric][temp])

COLOR_SCHEME = {
    'temperature_0': '#1f77b4', 'temperature_0.25': '#2ca02c', 'temperature_0.5': '#ff7f0e',
    'temperature_0.75': '#d62728', 'temperature_1': '#9467bd',
}
LEGEND_MAP = {
    'temperature_0': 'Temperature = 1e-6', 'temperature_0.25': 'Temperature = 0.25',
    'temperature_0.5': 'Temperature = 0.5', 'temperature_0.75': 'Temperature = 0.75',
    'temperature_1': 'Temperature = 1'
}

output_path = "./../../Results/Visualizations/Temperature_ablation"
os.makedirs(output_path, exist_ok=True)

fig, axs = plt.subplots(2, 3, figsize=(18, 4.5))

for idx, metric in enumerate(metrics):
    ax = axs[idx // 3, idx % 3]

    for temp in metric_dicts_avg[metric]:
        values = metric_dicts_avg[metric][temp]
        std_values = metric_dicts_std[metric][temp]

        if len(values) == len(num_generations):
            # Extract error bar values
            filtered_generations = [g for g in num_generations if g in error_bar_indices]
            filtered_values = [values[num_generations.index(g)] for g in filtered_generations]
            filtered_errors = [std_values[num_generations.index(g)] for g in filtered_generations]

            ax.plot(num_generations, values, marker='o',
                    label=LEGEND_MAP[temp],
                    color=COLOR_SCHEME[temp],
                    linewidth=2, alpha=0.7)

            ax.errorbar(filtered_generations, filtered_values, yerr=filtered_errors,
                        fmt='o', color=COLOR_SCHEME[temp], elinewidth=1,
                        capsize=4, capthick=1, alpha=0.6)

    ax.set_ylabel(metric, fontsize=14, fontweight='bold')
    ax.grid(True)

    if idx == 2 and metric == 'BLEU':
        fig.legend(
            *ax.get_legend_handles_labels(),
            loc='upper right',
            bbox_to_anchor=(0.989, 0.92),
            ncol=2,
            fontsize=11
        )

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f"{output_path}/Temperature_ablation_all.pdf", bbox_inches='tight', dpi=300, facecolor='white')
plt.close()
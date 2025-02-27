import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(context='notebook', style='whitegrid', font_scale=1.2)

results_path = "./../../Results/Evaluation/Multilingual multiplayer"


json_files = glob.glob(os.path.join(results_path, '**', '*.json'), recursive=True)
json_files = [file for file in json_files if 'news2024' in file or 'booksum' in file or 'scriptbase' in file]


multilingual_files = {
    "setting1": {
        'factscore': './../../Results/Evaluation/Multilingual multiplayer/setting1/translated_passages_news2024_llama3.1_8b_mistral7bv2_trans_th_fr_20250121_115359_1_factscore.json',
        'additional': './../../Results/Evaluation/Multilingual multiplayer/setting1/translated_passages_news2024_llama3.1_8b_mistral7bv2_trans_th_fr_20250121_115359_1_additional.json'
    },
    "setting2": {
        'factscore': './../../Results/Evaluation/Multilingual multiplayer/setting2/translated_passages_news2024_mistral7bv2_llama3.1_8b_gemma2_9b_it_trans_th_fr_20250124_164248_3_factscore.json',
        'additional': './../../Results/Evaluation/Multilingual multiplayer/setting2/translated_passages_news2024_mistral7bv2_llama3.1_8b_gemma2_9b_it_trans_th_fr_20250124_164248_3_additional.json'
    },
    "setting3": {
        'factscore': './../../Results/Evaluation/Multilingual multiplayer/setting3/translated_passages_news2024_mistral7bv2_llama3.1_8b_trans_fr_th_zh_nl_20250123_074916_2_factscore.json',
        'additional': './../../Results/Evaluation/Multilingual multiplayer/setting3/translated_passages_news2024_mistral7bv2_llama3.1_8b_trans_fr_th_zh_nl_20250123_074916_2_additional.json'
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
    'setting1': '#1f77b4', 'setting2': '#2ca02c', 'setting3': '#ff7f0e',
}
LEGEND_MAP = {
    'setting1': 'Setting 1', 'setting2': 'Setting 2', 'setting3': 'Setting 3'
}


output_path = "./../../Results/Visualizations/MM"
os.makedirs(output_path, exist_ok=True)

fig, axs = plt.subplots(2, 3, figsize=(18, 4.5))

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

            ax.plot(num_generations, values, marker='o',
                    label=LEGEND_MAP[lang_str],
                    color=COLOR_SCHEME[lang_str],
                    linewidth=2, alpha=0.7)

            ax.errorbar(filtered_generations, filtered_values, yerr=filtered_errors,
                        fmt='o', color=COLOR_SCHEME[lang_str], elinewidth=1,
                        capsize=4, capthick=1, alpha=0.6)

    ax.set_ylabel(metric, fontsize=14, fontweight='bold')
    ax.grid(True)

    if idx == 2 and metric == 'BLEU':
        fig.legend(
            *ax.get_legend_handles_labels(),
            loc='upper right',
            bbox_to_anchor=(0.994, 0.93),
            ncol=1,
            fontsize=16
        )
    filtered_ticks = [x for x in num_generations if x not in [2, 6, 10 , 16, 25]]
    ax.set_xticks(filtered_ticks)
    ax.set_xlabel('Iteration', fontsize=16)
    ax.tick_params(axis='x' , labelsize=14)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f"{output_path}/MM_all.pdf", bbox_inches='tight', dpi=300, facecolor='white')
plt.close()
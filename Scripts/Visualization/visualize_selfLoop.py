import os
import re
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# Define language order, colors, and legend labels
LANGUAGE_ORDER = [
    # 'en_ar',  # **English → Arabic → English**
    'en_zh',  # **English → Chinese → English**
    'en_nl',  # **English → Dutch → English**
    'en_fr',  # **English → French → English**
    'en_de',  # **English → German → English**
    # 'en_hi',  # **English → Hindi → English**
    'en_th',  # **English → Thai → English**
    'en_vn',  # **English → Vietnamese → English**
]

COLOR_SCHEME = {
    # 'en_ar': '#9467bd',  # purple
    'en_zh': '#1f77b4',  # blue
    'en_nl': '#2ca02c',  # green
    'en_fr': '#ff7f0e',  # orange
    'en_de': '#8c564b',  # brown
    # 'en_hi': '#7f7f7f',  # gray
    'en_th': '#d62728',  # red
    'en_vn': '#e377c2',  # pink
}

LANG_LEGEND_MAP = {
    # 'en_ar': 'EN ↔ AR',
    'en_zh': 'EN ↔ ZH',
    'en_nl': 'EN ↔ NL',
    'en_fr': 'EN ↔ FR',
    'en_de': 'EN ↔ DE',
    # 'en_hi': 'EN ↔ HI',
    'en_th': 'EN ↔ TH',
    'en_vn': 'EN ↔ VN',
}


def get_json_files(base_path, max_depth):
    """
    Get all .json files in a directory and its subdirectories up to max_depth levels.
    """
    json_files = []
    for depth in range(max_depth + 1):
        search_pattern = os.path.join(base_path, *['*'] * depth, '*.json')
        json_files.extend(glob.glob(search_pattern))
    return json_files


def extract_info_from_path(path):
    """
    Extract components from the file path.
    """
    parts = path.split('/')
    # Get basic components (adjust indices as needed)
    setting = parts[5]
    metrics_type = parts[6]
    model = parts[7]
    dataset = parts[8]
    # Extract language abbreviations from the filename.
    filename = parts[-1]
    languages = [i for i in filename.split('_') if i in ['fr', 'th', 'vn', 'nl', 'de', 'zh']]
    # Ensure 'en' comes first if present.
    languages = list(set(languages))
    languages = ['en'] + [i for i in languages if i != 'en']
    return {
        'setting': setting,
        'metrics_type': metrics_type,
        'dataset': dataset,
        'model': model,
        'languages': languages
    }


def create_metric_visualizations(base_path, dataset, model):
    """
    Create one plot per metric (using a 2x3 grid), with one line per language combination.
    Each line is drawn with a marker for every point and connected with a line.
    The x-axis ticks now show one tick per generation value (i.e. no filtering).
    """
    # Set the Seaborn theme (whitegrid maintains the grid background)
    sns.set_theme(context='notebook', style='whitegrid', font_scale=1.2)

    # Metrics to visualize and the generation indices.
    metrics = ['ROUGE-1', 'METEOR', 'BLEU', 'BERTScore F1', 'CHR-F', 'factscore']
    num_generations = [0, 2, 4, 6, 8, 10, 13, 16, 20, 25, 30, 40, 50, 65, 80, 100]

    # Get JSON files from two folders.
    json_files = [
        os.path.join('Other metrics', model, dataset, f)
        for f in os.listdir(os.path.join(base_path, f'Other metrics/{model}/{dataset}'))
        if f.endswith('.json')
    ]
    json_files += [
        os.path.join('Factuality', model, dataset, f)
        for f in os.listdir(os.path.join(base_path, f'Factuality/{model}/{dataset}'))
        if f.endswith('.json')
    ]

    # Prepare a dictionary to store data for each metric by language.
    metric_data_avg = {metric: {} for metric in metrics}
    metric_data_std = {metric: {} for metric in metrics}

    # Extract info from the first JSON file (for output path components)
    info = extract_info_from_path(os.path.join(base_path, json_files[0]))
    dataset_name = dataset
    setting = info['setting']

    # Process each file (each language combination)
    for json_file in json_files:
        print(f"Processing {json_file}")
        info = extract_info_from_path(os.path.join(base_path, json_file))
        # Create a language string (e.g., "en_th") using the non-English language code.
        non_en_langs = [lang for lang in info['languages'] if lang != 'en']
        lang_str = f"en_{non_en_langs[0]}" if non_en_langs else "en"
        print(f"Processing language combination: {lang_str}")

        # Load the JSON file as a DataFrame.
        df = pd.read_json(os.path.join(base_path, json_file))
        passage_titles = df.columns.tolist()

        # For each generation, compute the average value for each metric.
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
                    if lang_str not in metric_data_avg[metric]:
                        metric_data_avg[metric][lang_str] = []
                    metric_data_avg[metric][lang_str].append(sum(values) / len(values))
                    if lang_str not in metric_data_std[metric]:
                        metric_data_std[metric][lang_str] = []
                    metric_data_std[metric][lang_str].append(np.std(values))

    # Create the output directory.
    output_path = os.path.join("./../../Results/Visualizations", setting, model, dataset_name)
    print(f"Saving output to: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    # Create a 2x3 grid of subplots.
    fig, axs = plt.subplots(2, 3, figsize=(18, 6), sharex=True)

    # Plot each metric in its subplot.
    for idx, metric in enumerate(metrics):
        ax = axs[idx // 3, idx % 3]
        # Plot data for each language (if available).
        for lang in LANGUAGE_ORDER:
            if lang in metric_data_avg[metric]:
                values = metric_data_avg[metric][lang]
                if len(values) == len(num_generations):
                    # ax.plot(num_generations, values, marker='o', linestyle='-',
                    #         label=LANG_LEGEND_MAP[lang],
                    #         color=COLOR_SCHEME[lang],
                    #         linewidth=2,
                    #         alpha=0.8)
                # plot the line with error bars
                    # ax.errorbar(num_generations, values, yerr=metric_data_std[metric][lang],
                    #     label=LANG_LEGEND_MAP[lang],
                    #     color=COLOR_SCHEME[lang],
                    #     linewidth=2,  # Line thickness
                    #     alpha=0.8,    # Full opacity for the main line
                    #     elinewidth=1, # Thinner error bars
                    #     capsize=4,     # Small caps on error bars
                    #     capthick=1,   # Cap thickness
                    #     ecolor=COLOR_SCHEME[lang])    # Lower transparency for error bars

                    # Define the indices where error bars should be displayed
                    error_bar_indices = {4, 10, 20, 30, 40, 50, 65, 80, 100}

                    # Extract only the values for the error bars at specific indices
                    filtered_generations = [g for g in num_generations if g in error_bar_indices]
                    filtered_values = [values[num_generations.index(g)] for g in filtered_generations]
                    filtered_errors = [metric_data_std[metric][lang][num_generations.index(g)] for g in filtered_generations]

                    # Plot the main line without error bars
                    ax.plot(num_generations, values, marker='o', linestyle='-',
                            label=LANG_LEGEND_MAP[lang],
                            color=COLOR_SCHEME[lang],
                            linewidth=1.5,
                            alpha=0.7)  # Keep the line fully visible

                    # Overlay error bars **only** at the selected indices
                    ax.errorbar(filtered_generations, filtered_values, yerr=filtered_errors,
                                fmt='o',  # Marker style only (no connecting line)
                                color=COLOR_SCHEME[lang],  # Match line color
                                elinewidth=1,  # Thinner error bars
                                capsize=4,      # Small caps on error bars
                                capthick=1,     # Cap thickness
                                alpha=0.6)      # More transparent error bars


                                        
        # ax.set_title(metric, fontsize=20, fontweight='bold')
        ax.grid(True)
        # Set x-axis ticks to show one tick per generation value (no filtering)
        # ax.set_xticks(num_generations)
        # ax.set_xticklabels(num_generations, fontsize=14, rotation=45, ha='right')
        ax.set_ylabel(metric, fontsize=20, fontweight='bold')
        ax.tick_params(axis='y', labelsize=14)
        # if metric != "BERTScore F1":
        #     ax.set_ylim(bottom=-0.05)
        # else:
        #     ax.set_ylim(bottom=0.7)
        # Bottom row: keep x-axis labels and ticks
        if idx in [3 , 4, 5]:
            ax.set_xlabel('Iteration', fontsize=16, fontweight='bold')
            ax.tick_params(axis='x' , labelsize=14)
            filtered_ticks = [x for x in num_generations if x not in [2, 6, 10 , 16, 25]]
            ax.set_xticks(filtered_ticks)
        else:
            ax.set_xlabel('')
            ax.tick_params(axis='x' , labelsize=14)


    # Deduplicate legend entries across all subplots and create a single figure-level legend.
    handles_all, labels_all = [], []
    for ax in axs.flat:
        h, l = ax.get_legend_handles_labels()
        handles_all.extend(h)
        labels_all.extend(l)
    by_label = {}
    for handle, label in zip(handles_all, labels_all):
        if label not in by_label:
            by_label[label] = handle

    # # Bottom row: keep x-axis labels and ticks
    # filtered_ticks = [x for x in num_generations if x not in [2, 6, 10 , 16, 25]]
    # ax.set_xticks(filtered_ticks)
    # ax.set_xlabel('Iteration', fontsize=16, fontweight='bold')
    # ax.tick_params(axis='x' , labelsize=14)

    fig.legend(list(by_label.values()), list(by_label.keys()),
               loc='upper right', bbox_to_anchor=(0.99, 0.92), ncol=2, fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_file = os.path.join(output_path, f"{model}_{dataset}.pdf")
    plt.savefig(output_file, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()

# Execute for each model and dataset.
models = ['llama3.1_8b' , 'mistral7bv2']
datasets = ['news2024', 'booksum', 'scriptbase']

base_path = './../../Results/Evaluation/Bilingual self-loop/'
for model in models:
    for dataset in datasets:
        create_metric_visualizations(base_path, dataset, model)

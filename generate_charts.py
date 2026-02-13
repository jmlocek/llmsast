"""
Skrypt do generowania wykresów metryk ewaluacyjnych dla pracy magisterskiej.
Autor: [Twoje imię i nazwisko]
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# Ustawienia dla profesjonalnych wykresów do pracy magisterskiej
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True,
})


def load_results(results_dir: str, exclude_ragonhunter: bool = False, exclude_all_rag: bool = False) -> dict:
    """
    Wczytuje wszystkie pliki JSON z wynikami z folderu results.
    
    Args:
        results_dir: Ścieżka do folderu z wynikami
        exclude_ragonhunter: Czy pominąć konfiguracje RAG (hunter)
        exclude_all_rag: Czy pominąć wszystkie konfiguracje z RAG
        
    Returns:
        Słownik z wynikami dla każdego modelu i konfiguracji
    """
    results = {}
    results_path = Path(results_dir)
    
    # Definicja struktury wyników
    result_files = {
        # GPT-OSS-20B
        'GPT-OSS-20B': {
            'Jeden agent': 'gpt-oss-20b/diverse_vuln/diversevul_single_agent_progress.json',
            'Jeden agent + RAG': 'gpt-oss-20b/diverse_vuln/diversevul_single_agent_rag_progress.json',
            'Wiele agentów': 'gpt-oss-20b/diverse_vuln/diversevul_multi_agent_progress.json',
            'Wiele agentów + RAG': 'gpt-oss-20b/diverse_vuln/diversevul_multi_agent_rag_progress.json',
            'Wiele agentów + RAG (hunter)': 'gpt-oss-20b/diverse_vuln/diversevul_multi_agent_ragonhunter_progress.json',
        },
        # GPT-OSS-120B
        'GPT-OSS-120B': {
            'Jeden agent': 'gpt-oss-120b/diversevul_single_agent_progress.json',
            'Jeden agent + RAG': 'gpt-oss-120b/diversevul_single_agent_rag_progress.json',
            'Wiele agentów': 'gpt-oss-120b/diversevul_multi_agent_progress.json',
            'Wiele agentów + RAG': 'gpt-oss-120b/diversevul_multi_agent_rag_progress.json',
            'Wiele agentów + RAG (hunter)': 'gpt-oss-120b/diversevul_multi_agent_ragonhunter_progress.json',
        },
        # Granite
        'Granite': {
            'Jeden agent': 'granite/diversevul_single_agent_progress.json',
            'Jeden agent + RAG': 'granite/diversevul_single_agent_rag_progress .json',
            'Wiele agentów': 'granite/diversevul_multi_agent_progress.json',
            'Wiele agentów + RAG': 'granite/diversevul_multi_agent_rag_progress.json',
            'Wiele agentów + RAG (hunter)': 'granite/diversevul_multi_agent_ragonhunter_progress.json',
        },
    }
    
    for model_name, configs in result_files.items():
        results[model_name] = {}
        for config_name, file_path in configs.items():
            # Pomiń wszystkie konfiguracje z RAG jeśli flaga jest ustawiona
            if exclude_all_rag and 'RAG' in config_name:
                continue
            # Pomiń konfiguracje RAG (hunter) jeśli flaga jest ustawiona
            if exclude_ragonhunter and 'RAG (hunter)' in config_name:
                continue
            full_path = results_path / file_path
            if full_path.exists():
                with open(full_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results[model_name][config_name] = data
            else:
                print(f"Ostrzeżenie: Plik {full_path} nie istnieje")
    
    return results


def calculate_metrics(data: dict) -> dict:
    """
    Oblicza metryki na podstawie macierzy pomyłek.
    
    Args:
        data: Słownik z tp, fp, tn, fn
        
    Returns:
        Słownik z obliczonymi metrykami
    """
    tp = data.get('tp', 0)
    fp = data.get('fp', 0)
    tn = data.get('tn', 0)
    fn = data.get('fn', 0)
    
    # Precyzja (Precision)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Czułość (Recall/Sensitivity)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Wynik F1 (F1-Score)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Wynik F2 (F2-Score) - większa waga dla czułości
    f2 = 5 * (precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0
    
    # Dokładność (Accuracy)
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    # Swoistość (Specificity)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'Precyzja': precision,
        'Czułość': recall,
        'Wynik F1': f1,
        'Wynik F2': f2,
        'Dokładność': accuracy,
        'Swoistość': specificity,
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn,
    }


def create_comparison_bar_chart(results: dict, metrics: list, output_path: str, title: str):
    """
    Tworzy wykres słupkowy porównujący metryki dla różnych konfiguracji.
    
    Args:
        results: Słownik z wynikami
        metrics: Lista metryk do wyświetlenia
        output_path: Ścieżka do zapisu wykresu
        title: Tytuł wykresu
    """
    # Przygotowanie danych
    all_configs = []
    all_values = {metric: [] for metric in metrics}
    
    for model_name, configs in results.items():
        for config_name, data in configs.items():
            if data:  # Sprawdź czy dane istnieją
                metrics_data = calculate_metrics(data)
                label = f"{model_name}\n{config_name}"
                all_configs.append(label)
                for metric in metrics:
                    all_values[metric].append(metrics_data[metric])
    
    if not all_configs:
        print("Brak danych do utworzenia wykresu")
        return
    
    # Tworzenie wykresu
    x = np.arange(len(all_configs))
    width = 0.2
    multiplier = 0
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Kolory profesjonalne
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for metric, color in zip(metrics, colors):
        offset = width * multiplier
        values = all_values[metric]
        rects = ax.bar(x + offset, values, width, label=metric, color=color, edgecolor='black', linewidth=0.5)
        
        # Dodanie wartości na słupkach
        for rect, value in zip(rects, values):
            height = rect.get_height()
            ax.annotate(f'{value:.2f}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, rotation=0)
        
        multiplier += 1
    
    ax.set_ylabel('Wartość metryki')
    ax.set_title(title)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(all_configs, rotation=45, ha='right')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    
    plt.tight_layout()
    plt.savefig(output_path, format='png', bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Zapisano wykres: {output_path}")


def create_model_comparison_chart(results: dict, output_dir: str):
    """
    Tworzy pojedyncze wykresy dla każdej metryki, porównując wszystkie modele.
    
    Args:
        results: Słownik z wynikami
        output_dir: Folder do zapisu wykresów
    """
    metrics_pl = ['Precyzja', 'Czułość', 'Wynik F1', 'Wynik F2']
    
    for metric in metrics_pl:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Przygotowanie danych dla każdego modelu
        models = list(results.keys())
        all_configs = set()
        for model_name, configs in results.items():
            all_configs.update(configs.keys())
        all_configs = sorted(list(all_configs))
        
        x = np.arange(len(all_configs))
        width = 0.25
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        for i, (model_name, color) in enumerate(zip(models, colors)):
            values = []
            for config in all_configs:
                if config in results[model_name] and results[model_name][config]:
                    metrics_data = calculate_metrics(results[model_name][config])
                    values.append(metrics_data[metric])
                else:
                    values.append(0)
            
            offset = width * i
            rects = ax.bar(x + offset, values, width, label=model_name, color=color, 
                          edgecolor='black', linewidth=0.5)
            
            # Dodanie wartości na słupkach
            for rect, value in zip(rects, values):
                if value > 0:
                    height = rect.get_height()
                    ax.annotate(f'{value:.2f}',
                               xy=(rect.get_x() + rect.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=8)
        
        ax.set_ylabel('Wartość metryki')
        ax.set_title(f'Porównanie modeli - {metric}')
        ax.set_xticks(x + width)
        ax.set_xticklabels(all_configs, rotation=45, ha='right')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'porownanie_{metric.lower().replace(" ", "_")}.png')
        plt.savefig(output_path, format='png', bbox_inches='tight')
        plt.savefig(output_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
        plt.close()
        
        print(f"Zapisano wykres: {output_path}")


def create_radar_chart(results: dict, output_dir: str):
    """
    Tworzy wykres radarowy porównujący wszystkie metryki.
    
    Args:
        results: Słownik z wynikami
        output_dir: Folder do zapisu wykresów
    """
    metrics = ['Precyzja', 'Czułość', 'Wynik F1', 'Wynik F2']
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Zamknięcie wykresu
    
    # Policzenie liczby konfiguracji
    total_configs = sum(1 for configs in results.values() for data in configs.values() if data)
    colors = plt.cm.tab20(np.linspace(0, 1, max(total_configs, 1)))
    color_idx = 0
    
    for model_name, configs in results.items():
        for config_name, data in configs.items():
            if data:
                metrics_data = calculate_metrics(data)
                values = [metrics_data[m] for m in metrics]
                values += values[:1]  # Zamknięcie wykresu
                
                label = f"{model_name} - {config_name}"
                ax.plot(angles, values, 'o-', linewidth=2, label=label, color=colors[color_idx])
                ax.fill(angles, values, alpha=0.1, color=colors[color_idx])
                color_idx += 1
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
    ax.set_title('Porównanie metryk - wykres radarowy', y=1.08)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'wykres_radarowy.png')
    plt.savefig(output_path, format='png', bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Zapisano wykres: {output_path}")


def create_heatmap(results: dict, output_dir: str):
    """
    Tworzy mapę cieplną z wszystkimi metrykami.
    
    Args:
        results: Słownik z wynikami
        output_dir: Folder do zapisu wykresów
    """
    metrics = ['Precyzja', 'Czułość', 'Wynik F1', 'Wynik F2']
    
    # Przygotowanie danych
    labels = []
    data_matrix = []
    
    for model_name, configs in results.items():
        for config_name, data in configs.items():
            if data:
                metrics_data = calculate_metrics(data)
                labels.append(f"{model_name}\n{config_name}")
                data_matrix.append([metrics_data[m] for m in metrics])
    
    if not data_matrix:
        print("Brak danych do utworzenia mapy cieplnej")
        return
    
    data_matrix = np.array(data_matrix)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Ustawienia osi
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(metrics)
    ax.set_yticklabels(labels)
    
    # Rotacja etykiet osi X
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Dodanie wartości do komórek
    for i in range(len(labels)):
        for j in range(len(metrics)):
            value = data_matrix[i, j]
            text_color = 'white' if value < 0.4 or value > 0.8 else 'black'
            text = ax.text(j, i, f'{value:.2f}',
                          ha="center", va="center", color=text_color, fontsize=10)
    
    ax.set_title('Mapa cieplna metryk ewaluacyjnych')
    
    # Pasek kolorów
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Wartość metryki', rotation=-90, va="bottom")
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'mapa_cieplna.png')
    plt.savefig(output_path, format='png', bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Zapisano wykres: {output_path}")


def create_grouped_metrics_chart(results: dict, output_dir: str):
    """
    Tworzy zgrupowany wykres słupkowy dla głównych metryk.
    
    Args:
        results: Słownik z wynikami
        output_dir: Folder do zapisu wykresów
    """
    metrics = ['Precyzja', 'Czułość', 'Wynik F1', 'Wynik F2']
    
    # Przygotowanie danych
    labels = []
    data = {metric: [] for metric in metrics}
    
    for model_name, configs in results.items():
        for config_name, raw_data in configs.items():
            if raw_data:
                metrics_data = calculate_metrics(raw_data)
                labels.append(f"{model_name}\n{config_name}")
                for metric in metrics:
                    data[metric].append(metrics_data[metric])
    
    if not labels:
        print("Brak danych do utworzenia wykresu")
        return
    
    x = np.arange(len(labels))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(16, 9))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        offset = (i - 1.5) * width
        rects = ax.bar(x + offset, data[metric], width, label=metric, color=color,
                      edgecolor='black', linewidth=0.5)
        
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                           xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=7, rotation=90)
    
    ax.set_ylabel('Wartość metryki')
    ax.set_title('Porównanie metryk ewaluacyjnych dla wszystkich konfiguracji')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, 1.2)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    
    # Dodanie linii siatki
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'metryki_wszystkie_konfiguracje.png')
    plt.savefig(output_path, format='png', bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Zapisano wykres: {output_path}")


def create_f_scores_comparison(results: dict, output_dir: str):
    """
    Tworzy wykres porównujący wyniki F1 i F2.
    
    Args:
        results: Słownik z wynikami
        output_dir: Folder do zapisu wykresów
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    labels = []
    f1_scores = []
    f2_scores = []
    
    for model_name, configs in results.items():
        for config_name, data in configs.items():
            if data:
                metrics_data = calculate_metrics(data)
                labels.append(f"{model_name}\n{config_name}")
                f1_scores.append(metrics_data['Wynik F1'])
                f2_scores.append(metrics_data['Wynik F2'])
    
    x = np.arange(len(labels))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, f1_scores, width, label='Wynik F1', color='#2E86AB', 
                   edgecolor='black', linewidth=0.5)
    rects2 = ax.bar(x + width/2, f2_scores, width, label='Wynik F2', color='#F18F01',
                   edgecolor='black', linewidth=0.5)
    
    # Dodanie wartości
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                           xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Wartość metryki')
    ax.set_title('Porównanie wyników F1 i F2 dla wszystkich konfiguracji')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'porownanie_f1_f2.png')
    plt.savefig(output_path, format='png', bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Zapisano wykres: {output_path}")


def create_precision_recall_chart(results: dict, output_dir: str):
    """
    Tworzy wykres porównujący precyzję i czułość.
    
    Args:
        results: Słownik z wynikami
        output_dir: Folder do zapisu wykresów
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    labels = []
    precision_scores = []
    recall_scores = []
    
    for model_name, configs in results.items():
        for config_name, data in configs.items():
            if data:
                metrics_data = calculate_metrics(data)
                labels.append(f"{model_name}\n{config_name}")
                precision_scores.append(metrics_data['Precyzja'])
                recall_scores.append(metrics_data['Czułość'])
    
    x = np.arange(len(labels))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, precision_scores, width, label='Precyzja', color='#2ca02c',
                   edgecolor='black', linewidth=0.5)
    rects2 = ax.bar(x + width/2, recall_scores, width, label='Czułość', color='#d62728',
                   edgecolor='black', linewidth=0.5)
    
    # Dodanie wartości
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                           xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Wartość metryki')
    ax.set_title('Porównanie precyzji i czułości dla wszystkich konfiguracji')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'porownanie_precyzja_czulosc.png')
    plt.savefig(output_path, format='png', bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Zapisano wykres: {output_path}")


def generate_summary_table(results: dict, output_dir: str):
    """
    Generuje tabelę podsumowującą wszystkie wyniki.
    
    Args:
        results: Słownik z wynikami
        output_dir: Folder do zapisu
    """
    output_path = os.path.join(output_dir, 'tabela_wynikow.txt')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 120 + "\n")
        f.write("TABELA WYNIKÓW EWALUACJI\n")
        f.write("=" * 120 + "\n\n")
        
        header = f"{'Model':<20} {'Konfiguracja':<30} {'Precyzja':>10} {'Czułość':>10} {'F1':>10} {'F2':>10} {'TP':>8} {'FP':>8} {'TN':>8} {'FN':>8}\n"
        f.write(header)
        f.write("-" * 120 + "\n")
        
        for model_name, configs in results.items():
            for config_name, data in configs.items():
                if data:
                    metrics = calculate_metrics(data)
                    row = f"{model_name:<20} {config_name:<30} {metrics['Precyzja']:>10.4f} {metrics['Czułość']:>10.4f} {metrics['Wynik F1']:>10.4f} {metrics['Wynik F2']:>10.4f} {metrics['TP']:>8} {metrics['FP']:>8} {metrics['TN']:>8} {metrics['FN']:>8}\n"
                    f.write(row)
        
        f.write("=" * 120 + "\n")
    
    print(f"Zapisano tabelę: {output_path}")
    
    # Generowanie również w formacie LaTeX
    latex_path = os.path.join(output_dir, 'tabela_wynikow.tex')
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Wyniki ewaluacji modeli}\n")
        f.write("\\label{tab:wyniki}\n")
        f.write("\\begin{tabular}{llcccc}\n")
        f.write("\\toprule\n")
        f.write("Model & Konfiguracja & Precyzja & Czułość & F1 & F2 \\\\\n")
        f.write("\\midrule\n")
        
        for model_name, configs in results.items():
            for config_name, data in configs.items():
                if data:
                    metrics = calculate_metrics(data)
                    f.write(f"{model_name} & {config_name} & {metrics['Precyzja']:.4f} & {metrics['Czułość']:.4f} & {metrics['Wynik F1']:.4f} & {metrics['Wynik F2']:.4f} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"Zapisano tabelę LaTeX: {latex_path}")


def generate_all_charts(results: dict, charts_dir: str, metrics: list):
    """Generuje wszystkie wykresy dla podanych wyników."""
    # 1. Główny wykres porównawczy
    create_comparison_bar_chart(
        results, 
        metrics, 
        os.path.join(charts_dir, 'porownanie_glowne.png'),
        'Porównanie metryk ewaluacyjnych dla wszystkich modeli i konfiguracji'
    )
    
    # 2. Wykresy dla poszczególnych modeli
    create_model_comparison_chart(results, charts_dir)
    
    # 3. Wykres radarowy
    create_radar_chart(results, charts_dir)
    
    # 4. Mapa cieplna
    create_heatmap(results, charts_dir)
    
    # 5. Zgrupowany wykres metryk
    create_grouped_metrics_chart(results, charts_dir)
    
    # 6. Porównanie F1 i F2
    create_f_scores_comparison(results, charts_dir)
    
    # 7. Porównanie precyzji i czułości
    create_precision_recall_chart(results, charts_dir)
    
    # 8. Tabela podsumowująca
    generate_summary_table(results, charts_dir)


def main():
    """Główna funkcja skryptu."""
    # Parser argumentów
    parser = argparse.ArgumentParser(description='Generowanie wykresów metryk ewaluacyjnych')
    parser.add_argument('--no-ragonhunter', action='store_true', 
                        help='Generuj wykresy bez konfiguracji RAG (hunter)')
    parser.add_argument('--no-rag', action='store_true',
                        help='Generuj wykresy bez wszystkich konfiguracji RAG')
    parser.add_argument('--all', action='store_true',
                        help='Generuj wszystkie wersje wykresów (pełna, bez RAG hunter, bez RAG)')
    args = parser.parse_args()
    
    # Ścieżki
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    charts_dir = os.path.join(script_dir, 'charts')
    
    # Utworzenie folderu na wykresy
    os.makedirs(charts_dir, exist_ok=True)
    
    print("=" * 60)
    print("GENEROWANIE WYKRESÓW DO PRACY MAGISTERSKIEJ")
    print("=" * 60)
    print()
    
    metrics = ['Precyzja', 'Czułość', 'Wynik F1', 'Wynik F2']
    
    # Wczytanie wyników
    print("Wczytywanie wyników...")
    
    if args.all:
        # Generuj wszystkie wersje
        print("\n--- Wersja pełna (z RAG hunter) ---")
        results_full = load_results(results_dir, exclude_ragonhunter=False, exclude_all_rag=False)
        print("\nGenerowanie wykresów...")
        print("-" * 40)
        generate_all_charts(results_full, charts_dir, metrics)
        print(f"\nWykresy zapisane w: {charts_dir}")
        
        print("\n--- Wersja bez RAG hunter ---")
        charts_dir_no_hunter = os.path.join(script_dir, 'charts', 'bez_ragonhunter')
        os.makedirs(charts_dir_no_hunter, exist_ok=True)
        results_no_hunter = load_results(results_dir, exclude_ragonhunter=True, exclude_all_rag=False)
        print("\nGenerowanie wykresów...")
        print("-" * 40)
        generate_all_charts(results_no_hunter, charts_dir_no_hunter, metrics)
        print(f"\nWykresy zapisane w: {charts_dir_no_hunter}")
        
        print("\n--- Wersja bez RAG (tylko podstawowe konfiguracje) ---")
        charts_dir_no_rag = os.path.join(script_dir, 'charts', 'bez_rag')
        os.makedirs(charts_dir_no_rag, exist_ok=True)
        results_no_rag = load_results(results_dir, exclude_ragonhunter=False, exclude_all_rag=True)
        print("\nGenerowanie wykresów...")
        print("-" * 40)
        generate_all_charts(results_no_rag, charts_dir_no_rag, metrics)
        print(f"\nWykresy zapisane w: {charts_dir_no_rag}")
    elif args.no_rag:
        # Tylko wersja bez RAG
        charts_dir = os.path.join(script_dir, 'charts', 'bez_rag')
        os.makedirs(charts_dir, exist_ok=True)
        results = load_results(results_dir, exclude_ragonhunter=False, exclude_all_rag=True)
        print("\nGenerowanie wykresów (bez RAG)...")
        print("-" * 40)
        generate_all_charts(results, charts_dir, metrics)
    elif args.no_ragonhunter:
        # Tylko wersja bez RAG hunter
        charts_dir = os.path.join(script_dir, 'charts', 'bez_ragonhunter')
        os.makedirs(charts_dir, exist_ok=True)
        results = load_results(results_dir, exclude_ragonhunter=True, exclude_all_rag=False)
        print("\nGenerowanie wykresów (bez RAG hunter)...")
        print("-" * 40)
        generate_all_charts(results, charts_dir, metrics)
    else:
        # Domyślnie - pełna wersja
        results = load_results(results_dir, exclude_ragonhunter=False, exclude_all_rag=False)
    
        print("\nGenerowanie wykresów...")
        print("-" * 40)
        generate_all_charts(results, charts_dir, metrics)
    
    print()
    print("=" * 60)
    print(f"Wykresy zostały zapisane w folderze: {charts_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

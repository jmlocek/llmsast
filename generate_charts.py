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
    # Uwaga: kolejność kluczy odpowiada domyślnej kolejności modeli na wykresach.
    result_files = {
        # Granite-4h-Tiny
        'Granite-4h-Tiny': {
            'Podejście klasyczne': 'granite/diversevul_single_agent_progress.json',
            'Podejście klasyczne + RAG': 'granite/diversevul_single_agent_rag_progress .json',
            'Podejście łańcuchowe': 'granite/diversevul_multi_agent_progress.json',
            'Podejście łańcuchowe + RAG (dla agenta detekcji i agenta weryfikacji)': 'granite/diversevul_multi_agent_rag_progress.json',
            'Podejście łańcuchowe + RAG (dla agenta weryfikacji)': 'granite/diversevul_multi_agent_ragonhunter_progress.json',
        },
        # gpt-oss-20b
        'gpt-oss-20b': {
            'Podejście klasyczne': 'gpt-oss-20b/diverse_vuln/diversevul_single_agent_progress.json',
            'Podejście klasyczne + RAG': 'gpt-oss-20b/diverse_vuln/diversevul_single_agent_rag_progress.json',
            'Podejście łańcuchowe': 'gpt-oss-20b/diverse_vuln/diversevul_multi_agent_progress.json',
            'Podejście łańcuchowe + RAG (dla agenta detekcji i agenta weryfikacji)': 'gpt-oss-20b/diverse_vuln/diversevul_multi_agent_rag_progress.json',
            'Podejście łańcuchowe + RAG (dla agenta weryfikacji)': 'gpt-oss-20b/diverse_vuln/diversevul_multi_agent_ragonhunter_progress.json',
        },
        # gpt-oss-120b
        'gpt-oss-120b': {
            'Podejście klasyczne': 'gpt-oss-120b/diversevul_single_agent_progress.json',
            'Podejście klasyczne + RAG': 'gpt-oss-120b/diversevul_single_agent_rag_progress.json',
            'Podejście łańcuchowe': 'gpt-oss-120b/diversevul_multi_agent_progress.json',
            'Podejście łańcuchowe + RAG (dla agenta detekcji i agenta weryfikacji)': 'gpt-oss-120b/diversevul_multi_agent_rag_progress.json',
            'Podejście łańcuchowe + RAG (dla agenta weryfikacji)': 'gpt-oss-120b/diversevul_multi_agent_ragonhunter_progress.json',
        },
    }
    
    for model_name, configs in result_files.items():
        results[model_name] = {}
        for config_name, file_path in configs.items():
            # Pomiń wszystkie konfiguracje z RAG jeśli flaga jest ustawiona
            if exclude_all_rag and 'RAG' in config_name:
                continue
            # Pomiń konfiguracje RAG (Hunter) - tylko wariant "Hunter" bez "Hunter + FP Remover"
            if exclude_ragonhunter and 'RAG (Hunter)' in config_name and 'Hunter + FP Remover' not in config_name:
                continue
            full_path = results_path / file_path
            if full_path.exists():
                with open(full_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results[model_name][config_name] = data
            else:
                print(f"Ostrzeżenie: Plik {full_path} nie istnieje")
    
    return results


def get_models_and_approaches(results: dict):
    """Zwraca listę modeli i podejść w spójnej kolejności."""
    desired_models = ['Granite-4h-Tiny', 'gpt-oss-20b', 'gpt-oss-120b']
    models = [m for m in desired_models if m in results]
    for m in results.keys():
        if m not in models:
            models.append(m)
    desired_order = [
        'Podejście klasyczne',
        'Podejście klasyczne + RAG',
        'Podejście łańcuchowe',
        'Podejście łańcuchowe + RAG (dla agenta detekcji i agenta weryfikacji)',
        'Podejście łańcuchowe + RAG (dla agenta weryfikacji)',
    ]

    present = set()
    for _, configs in results.items():
        present.update(configs.keys())

    approaches = [name for name in desired_order if name in present]
    if not approaches:
        approaches = sorted(list(present))

    return models, approaches


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
    # Wersja przeglądowa: siatka 2x2 (metryki) z osiami X = modele i legendą = podejścia
    models, approaches = get_models_and_approaches(results)
    if not models or not approaches:
        print("Brak danych do utworzenia wykresu")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    axes = axes.flatten()

    colors = plt.cm.Set2(np.linspace(0, 1, max(len(approaches), 1)))
    x = np.arange(len(models))
    width = min(0.18, 0.8 / max(len(approaches), 1))

    for idx, metric in enumerate(metrics[:4]):
        ax = axes[idx]
        for j, (approach, color) in enumerate(zip(approaches, colors)):
            values = []
            for model in models:
                data = results.get(model, {}).get(approach)
                if data:
                    values.append(calculate_metrics(data)[metric])
                else:
                    values.append(0)

            offset = (j - (len(approaches) - 1) / 2) * width
            ax.bar(x + offset, values, width, label=approach, color=color, edgecolor='black', linewidth=0.4)

        ax.set_title(
            f'Porównanie metryki {metric} dla wybranych dużych modeli językowych\n'
            f'oraz podejścia klasycznego i autorskiego łańcuchowego'
        )
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=0)
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_ylabel('Wartość metryki' if idx % 2 == 0 else '')

    # Ukryj ewentualne nadmiarowe osie
    for k in range(len(metrics[:4]), 4):
        axes[k].axis('off')

    fig.suptitle(title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(len(approaches), 3), framealpha=0.9)

    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
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
    
    models, approaches = get_models_and_approaches(results)
    if not models or not approaches:
        print("Brak danych do utworzenia wykresów")
        return

    colors = plt.cm.Set2(np.linspace(0, 1, max(len(approaches), 1)))
    x = np.arange(len(models))
    width = min(0.18, 0.8 / max(len(approaches), 1))

    for metric in metrics_pl:
        fig, ax = plt.subplots(figsize=(12, 7))

        for j, (approach, color) in enumerate(zip(approaches, colors)):
            values = []
            present_mask = []
            for model in models:
                data = results.get(model, {}).get(approach)
                if data:
                    values.append(calculate_metrics(data)[metric])
                    present_mask.append(True)
                else:
                    values.append(0)
                    present_mask.append(False)

            offset = (j - (len(approaches) - 1) / 2) * width
            rects = ax.bar(x + offset, values, width, label=approach, color=color,
                           edgecolor='black', linewidth=0.4)

            for rect, value, is_present in zip(rects, values, present_mask):
                if is_present:
                    height = rect.get_height()
                    ax.annotate(f'{value:.2f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=8)

        ax.set_ylabel('Wartość metryki')
        ax.set_title(
            f'Porównanie metryki {metric} dla wybranych dużych modeli językowych\n'
            f'oraz podejścia klasycznego i autorskiego łańcuchowego'
        )
        ax.set_xticks(x)
        ax.set_xticklabels(models)
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
    ax.set_title(
        'Porównanie metryki Precyzja, Czułość, Wynik F1 i Wynik F2\n'
        'dla wybranych dużych modeli językowych oraz podejścia klasycznego i autorskiego łańcuchowego',
        y=1.08,
    )
    
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
    
    ax.set_title(
        'Porównanie metryki Precyzja, Czułość, Wynik F1 i Wynik F2\n'
        'w mapie cieplnej dla wybranych dużych modeli językowych\n'
        'oraz podejścia klasycznego i autorskiego łańcuchowego'
    )
    
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
    # Wykres podsumowujący: średnia z (Precyzja, Czułość, F1, F2) per model i podejście
    metrics = ['Precyzja', 'Czułość', 'Wynik F1', 'Wynik F2']
    models, approaches = get_models_and_approaches(results)
    if not models or not approaches:
        print("Brak danych do utworzenia wykresu")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(approaches), 1)))
    x = np.arange(len(models))
    width = min(0.18, 0.8 / max(len(approaches), 1))

    for j, (approach, color) in enumerate(zip(approaches, colors)):
        values = []
        present_mask = []
        for model in models:
            raw = results.get(model, {}).get(approach)
            if raw:
                m = calculate_metrics(raw)
                values.append(float(np.mean([m[k] for k in metrics])))
                present_mask.append(True)
            else:
                values.append(0)
                present_mask.append(False)

        offset = (j - (len(approaches) - 1) / 2) * width
        rects = ax.bar(x + offset, values, width, label=approach, color=color,
                       edgecolor='black', linewidth=0.4)

        for rect, value, is_present in zip(rects, values, present_mask):
            if is_present:
                ax.annotate(f'{value:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Średnia wartość metryk')
    ax.set_title(
        'Porównanie metryki Średnia (Precyzja, Czułość, Wynik F1, Wynik F2)\n'
        'dla wybranych dużych modeli językowych oraz podejścia klasycznego i autorskiego łańcuchowego'
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

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
    models, approaches = get_models_and_approaches(results)
    if not models or not approaches:
        print("Brak danych do utworzenia wykresu")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(approaches), 1)))
    x = np.arange(len(models))
    width = min(0.18, 0.8 / max(len(approaches), 1))

    for ax, metric, title in zip(
        axes,
        ['Wynik F1', 'Wynik F2'],
        [
            'Porównanie metryki Wynik F1 dla wybranych dużych modeli językowych\n'
            'oraz podejścia klasycznego i autorskiego łańcuchowego',
            'Porównanie metryki Wynik F2 dla wybranych dużych modeli językowych\n'
            'oraz podejścia klasycznego i autorskiego łańcuchowego',
        ],
    ):
        for j, (approach, color) in enumerate(zip(approaches, colors)):
            values = []
            present_mask = []
            for model in models:
                raw = results.get(model, {}).get(approach)
                if raw:
                    values.append(calculate_metrics(raw)[metric])
                    present_mask.append(True)
                else:
                    values.append(0)
                    present_mask.append(False)

            offset = (j - (len(approaches) - 1) / 2) * width
            rects = ax.bar(x + offset, values, width, label=approach, color=color,
                           edgecolor='black', linewidth=0.4)

            for rect, value, is_present in zip(rects, values, present_mask):
                if is_present:
                    ax.annotate(f'{value:.2f}',
                                xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=8)

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    axes[0].set_ylabel('Wartość metryki')
    fig.suptitle(
        'Porównanie metryki Wynik F1 i Wynik F2 dla wybranych dużych modeli językowych\n'
        'oraz podejścia klasycznego i autorskiego łańcuchowego'
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(len(approaches), 3), framealpha=0.9)

    plt.tight_layout(rect=[0, 0.08, 1, 0.92])

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
    models, approaches = get_models_and_approaches(results)
    if not models or not approaches:
        print("Brak danych do utworzenia wykresu")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(approaches), 1)))
    x = np.arange(len(models))
    width = min(0.18, 0.8 / max(len(approaches), 1))

    for ax, metric, title in zip(
        axes,
        ['Precyzja', 'Czułość'],
        [
            'Porównanie metryki Precyzja dla wybranych dużych modeli językowych\n'
            'oraz podejścia klasycznego i autorskiego łańcuchowego',
            'Porównanie metryki Czułość dla wybranych dużych modeli językowych\n'
            'oraz podejścia klasycznego i autorskiego łańcuchowego',
        ],
    ):
        for j, (approach, color) in enumerate(zip(approaches, colors)):
            values = []
            present_mask = []
            for model in models:
                raw = results.get(model, {}).get(approach)
                if raw:
                    values.append(calculate_metrics(raw)[metric])
                    present_mask.append(True)
                else:
                    values.append(0)
                    present_mask.append(False)

            offset = (j - (len(approaches) - 1) / 2) * width
            rects = ax.bar(x + offset, values, width, label=approach, color=color,
                           edgecolor='black', linewidth=0.4)

            for rect, value, is_present in zip(rects, values, present_mask):
                if is_present:
                    ax.annotate(f'{value:.2f}',
                                xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=8)

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    axes[0].set_ylabel('Wartość metryki')
    fig.suptitle(
        'Porównanie metryki Precyzja i Czułość dla wybranych dużych modeli językowych\n'
        'oraz podejścia klasycznego i autorskiego łańcuchowego'
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(len(approaches), 3), framealpha=0.9)

    plt.tight_layout(rect=[0, 0.08, 1, 0.92])

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
        'Porównanie metryki Precyzja, Czułość, Wynik F1 i Wynik F2\n'
        'dla wybranych dużych modeli językowych\n'
        'oraz podejścia klasycznego i autorskiego łańcuchowego'
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

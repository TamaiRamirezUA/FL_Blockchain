import json
import matplotlib.pyplot as plt
import os
import numpy as np
import tomli
from pathlib import Path

# Ruta al archivo
toml_path = Path(__file__).parent.parent / "pyproject.toml"
with toml_path.open("rb") as f:
    toml_config = tomli.load(f)
PROJECT_PATH = toml_config["tool"]["flwr"]["app"]["config"]["project-path"]
NUM_CLIENTS = toml_config["tool"]["flwr"]["app"]["config"]["num_clients"]

def load_results(json_paths):
    """Carga varios archivos JSON y devuelve una lista de diccionarios."""
    results = []
    for path in json_paths:
        if not os.path.exists(path):
            print(f"⚠️ Archivo no encontrado: {path}")
            continue
        with open(path, 'r') as f:
            data = json.load(f)
            results.append((path, data))
    return results

def plot_metrics(results, output_dir):
    """
    Genera dos gráficos:
    1️⃣ train_accuracy vs val_accuracy
    2️⃣ train_loss vs val_loss
    Cada archivo JSON genera dos curvas (train y val).
    """
    
    
    # --- Plot de accuracies ---
    plt.figure(figsize=(8, 5))
    for name, data in results:
        # Determinar etiqueta legible según la ruta del archivo
        if "/IID/" in name:
            label_base = "IID"
        elif "/Non-IID/1.0/" in name:
            label_base = "Non-IID_1.0"
        elif "/Non-IID/0.5/" in name:
            label_base = "Non-IID_0.5"
        elif "/Non-IID/0.1/" in name:
            label_base = "Non-IID_0.1"
        else:
            label_base = os.path.basename(name).replace(".json", "")
            
        rounds = []
        val_acc, test_acc = [], []
        for rnd_str in sorted(data.keys(), key=lambda x: int(x)):
            rnd = int(rnd_str)
            res = data[rnd_str]

            rounds.append(rnd)
            val_acc.append(res["val_accuracy"])
            test_acc.append(res["test_accuracy"])
            
        plt.plot(rounds, val_acc, linestyle='--', marker='o', label=f"{label_base} - Val Acc")
        plt.plot(rounds, test_acc, linestyle='-', marker='x', label=f"{label_base} - Test Acc")
        
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.ylim(0,1)
    plt.xticks(rounds)
    plt.title("Global Model Accuracy Curve (Val vs Test)")
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/accuracies_global.png")
    plt.close()

    # --- Plot de pérdidas ---
    plt.figure(figsize=(8, 5))
    for name, data in results:
        # Determinar etiqueta legible según la ruta del archivo
        if "/IID/" in name:
            label_base = "IID"
        elif "/Non-IID/1.0/" in name:
            label_base = "Non-IID_1.0"
        elif "/Non-IID/0.5/" in name:
            label_base = "Non-IID_0.5"
        elif "/Non-IID/0.1/" in name:
            label_base = "Non-IID_0.1"
        else:
            label_base = os.path.basename(name).replace(".json", "")
            
        rounds = []
        val_loss, test_loss = [], []
        for rnd_str in sorted(data.keys(), key=lambda x: int(x)):
            rnd = int(rnd_str)
            res = data[rnd_str]

            rounds.append(rnd)
            val_loss.append(res["loss_val"])
            test_loss.append(res["loss_test"])
        plt.plot(rounds, val_loss, linestyle='--', marker='o', label=f"{label_base} - Val Loss")
        plt.plot(rounds, test_loss, linestyle='-', marker='x', label=f"{label_base} - Test Loss")
    
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.ylim(0,2)
    plt.xticks(rounds)
    plt.title("Global Model Loss Curve (Val vs Test)")
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/losess_global.png")
    plt.close()
    
    # --- Plot de CV ---
    plt.figure(figsize=(8, 5))
    for name, data in results:
        # Determinar etiqueta legible según la ruta del archivo
        if "/IID/" in name:
            label_base = "IID"
        elif "/Non-IID/1.0/" in name:
            label_base = "Non-IID_1.0"
        elif "/Non-IID/0.5/" in name:
            label_base = "Non-IID_0.5"
        elif "/Non-IID/0.1/" in name:
            label_base = "Non-IID_0.1"
        else:
            label_base = os.path.basename(name).replace(".json", "")
            
        rounds = []
        cv = []
        for rnd_str in sorted(data.keys(), key=lambda x: int(x)):
            rnd = int(rnd_str)
            res = data[rnd_str]

            rounds.append(rnd)
            cv.append(res["CV"])
        plt.plot(rounds, cv, linestyle='--', marker='o', label=f"{label_base} - CV")
    
    plt.xlabel("Round")
    plt.ylabel("CV")
    plt.ylim(0,2)
    plt.xticks(rounds)
    plt.title("Global Model Stability")
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/cv.png")
    plt.close()



if __name__ == "__main__":
    # 🔹 Rutas de ejemplo (puedes añadir las que quieras)
    output_dir = f"{PROJECT_PATH}/NC_{NUM_CLIENTS}"
    os.makedirs(output_dir, exist_ok=True)
    json_files = [
        f"{output_dir}/IID/results_global.json",
        f"{output_dir}/Non-IID/1.0/results_global.json",
        f"{output_dir}/Non-IID/0.5/results_global.json",
        f"{output_dir}/Non-IID/0.1/results_global.json"
    ]

    results = load_results(json_files)
    if results:
        plot_metrics(results, output_dir=output_dir)
    else:
        print("No se cargaron resultados válidos.")

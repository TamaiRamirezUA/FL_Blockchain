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

# Carpetas principales
folders = ["NC_4", "NC_6", "NC_8"]

# Escenarios a analizar
scenarios = {
    "IID": "IID",
    "Non-IID_1.0": os.path.join("Non-IID", "1.0"),
    "Non-IID_0.5": os.path.join("Non-IID", "0.5"),
    "Non-IID_0.1": os.path.join("Non-IID", "0.1")
}

# Carpeta de salida para las gráficas
output_dir = "plot_all"
os.makedirs(output_dir, exist_ok=True)

# Función para leer los resultados JSON
def load_results(path):
    with open(path, "r") as f:
        data = json.load(f)
    iterations = sorted(map(int, data.keys()))
    val_acc = [data[str(i)]["val_accuracy"] for i in iterations]
    test_acc = [data[str(i)]["test_accuracy"] for i in iterations]
    return iterations, val_acc, test_acc

# Generar y guardar las gráficas
for scenario_name, subpath in scenarios.items():
    plt.figure(figsize=(8, 6))
    
    for folder in folders:
        json_path = os.path.join(folder, subpath, "results_global.json")
        num_clients = folder.split("_")[1]
        if os.path.exists(json_path):
            iterations, val_acc, test_acc = load_results(json_path)
            plt.plot(iterations, val_acc, marker='o', linestyle='-', label=f"{num_clients} vehicles - val acc")
            plt.plot(iterations, test_acc, marker='s', linestyle='--', label=f"{num_clients} vehicles - test acc")
        else:
            print(f"⚠️ No se encontró {json_path}")
    
    # Configuración de la gráfica
    plt.title(f"Global Model Accuracy Comparison - {scenario_name} scenario across vehicles", fontsize=14)
    plt.xlabel("Round", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(0,1)
    plt.xticks(range(1,6))
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Guardar figura
    output_path = os.path.join(output_dir, f"{scenario_name}.png")
    plt.savefig(output_path, dpi=300)
    print(f"✅ Gráfico guardado en: {output_path}")
    
    plt.close()

print("\n🎉 Todos los gráficos fueron generados y guardados correctamente.")

import json
import matplotlib.pyplot as plt
import os
import numpy as np
import tomli
from pathlib import Path

# Ruta al archivo pyproject.toml
toml_path = Path(__file__).parent.parent / "pyproject.toml"
with toml_path.open("rb") as f:
    toml_config = tomli.load(f)

PROJECT_PATH = toml_config["tool"]["flwr"]["app"]["config"]["project-path"]

# Directorio de resultados (ejemplo: NC_2, NC_5, NC_10, etc.)
base_dir = Path(PROJECT_PATH)

# Encontrar todas las carpetas NC_*
nc_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("NC_")])

# Carpeta de salida para los gráficos globales
output_dir = base_dir / "plots_global"
os.makedirs(output_dir, exist_ok=True)

plt.figure()

# Iterar sobre cada experimento (diferente número de clientes)
for nc_dir in nc_dirs:
    num_clients = nc_dir.name.split("_")[1]
    results_path = nc_dir / "results.json"
    
    if not results_path.exists():
        print(f"⚠️ No se encontró {results_path}, se omite.")
        continue

    # Cargar resultados
    with open(results_path, "r") as f:
        results = json.load(f)

    rounds = []
    val_acc, test_acc = [], []
    val_loss, test_loss = [], []

    for rnd_str in sorted(results.keys(), key=lambda x: int(x)):
        rnd = int(rnd_str)
        data = results[rnd_str]

        rounds.append(rnd)
        val_acc.append(data["val_accuracy"])
        test_acc.append(data["test_accuracy"])
        val_loss.append(data["loss_val"])
        test_loss.append(data["loss_test"])

    # Graficar en la misma figura
    plt.plot(rounds, test_acc, label=f"Test Acc - {num_clients} clients")
    plt.plot(rounds, val_acc, linestyle="--", label=f"Val Acc - {num_clients} clients")
    # Si quieres también las pérdidas, puedes activarlas
    # plt.plot(rounds, val_loss, linestyle=":", label=f"Val Loss - {num_clients} clients")
    # plt.plot(rounds, test_loss, linestyle="-.", label=f"Test Loss - {num_clients} clients")

# Línea de referencia
plt.axhline(y=1, color='r', linestyle='--', linewidth=1.5)

# Configuración de la gráfica
plt.xlabel("Round")
plt.ylabel("Metrics")
plt.title("Federated Learning - Comparison across clients")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Guardar gráfico
plt.savefig(output_dir / "comparison_metrics.png")
plt.close()

print(f"✅ Gráfico comparativo guardado en: {output_dir}")

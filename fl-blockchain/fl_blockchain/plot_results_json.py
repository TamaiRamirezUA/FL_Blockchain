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

results_path = f"{PROJECT_PATH}/NC_{NUM_CLIENTS}/results.json"
output_dir = f"{PROJECT_PATH}/NC_{NUM_CLIENTS}/plots/"
os.makedirs(output_dir, exist_ok=True)

# Cargar los resultados
with open(results_path, "r") as f:
    results = json.load(f)

# Inicializar diccionarios para cada métrica
rounds = []
train_acc, val_acc, test_acc = [], [], []
train_loss, val_loss, test_loss = [], [], []

# Extraer datos
for rnd_str in sorted(results.keys(), key=lambda x: int(x)):
    rnd = int(rnd_str)
    data = results[rnd_str]

    rounds.append(rnd)
    # train_acc.append(data["train_accuracy"])
    val_acc.append(data["val_accuracy"])
    test_acc.append(data["test_accuracy"])
    
    # train_loss.append(data["loss_train"])
    val_loss.append(data["loss_val"])
    test_loss.append(data["loss_test"])

# Plot: Accuracy and Loss
plt.figure()
# plt.plot(rounds, train_acc, label="Train Accuracy")
plt.plot(rounds, val_acc, label="Validation Accuracy")
plt.plot(rounds, test_acc, label="Test Accuracy")
plt.plot(rounds, val_loss, label="Validation Loss")
plt.plot(rounds, test_loss, label="Test Loss")
plt.axhline(y=1, color='r', linestyle='--', linewidth=1.5)
plt.xlabel("Round")
plt.ylabel("Metrics")
plt.title("Global Evaluation")
plt.xticks(np.arange(1,6,1))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/metrics_over_rounds.png")
plt.close()


print(f"Gráficos guardados en: {output_dir}")

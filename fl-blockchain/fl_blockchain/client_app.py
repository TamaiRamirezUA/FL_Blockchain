"""fl-blockchain: A Flower / TensorFlow app."""

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from fl_blockchain.task import load_data, load_model, PROJECT_PATH, NUM_CLIENTS, NON_IID, ALPHA
import matplotlib.pyplot as plt
import os
import json
from sklearn.metrics import f1_score
import numpy as np

def smooth_curve(points, factor=0.8):
    """Aplica una media móvil exponencial a los puntos."""
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(
        self, model, epochs, batch_size, verbose, partition_id, num_partitions, non_iid, alpha 
    ):
        self.model = model
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = load_data(csv_path=PROJECT_PATH+"/Dataset/tratado/dataset_etiquetado.csv", 
                                                                    partition_id=partition_id,
                                                                    num_partitions=num_partitions,
                                                                    non_iid=non_iid,
                                                                    alpha=alpha)
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.partition_id = partition_id
        self.num_partitions = num_partitions
        self.non_iid = non_iid
        self.alpha = alpha

    def fit(self, parameters, config):
        actual_round = config["round"]
        self.model.set_weights(parameters)
        
        H = self.model.fit(self.X_train, self.y_train,
            epochs=self.epochs,
            validation_data=(self.X_val, self.y_val),
            batch_size=self.batch_size,
            verbose=self.verbose
        )

        # # Get Train Metrics
        # train_acc = H.history["accuracy"][-1]
        # val_acc = H.history["val_accuracy"][-1]


        
        # Gráfico de precisión (accuracy)
        # plt.figure()
        # plt.plot(smooth_curve(H.history["accuracy"], factor=0.5), label="Train Accuracy")
        # plt.plot(smooth_curve(H.history["val_accuracy"], factor=0.55), label="Validation Accuracy")
        # plt.plot(smooth_curve(H.history["loss"], factor=0.5), label="Train Loss")
        # plt.plot(smooth_curve(H.history["val_loss"], factor=0.55), label="Validation Loss")
        # plt.axhline(y=1, color='r', linestyle='--', linewidth=1.5)
        # plt.title(f"Client {self.partition_id} - Metrics (Round {actual_round})")
        # plt.xlabel("Epoch")
        # plt.ylabel("Metrics")
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig(f"{plot_dir}/metrics_round_{actual_round}.png")
        # plt.close()
        
        results_client = {
            "train_accuracy": H.history["accuracy"],
            "val_accuracy": H.history["val_accuracy"],
            "train_loss": H.history["loss"],
            "val_loss": H.history["val_loss"],
            "communication_cost": 2 * self.model.count_params() * 4 / (1024 * 1024),
            "size_model": self.model.count_params() * 4 / (1024 * 1024)
        }

        # Directorio de guardado
        if NON_IID:
            save_path=f"{PROJECT_PATH}/NC_{NUM_CLIENTS}/Non-IID/{ALPHA}"
        else:
            save_path=f"{PROJECT_PATH}/NC_{NUM_CLIENTS}/IID"
        
        # Guardar en JSON
        with open(f"{save_path}/results_client_{self.partition_id}_{actual_round}.json", "w") as f:
            json.dump(results_client, f, indent=4)
            
        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        print("Evaluating Model Performance...")
        # loss_train, acc_train = self.model.evaluate(self.images["train"], self.labels["train"], verbose=0)
        
        loss_val, acc_val = self.model.evaluate(self.X_val, self.y_val, verbose=1)
        loss_test, acc_test = self.model.evaluate(self.X_test, self.y_test, verbose=1)
        y_pred = self.model.predict(self.X_test)
        y_pred_labels = np.argmax(y_pred, axis=1)

        f1_score_test = f1_score(self.y_test, y_pred_labels, average='macro')
        return loss_test, len(self.X_test), {
            # "loss_train": loss_train,
            "loss_val": loss_val,
            "loss_test": loss_test,
            # "train_accuracy": acc_train,
            "val_accuracy": acc_val,
            "test_accuracy": acc_test,
            "test_f1-score": f1_score_test
        }


def client_fn(context: Context):
    # Load model and data
    net = load_model()

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose")
    non_iid = context.run_config.get("non-iid")
    alpha = context.run_config.get("alpha")
    # Return Client instance
    return FlowerClient(
        net, epochs, batch_size, verbose, partition_id, num_partitions, non_iid, alpha
    ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)

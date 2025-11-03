"""fl-blockchain: A Flower / TensorFlow app."""

import os

import keras
from keras import layers
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, Activation, BatchNormalization, Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras import optimizers
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.utils import get_file
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, array_to_img, img_to_array
from tensorflow.keras.optimizers.schedules import ExponentialDecay

import matplotlib.pyplot as plt
import tomli
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # tf.config.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    # tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*7)])
  except RuntimeError as e:
    print(e)

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

tf.random.set_seed(
    42
)

toml_path = Path(__file__).parent.parent / "pyproject.toml"
with toml_path.open("rb") as f:
    toml_config = tomli.load(f)
PROJECT_PATH = toml_config["tool"]["flwr"]["app"]["config"]["project-path"]
NUM_CLIENTS = toml_config["tool"]["flwr"]["app"]["config"]["num_clients"]
NON_IID = toml_config["tool"]["flwr"]["app"]["config"]["non_iid"]
ALPHA = toml_config["tool"]["flwr"]["app"]["config"]["alpha"]

def load_model():
    model = tf.keras.models.Sequential([
        # Bloque 1: 32 filtros
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3), kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.4),
        
        # Bloque 2: 64 filtros
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Bloque 3: 128 filtros
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # # Bloque 4: 256 filtros
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Capas densas
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.2),

        # Capa de salida
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    
    lr_schedule = ExponentialDecay(
        initial_learning_rate=5e-5,
        decay_steps=10000,
        decay_rate=0.9,
        staircase=True
    )

    # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-07)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
    
""" This code has been developed by Tamai Ramírez Gordillo (GitHub: TamaiRamirezUA)"""


class DirichletPartitioner:
    """Particiona un dataset en varias partes usando una distribución Dirichlet (no-IID)."""
    def __init__(self, y, num_partitions=5, alpha=0.5, seed=42):
        self.y = np.array(y)
        self.num_partitions = num_partitions
        self.alpha = alpha
        self.rng = np.random.RandomState(seed)

    def get_partitions(self):
        num_classes = len(np.unique(self.y))
        idx_by_class = {c: np.where(self.y == c)[0] for c in range(num_classes)}
        partitions = [[] for _ in range(self.num_partitions)]

        for c, idxs in idx_by_class.items():
            self.rng.shuffle(idxs)
            proportions = self.rng.dirichlet(np.repeat(self.alpha, self.num_partitions))
            proportions = np.array(proportions) / proportions.sum()
            split_points = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
            splits = np.split(idxs, split_points)
            for p, idxs_p in zip(partitions, splits):
                p.extend(idxs_p)
        return partitions

# Función para generar imágenes aumentadas de una clase específica
def augment_class(images, labels, target_class, num_to_generate, datagen):
    # Filtrar imágenes de la clase objetivo
    class_images = images[labels == target_class]
    class_labels = labels[labels == target_class]
    
    # Generar imágenes aumentadas
    datagen.fit(class_images)
    generator = datagen.flow(class_images, class_labels, batch_size=1)
    
    generated_images = []
    generated_labels = []
    
    for _ in range(num_to_generate):
        batch_x, batch_y = next(generator)
        generated_images.append(batch_x[0])  # Tomar la primera (y única) imagen del batch
        generated_labels.append(batch_y[0])
    
    return generated_images, generated_labels

def load_data(csv_path, img_size=(128, 128), partition_id=0, num_partitions=1,
              non_iid=False, alpha=0.5):
    """
    Lee csv con columnas 'file_path' y 'label', carga imágenes, codifica labels,
    equilibra clases mediante aumento, y devuelve (train/val/test) del IID o Non-IID.

    Parámetros:
        csv_path : str -> ruta del CSV
        img_size : tuple -> tamaño de las imágenes
        partition_id : int -> índice de la partición a usar (0..num_partitions-1)
        num_partitions : int -> número de particiones totales
        non_iid : bool -> si True, usa particionado Dirichlet (Non-IID)
        alpha : float -> parámetro Dirichlet (menor = más desigual)
    """
    # --- Lectura y codificación ---
    df = pd.read_csv(csv_path)
    if 'file_path' not in df.columns or 'label' not in df.columns:
        raise ValueError("El CSV debe contener columnas 'file_path' y 'label'")

    label_encoder = LabelEncoder()
    df['label_enc'] = label_encoder.fit_transform(df['label'].astype(str))

    # --- Carga de imágenes ---
    images, labels = [], []
    for _, row in df.iterrows():
        img_path = row['file_path']
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Imagen no encontrada: {img_path}")
        img = load_img(img_path, target_size=img_size)
        img_array = img_to_array(img).astype('float32') / 255.0
        images.append(img_array)
        labels.append(int(row['label_enc']))

    images = np.array(images, dtype='float32')
    labels = np.array(labels, dtype='int32')

    # --- Balanceo de clases ---
    datagen = ImageDataGenerator(rotation_range=10,
                                    width_shift_range=0.01,
                                    height_shift_range=0.01,
                                    zoom_range=0.01,
                                    # brightness_range=[0.9, 1.1],
                                    horizontal_flip=True)
    
    # unique_labels, counts = np.unique(labels, return_counts=True)
    # target_count = counts.max()

    # X_balanced, y_balanced = [], []
    # for lbl in unique_labels:
    #     idxs = np.where(labels == lbl)[0]
    #     X_cls = images[idxs]
    #     y_cls = labels[idxs]

    #     X_balanced.extend(list(X_cls))
    #     y_balanced.extend(list(y_cls))

    #     n_to_gen = int(target_count - len(X_cls))
    #     if n_to_gen > 0 and len(X_cls) > 0:
    #         gen = datagen.flow(X_cls, y_cls, batch_size=1, shuffle=True)
    #         for _ in range(n_to_gen):
    #             x_batch, y_batch = next(gen)
    #             X_balanced.append(x_batch[0])
    #             y_balanced.append(int(y_batch[0]))
    
    # Aumentado de datos

    X_img = []
    y_label = []
    
    # Generar imágenes aumentadas para cada clase
    # Clase 0 (Accidente) - generar 364 imágenes
    aug_images_0, aug_labels_0 = augment_class(images, labels, target_class=0, num_to_generate=364, datagen=datagen)
    X_img.extend(aug_images_0)
    y_label.extend(aug_labels_0)

    # Clase 1 (Tráfico denso) - generar 249 imágenes
    aug_images_1, aug_labels_1 = augment_class(images, labels, target_class=1, num_to_generate=249, datagen=datagen)
    X_img.extend(aug_images_1)
    y_label.extend(aug_labels_1)

    # Clase 3 (Tráfico escaso) - generar 80 imágenes
    aug_images_3, aug_labels_3 = augment_class(images, labels, target_class=3, num_to_generate=80, datagen=datagen)
    X_img.extend(aug_images_3)
    y_label.extend(aug_labels_3)

    # Añadir imágenes originales
    X_img.extend(images)
    y_label.extend(labels)

    X_balanced = np.array(X_img, dtype='float32')
    y_balanced = np.array(y_label, dtype='int32')

    # # --- Mezclar ---
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(X_balanced))
    X_balanced = X_balanced[perm]
    y_balanced = y_balanced[perm]

    # --- IID o Non-IID ---
    if NON_IID:
        print(f"Generando particiones Non-IID (Dirichlet alpha={alpha})...")
        dp = DirichletPartitioner(y_balanced, num_partitions=num_partitions, alpha=alpha)
        partitions = dp.get_partitions()
        selected_idxs = partitions[partition_id]

        # Visualización
        plt.figure(figsize=(8, 5))
        counts_by_part = []
        for i, idxs in enumerate(partitions):
            unique, cts = np.unique(y_balanced[idxs], return_counts=True)
            counts_by_part.append(dict(zip(unique, cts)))
        
        # ✅ Colores fijos por cliente
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(num_partitions)]
        all_classes = np.unique(y_balanced)
        width = 0.8 / num_partitions
        for i, cls in enumerate(all_classes):
            for j in range(num_partitions):
                count = counts_by_part[j].get(cls, 0)
                plt.bar(i + j * width, count, width=width, color=colors[j], label=f'Client {j}' if i == 0 else "")
        
        plt.xticks(np.arange(len(all_classes)) + 0.4, labels=['Accident', 'Dense Traffic', 'Meteorology', 'Sparse Traffic'])
        plt.xlabel("Classes")
        plt.ylabel("Number of examples")
        plt.title(f"Class Distribution (Non-IID, alpha={alpha})")
        plt.legend(loc='lower left')
        plt.tight_layout()
        save_path=f"{PROJECT_PATH}/NC_{NUM_CLIENTS}/Non-IID/{ALPHA}"
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/Non_IID_distribution_partition.png")
        plt.close()
    else:
        print("Generando dataset IID...")
        indices = np.linspace(0, len(X_balanced), num_partitions + 1, dtype=int)
        partitions = [
            np.arange(indices[i], indices[i + 1], dtype=int)
            for i in range(num_partitions)
        ]
        selected_idxs = partitions[partition_id]

        # --- Visualización IID (igual estilo que Non-IID) ---
        plt.figure(figsize=(8, 5))
        counts_by_part = []
        for i, idxs in enumerate(partitions):
            unique, cts = np.unique(y_balanced[idxs], return_counts=True)
            counts_by_part.append(dict(zip(unique, cts)))

        # ✅ Colores fijos por cliente
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(num_partitions)]
        all_classes = np.unique(y_balanced)
        width = 0.8 / num_partitions
        for i, cls in enumerate(all_classes):
            for j in range(num_partitions):
                count = counts_by_part[j].get(cls, 0)
                plt.bar(i + j * width, count, width=width, color=colors[j], label=f'Client {j}' if i == 0 else "")

        plt.xticks(np.arange(len(all_classes)) + 0.4, labels=['Accident', 'Dense Traffic', 'Meteorology', 'Sparse Traffic'])
        plt.xlabel("Classes")
        plt.ylabel("Number of examples")
        plt.title("Class Distribution (IID)")
        plt.legend(loc='lower left')
        plt.tight_layout()
        save_path = f"{PROJECT_PATH}/NC_{NUM_CLIENTS}/IID"
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/IID_distribution.png")
        plt.close()

    # --- Seleccionar datos de la partición ---
    X_part = X_balanced[selected_idxs]
    y_part = y_balanced[selected_idxs]

    # --- Train / Val / Test ---
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_part, y_part, test_size=0.3, random_state=42, shuffle=True
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True
    )
    
    mostrar_balance(y_train, "Train")
    mostrar_balance(y_val, "Validation")
    mostrar_balance(y_test, "Test")

    return X_train, y_train, X_val, y_val, X_test, y_test

# Mostrar porcentaje de cada clase en cada subconjunto
def mostrar_balance(y, nombre):
    valores, conteos = np.unique(y, return_counts=True)
    porcentajes = conteos / len(y)
    print(f"\nBalance de clases en {nombre}:")
    for v, p in zip(valores, porcentajes):
        print(f"Clase {v}: {p:.3f}")
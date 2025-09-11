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


toml_path = Path(__file__).parent.parent / "pyproject.toml"
with toml_path.open("rb") as f:
    toml_config = tomli.load(f)
PROJECT_PATH = toml_config["tool"]["flwr"]["app"]["config"]["project-path"]
NUM_CLIENTS = toml_config["tool"]["flwr"]["app"]["config"]["num_clients"]

def load_model():
    model = Sequential()
    model.add(Input(shape=(128, 128, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))
    
    model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    return model
    
""" This code has been developed by Tamai Ramírez Gordillo (GitHub: TamaiRamirezUA)"""


def load_data(csv_path, img_size=(128, 128), partition_id=0, num_partitions=1):
    """
    Lee csv con columnas 'file_path' y 'label', carga imágenes, codifica labels,
    equilibra clases mediante aumento (hasta la clase más grande) y devuelve
    X_train, y_train, X_val, y_val, X_test, y_test (arrays numpy).
    """
    # --- Lectura y codificación ---
    df = pd.read_csv(csv_path)
    if 'file_path' not in df.columns or 'label' not in df.columns:
        raise ValueError("El CSV debe contener columnas 'file_path' y 'label'")

    label_encoder = LabelEncoder()
    df['label_enc'] = label_encoder.fit_transform(df['label'].astype(str))

    # --- Carga de imágenes ---
    images = []
    labels = []
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

    # --- Preparar aumento y balanceo ---
    datagen = ImageDataGenerator(rotation_range=10)
    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) == 0:
        raise ValueError("No hay ejemplos en el dataset")

    # Balancear hasta la clase con mayor número de ejemplos (más robusto que usar siempre la clase 0)
    target_count = counts.max()

    X_balanced = []
    y_balanced = []

    for lbl in unique_labels:
        idxs = np.where(labels == lbl)[0]
        X_cls = images[idxs]
        y_cls = labels[idxs]

        # Añadir originales de la clase
        X_balanced.extend(list(X_cls))
        y_balanced.extend(list(y_cls))

        # Generar aumentos si hace falta
        n_to_gen = int(target_count - len(X_cls))
        if n_to_gen <= 0:
            continue
        if len(X_cls) == 0:
            # no hay ejemplos para esta clase, saltar (o podrías duplicar imágenes de otra clase)
            continue

        gen = datagen.flow(X_cls, y_cls, batch_size=1, shuffle=True)
        for _ in range(n_to_gen):
            x_batch, y_batch = next(gen)   # devuelve (X_batch, y_batch)
            X_balanced.append(x_batch[0])
            # y_batch puede venir como array float (p.ej. array([1.])), convertir a int
            y_balanced.append(int(y_batch[0]))

    # Convertir a numpy arrays y barajar
    X_balanced = np.array(X_balanced, dtype='float32')
    y_balanced = np.array(y_balanced, dtype='int32')

    # Mezclar reproduciblemente
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(X_balanced))
    X_balanced = X_balanced[perm]
    y_balanced = y_balanced[perm]


    indices = np.linspace(0, len(X_balanced), num_partitions + 1, dtype=int)
    start = indices[partition_id]
    end = indices[partition_id + 1]
    X_part = X_balanced[start:end]
    y_part = y_balanced[start:end]

    # primero train vs temp (temp será VAL+TEST)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_part, y_part, test_size=0.3, random_state=42, shuffle=True
    )
    # dividir temp en val y test a partes iguales
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True
    )


    return X_train, y_train, X_val, y_val, X_test, y_test

# def data_augmentation(images, labels):
#     # Aumentado de datos
#     X_img = []
#     y_label = []

#     for i in range(0, len(images)):
#         X_img.append(images[i])
#         y_label.append(labels[i])
#     # Accidente

#     X_accident = []
#     y_accident = []

#     for i in range(0, len(images)):
#         if(labels[i] == 0):
#             X_accident.append(images[i])
#             y_accident.append(labels[i])

#     X_to_array = np.array(X_accident)
#     y_to_array = np.array(y_accident)

#     datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=10)

#     datagen.fit(X_to_array)
#     X_gen = datagen.flow(X_to_array,y_to_array,batch_size = 1)

    
#     for i in range(0, 853):
#         imagen = next(X_gen)[0]
#         for im in imagen:
#             im = np.asarray(im)
#             im = im.squeeze()
#             X_img.append(im) 
#             y_label.append(0)

#     # Trafico denso
#     X_traffic = []
#     y_traffic = []

#     for i in range(0, len(images)):
#         if(labels[i] == 1):
#             X_traffic.append(images[i])
#             y_traffic.append(labels[i])

#     X_to_array = np.array(X_traffic)
#     y_to_array = np.array(y_traffic)

#     datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=10)

#     datagen.fit(X_to_array)
#     X_gen = datagen.flow(X_to_array,y_to_array,batch_size = 1)

#     for i in range(0, 898):
#         imagen = next(X_gen)[0]
#         for im in imagen:
#             im = np.asarray(im)
#             im = im.squeeze()
#             X_img.append(im) 
#             y_label.append(0)

#     # Trafico escaso
#     X_traffic = []
#     y_traffic = []

#     for i in range(0, len(images)):
#         if(labels[i] == 3):
#             X_traffic.append(images[i])
#             y_traffic.append(labels[i])

#     X_to_array = np.array(X_traffic)
#     y_to_array = np.array(y_traffic)

#     datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=10)

#     datagen.fit(X_to_array)
#     X_gen = datagen.flow(X_to_array,y_to_array,batch_size = 1)

#     for i in range(0, 868):
#         imagen = next(X_gen)[0]
#         for im in imagen:
#             im = np.asarray(im)
#             im = im.squeeze()
#             X_img.append(im) 
#             y_label.append(0)

#     X_img = np.array(X_img)
#     y_label = np.array(y_label)

#     X_train, X_test, y_train, y_test = train_test_split(X_img, y_label, test_size=0.2, random_state=42)

#     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

#     return X_train, y_train, X_val, y_val, X_test, y_test
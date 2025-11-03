import os
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

from datetime import datetime
from datetime import timedelta
import time
import json
from sklearn.metrics import f1_score

tf.random.set_seed(
    42
)

start_time = time.time()

def cargar_datos(csv_path, img_size=(128, 128)):
    
    df = pd.read_csv(csv_path)
    
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])

    images = []
    labels = []

    for _, row in df.iterrows():
        img_path = row['file_path']
        label = row['label']
        
        # print(img_path)
        try:
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue

        images.append(img_array)
        labels.append(label)
    
    images = np.array(images, dtype='float32') / 255.0
    labels = np.array(labels, dtype='int32')
    
    return images, labels, label_encoder

csv_path = 'Dataset/tratado/dataset_etiquetado.csv'
images, labels, label_encoder = cargar_datos(csv_path)

"""
metereology = 0

for i in labels:
    if i == 3:
        metereology = metereology + 1

print(f"Hay {metereology} casos de meteorologia")
print(label_encoder.classes_)
"""

# Aumentado de datos

X_img = []
y_label = []

# Configurar data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.9, 1.1],
    horizontal_flip=True
)

# Función para generar imágenes aumentadas de una clase específica
def augment_class(images, labels, target_class, num_to_generate):
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

# Generar imágenes aumentadas para cada clase
# Clase 0 (Accidente) - generar 364 imágenes
aug_images_0, aug_labels_0 = augment_class(images, labels, target_class=0, num_to_generate=364)
X_img.extend(aug_images_0)
y_label.extend(aug_labels_0)

# Clase 1 (Tráfico denso) - generar 249 imágenes
aug_images_1, aug_labels_1 = augment_class(images, labels, target_class=1, num_to_generate=249)
X_img.extend(aug_images_1)
y_label.extend(aug_labels_1)

# Clase 3 (Tráfico escaso) - generar 80 imágenes
aug_images_3, aug_labels_3 = augment_class(images, labels, target_class=3, num_to_generate=80)
X_img.extend(aug_images_3)
y_label.extend(aug_labels_3)

# Añadir imágenes originales
X_img.extend(images)
y_label.extend(labels)

# Convertir a arrays numpy
new_X = np.array(X_img)
new_Y = np.array(y_label)

print(f"Forma final de X: {new_X.shape}")  # Debe ser (N, 128, 128, 3)
print(f"Forma final de Y: {new_Y.shape}")  # Debe ser (N,)

from collections import Counter
print(Counter(new_Y))

print("Generando dataset IID...")

import matplotlib.pyplot as plt

# Crear carpeta para guardar las gráficas si no existe
import os
output_dir = "/workspace/centralised/training_graphs"
os.makedirs(output_dir, exist_ok=True)
num_partitions = 1

indices = np.linspace(0, len(new_X), num_partitions + 1, dtype=int)
partitions = [
    np.arange(indices[i], indices[i + 1], dtype=int)
    for i in range(num_partitions)
]

# --- Visualización IID (igual estilo que Non-IID) ---
plt.figure(figsize=(8, 5))
counts_by_part = []
for i, idxs in enumerate(partitions):
    unique, cts = np.unique(new_Y[idxs], return_counts=True)
    counts_by_part.append(dict(zip(unique, cts)))

all_classes = np.unique(new_Y)
width = 0.8 / num_partitions
for i, cls in enumerate(all_classes):
    for j in range(num_partitions):
        count = counts_by_part[j].get(cls, 0)
        plt.bar(i + j * width, count, width=width, label=f'Client {j}' if i == 0 else "")

plt.xticks(np.arange(len(all_classes)), labels=['Accident', 'Dense Traffic', 'Meteorology', 'Sparse Traffic'])
plt.xlabel("Classes")
plt.ylabel("Number of examples")
plt.title("Class Distribution (IID)")
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'data_distribution.png'))
plt.close()

X_train_val, X_test, y_train_val, y_test = train_test_split(new_X, new_Y, test_size=0.15, random_state=42, stratify=new_Y)

X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.15/(0.85), random_state=42, stratify=y_train_val)

import numpy as np

# Mostrar porcentaje de cada clase en cada subconjunto
def mostrar_balance(y, nombre):
    valores, conteos = np.unique(y, return_counts=True)
    porcentajes = conteos / len(y)
    print(f"\nBalance de clases en {nombre}:")
    for v, p in zip(valores, porcentajes):
        print(f"Clase {v}: {p:.3f}")

mostrar_balance(y_train, "entrenamiento")
mostrar_balance(y_val, "validación")
mostrar_balance(y_test, "test")

# Arquitectura de la red neuronal

"""
def crear_modelo(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
"""

def crear_modelo(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        # Bloque 1: 32 filtros
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        
        # Bloque 2: 64 filtros
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Bloque 3: 128 filtros
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),

        # Bloque 4: 256 filtros
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.50),

        # Capas densas
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.50),
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.50),

        # Capa de salida
        tf.keras.layers.Dense(num_classes, activation='softmax')
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

# Crear el modelo
input_shape = X_train.shape[1:]
print(f"INPUT SHAPE: {input_shape}")
num_classes = len(label_encoder.classes_)
print(f"NUM CLASSES: {num_classes}")
model = crear_modelo(input_shape, num_classes)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    # ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    # ModelCheckpoint('mejor_modelo.h5', save_best_only=True)
]

# Entrenar el modelo
# batch_size = 64
batch_size = 32
epochs = 50

import numpy as np

print(np.unique(y_train))
print(np.unique(y_val))

print(f"Comenzando el entrenamiento del modelo... {epochs} epochs")

history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))

# Evaluar el modelo en el conjunto de validacion
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\nPrecisión en el conjunto de validacion: {val_acc}")

# Evaluar el modelo en el conjunto de test
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nPrecisión en el conjunto de prueba: {test_acc}")

y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

f1_score_test = f1_score(y_test, y_pred_labels, average='macro')
print(f"\nF1-Score en el conjunto de prueba: {f1_score_test:.4f}")

# Guardar el modelo entrenado
# model.save('modelo_categorias_aumentado_12epochs.h5')

# Plot: Loss during training and validation
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.axhline(y=1, color='r', linestyle='--', linewidth=1.5)
plt.title('Metrics')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.savefig(os.path.join(output_dir, 'training_metrics.png'))  # Save plot
plt.show()

# # Plot: Accuracy during training and validation
# plt.figure()
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Accuracy During Training and Validation')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.savefig(os.path.join(output_dir, 'training_validation_accuracy.png'))  # Save plot
# plt.show()

print(f"The graphs have been saved in the folder '{output_dir}'.")

elapsed = time.time() - start_time
elapsed_td = timedelta(seconds=int(elapsed))
with open(f"/workspace/centralised/Training_time.txt", 'w', encoding='utf-8') as archivo:
    archivo.write(f"Tiempo Total de entrenamiento: {elapsed_td}")
print(f"⏱ Tiempo total de entrenamiento: {elapsed_td}")

results_to_save = {
    "val_loss": val_loss,
    "val_acc": val_acc,
    "test_loss": test_loss,
    "test_acc": test_acc,
    "test_f1-score": f1_score_test
}

with open(f"/workspace/centralised/results.json", "w") as f:
    json.dump(results_to_save, f, indent=4)
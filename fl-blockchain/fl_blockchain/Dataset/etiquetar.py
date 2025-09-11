import os
import pandas as pd

def etiquetar_dataset(root_dir):
    """
    Etiqueta un dataset basado en las subcarpetas y guarda las etiquetas en un archivo CSV.
    
    :param root_dir: Directorio raíz que contiene las subcarpetas con los datos sin etiquetar.
    """
    # Lista para almacenar la información de las imágenes y sus etiquetas
    data = []

    # Recorrer todas las subcarpetas en el directorio raíz
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            # Etiqueta es el nombre de la carpeta
            label = folder_name
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path):
                    # Añadir información de la imagen y su etiqueta a la lista
                    data.append([file_path, label])

    # Crear un DataFrame de pandas
    df = pd.DataFrame(data, columns=['file_path', 'label'])

    # Guardar el DataFrame en un archivo CSV
    csv_path = os.path.join(root_dir, 'dataset_etiquetado.csv')
    df.to_csv(csv_path, index=False)

    print(f"Dataset etiquetado guardado en {csv_path}")

# Directorio raíz que contiene las subcarpetas con los datos sin etiquetar
root_dir = '/workspace/fl-blockchain/fl_blockchain/Dataset/tratado/'

# Llamar a la función para etiquetar el dataset
etiquetar_dataset(root_dir)

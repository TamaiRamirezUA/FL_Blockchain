import os
import pandas as pd

def etiquetar_dataset(root_dir):
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
                    file_path = f'Dataset/tratado/{folder_name}/{file_name}'
                    data.append([file_path, label])

    df = pd.DataFrame(data, columns=['file_path', 'label'])

    csv_path = os.path.join(root_dir, 'dataset_etiquetado.csv')
    df.to_csv(csv_path, index=False)

    print(f"Dataset etiquetado guardado en {csv_path}")

root_dir = 'tratado/'

etiquetar_dataset(root_dir)

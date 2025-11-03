import os
import shutil

# Carpetas principales
root_dirs = ["NC_4", "NC_6", "NC_8", "plot_all"]
# Carpeta de salida
output_dir = "/workspace/fl-blockchain/fl_blockchain/Plots"

# Crear la carpeta Plots si no existe
os.makedirs(output_dir, exist_ok=True)

# Recorrer cada carpeta raíz
for root_dir in root_dirs:
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".png"):
                # Ruta completa del archivo original
                src_path = os.path.join(dirpath, filename)
                
                # Partes de la ruta relativas a la carpeta raíz
                rel_path = os.path.relpath(dirpath, root_dir)
                
                # Crear el nuevo nombre según las reglas
                if rel_path == ".":
                    # El archivo está en la raíz de la carpeta
                    new_name = f"{root_dir}_{filename}"
                else:
                    # Está dentro de subcarpetas
                    subfolders = rel_path.replace(os.sep, "_")
                    new_name = f"{root_dir}_{subfolders}_{filename}"
                
                # Asegurarse de que no haya separadores dobles
                new_name = new_name.replace("__", "_")
                
                # Ruta de destino
                dst_path = os.path.join(output_dir, new_name)
                
                # Copiar el archivo
                shutil.copy2(src_path, dst_path)

print("✅ Archivos PNG copiados y renombrados en la carpeta 'Plots'.")


import os
import random
import shutil

# Configurações
source_folder = "."
output_folder = "./split"

# Proporções
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Criar pastas de destino
train_folder = os.path.join(output_folder, "train")
val_folder = os.path.join(output_folder, "val")
test_folder = os.path.join(output_folder, "test")

os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Listar todos os arquivos da pasta de origem
files = [
    f for f in os.listdir(source_folder)
    if os.path.isfile(os.path.join(source_folder, f))
]

# Embaralhar aleatoriamente
random.shuffle(files)

# Calcular quantidades
total = len(files)
train_count = int(total * train_ratio)
val_count = int(total * val_ratio)

# Separar arquivos
train_files = files[:train_count]
val_files = files[train_count:train_count + val_count]
test_files = files[train_count + val_count:]

# Copiar arquivos para as respectivas pastas
for f in train_files:
    shutil.copy2(
        os.path.join(source_folder, f),
        os.path.join(train_folder, f)
    )

for f in val_files:
    shutil.copy2(
        os.path.join(source_folder, f),
        os.path.join(val_folder, f)
    )

for f in test_files:
    shutil.copy2(
        os.path.join(source_folder, f),
        os.path.join(test_folder, f)
    )

# Mostrar resultados
print(f"Total de arquivos: {total}")
print(f"Train: {len(train_files)} ({len(train_files)/total*100:.1f}%)")
print(f"Val:   {len(val_files)} ({len(val_files)/total*100:.1f}%)")
print(f"Test:  {len(test_files)} ({len(test_files)/total*100:.1f}%)")


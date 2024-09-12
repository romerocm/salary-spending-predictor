import pandas as pd
import numpy as np
import joblib
import gdown
import os

# Asegurarse de que el directorio 'data' existe
os.makedirs('data', exist_ok=True)

# URL del archivo Predecir.csv en Google Drive
url = 'https://drive.google.com/file/d/1-QTtQokjC6UDXe9lD7qd12OMivPQ7t2V/view?usp=sharing'
file_id = url.split('/')[-2]
gdown.download(f'https://drive.google.com/uc?id={file_id}', 'data/Predecir.csv', quiet=False)

# Cargar los datos
df = pd.read_csv('data/Predecir.csv')

# Cargar el modelo entrenado
model = joblib.load('data/modelo_entrenado.joblib')

# Preparar los datos
X = df.drop(['Index'], axis=1)

# Generar predicciones
y_pred = model.predict(X)

# Asegurarse de que las predicciones sean no negativas
y_pred = np.maximum(y_pred, 0)

# Crear DataFrame con las predicciones
grupo_numero = "05"  # Cambia esto al número de tu grupo
resultado_df = pd.DataFrame({f'Pred_{grupo_numero}': y_pred})

# Guardar las predicciones en un archivo CSV
output_filename = f'data/Grupo_{grupo_numero}.csv'
resultado_df.to_csv(output_filename, index=False)

print(f"Se ha guardado el archivo '{output_filename}' con las predicciones.")
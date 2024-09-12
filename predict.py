import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import joblib
import gdown
import os

# URL del archivo en Google Drive
url = 'https://drive.google.com/file/d/1-QTtQokjC6UDXe9lD7qd12OMivPQ7t2V/view?usp=sharing'
file_id = url.split('/')[-2]
csv_filename = 'Predecir.csv'
gdown.download(f'https://drive.google.com/uc?id={file_id}', csv_filename, quiet=False)

# Cargar los datos
df = pd.read_csv(csv_filename)

# Separar variables predictoras
X = df.drop(['Index'], axis=1)

# Cargar el modelo entrenado
model = joblib.load('modelo_entrenado.joblib')

# Hacer predicciones
y_pred = model.predict(X)

# Crear el DataFrame con las predicciones
grupo_numero = "05"  # Cambia esto al número de tu grupo
resultado_df = pd.DataFrame({f'Pred_{grupo_numero}': y_pred})

# Guardar las predicciones en un archivo CSV
output_filename = f'Grupo_{grupo_numero}.csv'
resultado_df.to_csv(output_filename, index=False)

print(f"Se ha guardado el archivo '{output_filename}' con las predicciones.")

# Eliminar el archivo CSV descargado para limpiar
os.remove(csv_filename)
print(f"Se ha eliminado el archivo descargado '{csv_filename}'.")
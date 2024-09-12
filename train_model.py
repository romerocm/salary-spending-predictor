import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import gdown
import os

# Asegurarse de que el directorio 'data' existe
os.makedirs('data', exist_ok=True)

# URL del archivo de entrenamiento en Google Drive (necesitas proporcionar este enlace)
url = 'https://drive.google.com/file/d/11dqKxwZ3T_ObfqlURBv8RGqbHPWyjxYC/view?usp=sharing'
file_id = url.split('/')[-2]
gdown.download(f'https://drive.google.com/uc?id={file_id}', 'data/datos_entrenamiento.csv', quiet=False)

# Cargar los datos de entrenamiento
df = pd.read_csv('data/datos_entrenamiento.csv')

# Separar variables predictoras y variable objetivo
X = df.drop(['Spending', 'Index'], axis=1)
y = df['Spending']

# Definir transformadores para variables numéricas y categóricas
numeric_features = ['Age', 'Income', 'EducYears', 'HoursWPerWeek', 'PrevJobs', 'Distance', 'Credit', 'Savings', 'Expenses', 'Dependents', 'Health', 'YearJob']
categorical_features = ['Gender', 'Status', 'EducationLevel']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first', sparse=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Crear pipeline con RandomForestRegressor
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Guardar los datos de prueba para la visualización
X_test.to_csv('data/X_test.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)

# Ajustar el modelo
model.fit(X_train, y_train)

# Guardar el modelo entrenado
joblib.dump(model, 'data/modelo_entrenado.joblib')

print("Modelo entrenado y guardado como 'data/modelo_entrenado.joblib'")
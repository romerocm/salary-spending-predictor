import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import joblib
import gdown

# URL del archivo ProyectoSalarios.csv en Google Drive
url = 'https://drive.google.com/file/d/11dqKxwZ3T_ObfqlURBv8RGqbHPWyjxYC/view?usp=drive_link'
file_id = url.split('/')[-2]
gdown.download(f'https://drive.google.com/uc?id={file_id}', 'ProyectoSalarios.csv', quiet=False)

# Cargar los datos
df = pd.read_csv('ProyectoSalarios.csv')

# Separar variables predictoras y variable objetivo
X = df.drop(['Spending', 'Index'], axis=1)
y = df['Spending']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# Crear pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Ajustar el modelo
model.fit(X_train, y_train)

# Guardar el modelo entrenado
joblib.dump(model, 'modelo_entrenado.joblib')

print("Modelo entrenado y guardado como 'modelo_entrenado.joblib'")
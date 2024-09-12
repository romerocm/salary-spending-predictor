import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def visualize_results(X_test, y_test, model, output_dir='data'):
    # Asegurarse de que el directorio de salida y el subdirectorio de imágenes existen
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)

    # Hacer predicciones
    y_pred = model.predict(X_test)

    # Evaluar el modelo
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Error cuadrático medio: {mse}")
    print(f"R-cuadrado: {r2}")

    # Visualización de la importancia de las características
    if hasattr(model, 'named_steps') and hasattr(model.named_steps['regressor'], 'feature_importances_'):
        numeric_features = X_test.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X_test.select_dtypes(exclude=[np.number]).columns.tolist()
        
        feature_names = numeric_features + [f"{feat}_{cat}" for feat in categorical_features 
                                            for cat in model.named_steps['preprocessor'].named_transformers_['cat'].categories_[categorical_features.index(feat)][1:]]
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.named_steps['regressor'].feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)

        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
        plt.title('Top 15 Características más Importantes')
        plt.xlabel('Importancia')
        plt.ylabel('Característica')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'images', 'importancia_caracteristicas.png'))
        plt.close()

    # Gráfico de dispersión de valores reales vs predichos
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.title('Valores Reales vs Predicciones')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'images', 'valores_reales_vs_predicciones.png'))
    plt.close()

    # Gráfico de residuos
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicciones')
    plt.ylabel('Residuos')
    plt.title('Gráfico de Residuos')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'images', 'grafico_residuos.png'))
    plt.close()

    # Histograma de residuos
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuos')
    plt.title('Distribución de Residuos')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'images', 'distribucion_residuos.png'))
    plt.close()

    print("Se han guardado todas las visualizaciones en la carpeta 'data/images'.")

if __name__ == "__main__":
    # Cargar el modelo y los datos de prueba
    model = joblib.load('data/modelo_entrenado.joblib')
    
    # Cargar los datos de prueba (asumiendo que los guardaste durante el entrenamiento)
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv')

    # Generar visualizaciones
    visualize_results(X_test, y_test.values.ravel(), model)
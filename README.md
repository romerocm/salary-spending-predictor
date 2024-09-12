# Expense Prediction Model

This project implements a machine learning model to predict expenses based on various personal and financial factors. It uses Docker for easy setup and execution.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [Running the Project](#running-the-project)
4. [Outputs](#outputs)

## Prerequisites

- Docker
- Docker Compose

## Project Structure

```
project_directory/
├── data/
│   ├── images/
│   │   ├── importancia_caracteristicas.png
│   │   ├── valores_reales_vs_predicciones.png
│   │   ├── grafico_residuos.png
│   │   └── distribucion_residuos.png
│   ├── modelo_entrenado.joblib
│   ├── X_test.csv
│   ├── y_test.csv
│   └── Grupo_05.csv
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── train_model.py
├── predict.py
├── visualize_results.py
└── README.md
```

## Running the Project

1. Open a terminal and navigate to the project directory.
2. Run the following command to build and run the Docker containers:

   ```
   docker-compose up --build
   ```

   This command will:

   - Build the Docker image
   - Train the model
   - Make predictions
   - Generate visualizations

3. Once the process is complete, you can find the outputs in the `data/` directory.

## Outputs

The project generates the following outputs:

### Files

| File Name                      | Description                        |
| ------------------------------ | ---------------------------------- |
| `data/modelo_entrenado.joblib` | The trained machine learning model |
| `data/X_test.csv`              | Test set features                  |
| `data/y_test.csv`              | Test set target variable           |
| `data/Grupo_02.csv`            | Predictions for the provided data  |

### Visualizations

All visualizations are saved in the `data/images/` directory:

| Image Name                           | Description                                         |
| ------------------------------------ | --------------------------------------------------- |
| `importancia_caracteristicas.png`    | Bar plot showing the top 15 most important features |
| `valores_reales_vs_predicciones.png` | Scatter plot of real values vs. predictions         |
| `grafico_residuos.png`               | Scatter plot of residuals                           |
| `distribucion_residuos.png`          | Histogram of residuals                              |

### Console Output

The script will print the following metrics to the console:

- Mean Squared Error (MSE)
- R-squared (R²) score

These metrics provide an indication of the model's performance.

## Customization

To customize the model or prediction process:

1. Modify `train_model.py` to change the model parameters or use a different algorithm.
2. Adjust `predict.py` if you need to change how predictions are made or saved.
3. Update `visualize_results.py` to add or modify visualizations.

After making changes, rebuild and run the Docker containers using the command in the [Running the Project](#running-the-project) section.

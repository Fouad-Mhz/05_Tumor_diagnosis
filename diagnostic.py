import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
import time
from sklearn.ensemble import ExtraTreesClassifier

# Define paths for model loading
MODEL_DIR = os.environ['MODEL_DIR']
MODEL_NAME = os.environ['MODEL_NAME']
model_path = os.path.join(MODEL_DIR, MODEL_NAME)

# Load the best Extra Trees model (make sure it's already trained and saved)
best_model = joblib.load(model_path)

best_features = ['mean radius', 'mean texture', 'mean compactness',
       'mean concave points', 'radius error', 'worst radius', 'worst texture',
       'worst concavity', 'worst concave points']

# Load the breast cancer dataset
data = load_breast_cancer()

y = pd.DataFrame(data.target, columns=['diagnosis'])  # 0 for Malignant, 1 for Benign
x = pd.DataFrame(data.data, columns=data.feature_names)[best_features]

# Normalisation des données
scaler = StandardScaler()

# Create a DataFrame for scaled data with the same feature names
scaled_x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

# Function to make predictions for patients with indices
def predict_patients(patient_indices):
    # Diviser les données en entrainement et test
    start_time = time.time()

    # Select only the 'best_features' columns from the DataFrame
    patient_data_df = scaled_x.iloc[patient_indices]

    results = []  # Store results (Benign or Malignant), probabilities, and actual diagnosis for each patient

    for i, patient_data in enumerate(patient_data_df.values):
        # Predict for the patient
        prediction = best_model.predict([patient_data])
        probabilities = best_model.predict_proba([patient_data])[0]  # Probabilities for each class

        actual_diagnosis = "Malignant" if y.values[patient_indices[i]][0] == 0 else "Benign"
        result = {
            "Patient": "Patient {}".format(patient_indices[i] + 1),
            "Prediction": "Malignant" if prediction[0] == 0 else "Benign",
            "Proba_Benign": "{:.2%}".format(probabilities[1]),
            "Proba_Malignant": "{:.2%}".format(probabilities[0]),
            "Actual_Diagnosis": actual_diagnosis
        }
        results.append(result)

    results_df = pd.DataFrame(results)

    # Display the results using tabulate
    headers = results_df.columns
    print(tabulate(results_df, headers, tablefmt="fancy_grid"))

    print("-" * 40)  # Ligne de tirets pour la séparation

    # Temps d'entrainement
    end_time = time.time()  # Enregistrer le temps à la fin du processus
    elapsed_time = end_time - start_time  # Calculer le temps écoulé

    print("Temps d'exécution: {:.2f} secondes".format(elapsed_time))
    print("-" * 40)  # Ligne de tirets pour la séparation
    return results_df

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please enter patient IDs separated by commas as command-line arguments.")
    else:
        # Parse patient IDs from command-line arguments
        patient_ids = [int(patient_id) for patient_id in sys.argv[1].split(",")]

        # Check the validity of patient IDs and convert to 0-based indexing if needed
        valid_patient_ids = []
        for patient_id in patient_ids:
            patient_index = int(patient_id) - 1  # Convert to 0-based indexing
            if 0 <= patient_index < len(x):
                valid_patient_ids.append(patient_index)  # Add the 0-based index
            else:
                print(f"Patient number {patient_id} is out of range.")
        
        # Now, use valid_patient_ids to select patient data
        if valid_patient_ids:
            results_df = predict_patients(valid_patient_ids)
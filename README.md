# Tumor_diagnosis
This project aims to diagnose breast tumors as malignant (M) or benign (B) using machine learning techniques.

# Malignant and Benign Tumor Diagnostic Project

This project aims to diagnose breast tumors as malignant (M) or benign (B) using machine learning techniques. It includes a Jupyter notebook for data exploration, visualization, data preprocessing, model training, hyperparameter tuning, and a Docker application for deploying the trained model.

## Project Contents

The project contains the following elements:

1. **Notebook**: The notebook included in this project follows a multi-step methodology:

   - **Data Exploration**: The notebook starts by loading breast tumor data, examining the first entries to get a data overview. Next, exploratory data analysis is performed to understand the data structure, patient features, and tumor characteristics.

   - **Data Preprocessing**: This step involves preparing the data for modeling. It includes feature normalization, label encoding (Malignant -> 1, Benign -> 0), and dividing the data into training and testing sets. Additionally, Recursive Feature Elimination with Cross-Validation (RFECV) is used to determine the optimal number of features and the best features for a classification task using a RandomForestClassifier.

   - **Model Training**: Several machine learning models were trained, including 'Decision Tree', 'Extra Trees', 'Random Forest', 'Gradient Boosting', and 'XGBoost'. A 20-fold cross-validation was used to evaluate the performance of these models and determine the main model based on the scores obtained.

   - **Hyperparameter Tuning**: Hyperparameter tuning was performed with a 5-fold cross-validation and 10 iterations for the Extra Trees model using RandomizedSearchCV. The final model was evaluated in terms of accuracy, recall, F1-score, and the ROC curve.

2. **Dockerfile**: The Dockerfile is used to build a Docker image for deploying the trained model.

3. **Script `create_docker.sh`**: A script to create the Docker image from the Dockerfile.

4. **Script `launch_docker.sh`**: A script to launch the Docker container with the trained model.

5. **`diagnostic.py`**: A Python script that loads the trained model and makes predictions for tumor diagnosis.

## Usage Guide

Follow the steps below to use this project:

1. Install Docker on your system if not already installed.

2. Execute the `create_docker.sh` script to build the Docker image.

3. Execute the `launch_docker.sh` script to launch the Docker container with the trained model.

4. The application prompts you to enter a CSV file or manually enter patient IDs separated by commas. The program then displays the results for all patients in a table.

## Author

This project was created by Fouad Maherzi.


# Projet de Diagnostic de Tumeurs Malignes et Bénignes

Ce projet a pour objectif de diagnostiquer les tumeurs mammaires comme malignes (M) ou bénignes (B) en utilisant des techniques d'apprentissage automatique. Il inclut un notebook Jupyter pour explorer, visualiser les données, prétraiter les données, entraîner des modèles, ajuster les hyperparamètres, et une application Docker pour déployer le modèle entraîné.

## Contenu du Projet

Le projet contient les éléments suivants :

1. **Notebook** : Le notebook inclus dans ce projet suit une méthodologie en plusieurs étapes :

   - **Exploration des Données** : Le commence par charger les données de tumeurs mammaires, en examinant les premières entrées pour avoir un aperçu des données. Ensuite, une analyse exploratoire des données est effectuée pour comprendre la structure des données, les caractéristiques des patients, et les caractéristiques des tumeurs.

   - **Prétraitement des Données** : Cette étape consiste à préparer les données pour la modélisation. Elle inclut la normalisation des caractéristiques, l'encodage des étiquettes (Maligne -> 1, Bénigne -> 0), et la division des données en ensembles d'entraînement et de test. De plus, une Élimination Récursive des Caractéristiques avec Validation Croisée (RFECV) est utilisée pour déterminer le nombre optimal de caractéristiques et les meilleures caractéristiques pour une tâche de classification en utilisant un RandomForestClassifier.

   - **Entraînement du Modèle** : Plusieurs modèles d'apprentissage automatique ont été entraînés, notamment 'Decision Tree', 'Extra Trees', 'Random Forest', 'Gradient Boosting', et 'XGBoost'. Une validation croisée 20-fold a été utilisée pour évaluer les performances de ces modèles et déterminer le modèle principal en fonction des scores obtenus.

   - **Recherche des Hyperparamètres** : La recherche des hyperparamètres a été réalisée avec une validation croisée 5-fold et 10 itérations pour le modèle Extra Trees en utilisant RandomizedSearchCV. Le modèle final a été évalué en termes de précision, de rappel, de F1-score, et de courbe ROC.

2. **Dockerfile** : Le fichier Dockerfile permet de construire une image Docker pour le déploiement du modèle.

3. **Script `create_docker.sh`** : Un script pour créer l'image Docker à partir du Dockerfile.

4. **Script `launch_docker.sh`** : Un script pour lancer le conteneur Docker avec le modèle entraîné.

5. **`diagnostic.py`** : Un script Python qui charge le modèle entraîné et effectue des prédictions pour le diagnostic des tumeurs.

## Guide d'Utilisation

Suivez les étapes suivantes pour utiliser ce projet :

1. Installez Docker sur votre système si ce n'est pas déjà fait.

2. Exécutez le script `create_docker.sh` pour construire l'image Docker.

3. Exécutez le script `launch_docker.sh` pour lancer le conteneur Docker avec le modèle entraîné.

4. L'application demande d'entrer un fichier CSV ou de saisir manuellement les identifiants des patients séparés par des virgules. Le programme affiche ensuite les résultats de tous les patients dans un tableau.

## Auteur

Ce projet a été créé par Fouad Maherzi.

---

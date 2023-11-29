# Analyse du Data Set

!pip install numpy
!pip install pandas
!pip install scikit-learn
!pip install matplotlib
!pip install seaborn
!pip install scikit-learn
!pip install xgboost
!pip install joblib
!pip install tabulate

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
import joblib
import time
from tabulate import tabulate

data = load_breast_cancer()

y = pd.DataFrame(data.target, columns= ['diagnosis'])                      # M or B
x = pd.DataFrame(data.data, columns=data.feature_names)
df = x
df.head()

# feature names as a list
col = df.columns
print(col)

df.head()

df.describe()

df.isna().sum()

"""# 1.  Data Visualization"""

data.target_names

y2 = y.copy()
y2 = y2.replace({0: "M", 1: "B"})
y2.head()

y2.groupby('diagnosis').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.show()

sns.set(style="whitegrid", palette="muted")
data_dia = y
data2 = x
data_n_2 = (data2 - data2.mean()) / (data2.std())              # standardization
data2 = pd.concat([y2,data_n_2.iloc[:,0:10]],axis=1)            # pareil 10 20, 20 31
data2 = pd.melt(data2,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(30,10))
sns.swarmplot(x="features", y="value", hue="diagnosis", data=data2)

"""# 2.  Feature Selection"""

#correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(x.corr(), annot=True, linewidths=.5,fmt= '.1f',ax=ax)

"""**Voici les observations et les interprétations:**


1.   `Corrélation positive forte :`

  "mean radius," "mean perimeter," et "mean area" présentent des corrélations positives fortes (proches de 1.0) les unes avec les autres. Cela est attendu car ces caractéristiques sont liées à la taille de la tumeur.
  
  "mean compactness," "mean concavity," et "mean concave points" montrent
  également des corrélations positives fortes avec les autres. Cela suggère que ces caractéristiques sont liées et peuvent fournir des informations similaires sur la forme de la tumeur.


2.   `Corrélation négative forte :`

  "mean fractal dimension" est négativement corrélée avec de nombreuses caractéristiques, y compris "mean radius," "mean perimeter," "mean area," et "mean smoothness." Cela indique que les tumeurs ayant une dimension fractale moyenne plus élevée ont tendance à être de plus petite taille et moins lisses.


3.  `Corrélation positive modérée :`
  "mean texture" et "mean smoothness" ont une corrélation positive modérée (d'environ 0,33). Cela suggère que les tumeurs avec des valeurs de texture moyenne plus élevées ont également tendance à avoir des valeurs de lissage moyennes plus élevées.

4. `Corrélation négative modérée :`

  "mean texture" et "mean radius" ont une corrélation négative modérée (d'environ -0,32), ce qui signifie que les tumeurs avec des valeurs de texture moyenne plus élevées ont tendance à avoir des rayons moyens légèrement plus petits.

`À conserver :`

"mean radius" (caractéristique centrale)

"mean compactness" (caractéristique centrale)

"mean concave points" (caractéristique centrale)

"radius error" (caractéristique centrale)

"worst radius" (caractéristique centrale)



"""

drop_list = [
    "mean perimeter",
    "mean area",
    "mean concavity",
    "perimeter error",
    "area error",
    "worst perimeter",
    "worst area"
]

x1 = x.drop(drop_list, axis=1)
x1

"""

> Nous allons utiliser la méthode d'Élimination Récursive des Caractéristiques avec Validation Croisée (RFECV) pour déterminer le nombre optimal de caractéristiques et les meilleures caractéristiques pour une tâche de classification en utilisant un RandomForestClassifier.

"""

from sklearn.decomposition import PCA

# Normalisation des données
scaler = StandardScaler()
scaled_data = scaler.fit_transform(x)

# Création de l'objet PCA
pca = PCA()
pca.fit(scaled_data)

# Variance expliquée cumulée
explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# Courbe de la variance expliquée cumulée
plt.plot(explained_variance_ratio)
plt.xlabel("Nombre de composantes")
plt.ylabel("Variance expliquée cumulée")
plt.title("Variance expliquée cumulée")

# Courbe du coude
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o')
plt.xlabel("Nombre de composantes")
plt.ylabel("Variance expliquée cumulée")
plt.title("Courbe du coude (Elbow Method)")

plt.show()

# Identification du nombre optimal de dimensions
threshold = 0.95
optimal_n_components = np.where(explained_variance_ratio >= threshold)[0][0] + 1
print("Le nombre optimal de dimensions est :", optimal_n_components)

from sklearn.feature_selection import RFECV
Y = y['diagnosis']
# split data train 80 % and test 20 %
x_train, x_test, y_train, y_test = train_test_split(x1, Y, test_size=0.2, random_state=42)

# The "accuracy" scoring is proportional to the number of correct classifications
clf_rf = RandomForestClassifier()
rfecv = RFECV(estimator=clf_rf, step=1, cv=5,scoring='accuracy')   #5-fold cross-validation
rfecv = rfecv.fit(x_train, y_train)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', x_train.columns[rfecv.support_])

"""# 3. Model Selection

"""

Y = y['diagnosis']
le = LabelEncoder()
le.fit(Y)
y_encoded = le.transform(Y)
y_encoded

def models(X, y):
  models = {
      'Dummy': DummyClassifier(strategy='most_frequent'),
      'Decision Tree': DecisionTreeClassifier(),
      'Extra Trees': ExtraTreesClassifier(),
      'Random Forest': RandomForestClassifier(),
      'Gradient Boosting': GradientBoostingClassifier(),
      'XGBoost': XGBClassifier(),
  }

  results = []

  for name, model in models.items():
      start_time = time.time()

      scores = cross_val_score(model, X, y, cv=20)
      elapsed_time = time.time() - start_time

      # Formater les valeurs avec trois chiffres après la virgule
      mean_score = format(np.mean(scores) * 100, ".2f")
      std_deviation = format(np.std(scores) * 100, ".2f")
      elapsed_time = format(elapsed_time, ".2f")

      results.append([name, mean_score, elapsed_time, std_deviation])

  # Trier les résultats par score moyen
  sorted_results = sorted(results, key=lambda x: float(x[1]), reverse=True)

  # Enregistrer les 3 meilleurs modèles
  top_models = sorted_results[:3]
  for model_info in top_models:
      model_name = model_info[0]
      model_instance = models[model_name]
      joblib.dump(model_instance, f'{model_name}_model.pkl')

  headers = ["Model", "Mean Score (%)", "Time Elapsed (s)", "Std Deviation (%)"]
  print(tabulate(results, headers, tablefmt="pretty"))

  return  results, headers

results_X, headers_X = models(x, y_encoded)

results_x1, headers_x1 = models(x1, y_encoded)

from sklearn.preprocessing import RobustScaler
# Normalisation des données
scaler = RobustScaler()
scaled_x1 = scaler.fit_transform(x1)
results, headers = models(scaled_x1, y_encoded)

x2= x[x_train.columns[rfecv.support_]]
results_x2, headers_x2 = models(x2, y_encoded)

# Normalisation des données
scaler = StandardScaler()
scaled_x2 = scaler.fit_transform(x2)
results, headers = models(scaled_x2, y_encoded)

# @title
"""import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from tabulate import tabulate
import matplotlib.pyplot as plt
from joblib import dump

def random_forest_tuning(X, y):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Define hyperparameters for Random Forest
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [ 'sqrt', 'log2']
    }

    # Perform RandomizedSearchCV
    start_time = time.time()
    random_search = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=param_grid, n_iter=10, cv=5, scoring='accuracy')
    random_search.fit(X_train, y_train)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time elapsed for RandomizedSearchCV training: {:.2f} seconds".format(elapsed_time))
    print("-" * 40)

    # Get the best hyperparameters and the best model
    best_params = random_search.best_params_
    best_model = random_search.best_estimator_
    best_score = random_search.best_score_

    best_model.fit(X_train, y_train)
    predictions = best_model.predict(X_test)

    end_time2 = time.time()
    elapsed_time = end_time2 - end_time
    print("Time elapsed for Best Model training: {:.2f} seconds".format(elapsed_time))
    print("-" * 40)


    # Save the trained model to a file
    dump(best_model, 'best_sklearn_model.joblib')


    print("Best hyperparameters:")
    print(best_params)
    print("-" * 40)

    print("Score achieved with the best hyperparameters: {:.3f}".format(best_score))
    print("Score on test data: {:.4f}".format(best_model.score(X_test, y_test)))
    print("-" * 40)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, predictions)
    print("Confusion Matrix:")
    print(conf_matrix)
    print("-" * 40)

    # Precision, Recall, and F1 Score Table
    class_report = classification_report(y_test, predictions, output_dict=True)
    class_report_df = pd.DataFrame(class_report).T
    print("Precision, Recall, and F1 Score Table:")
    print(tabulate(class_report_df, headers='keys', tablefmt='fancy_grid'))
    print("-" * 40)

    # AUC-ROC
    auc_roc = roc_auc_score(y_test, predictions)
    print("AUC-ROC:")
    print(auc_roc)
    print("-" * 40)

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_roc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Create a DataFrame to store the results
    results = pd.DataFrame(random_search.cv_results_)
    return results, predictions, best_model, class_report, auc_roc, conf_matrix

# Example usage
rf_results, rf_predictions, rf_best_model, rf_class_report, rf_auc_roc, rf_conf_matrix = random_forest_tuning(x, y['diagnosis'])
"""

# @title
import xgboost as xgb
# Tuning XGBoost
xgb_model = xgb.XGBClassifier()

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.5],
}

performance_list = []

def xgb_tuning(X, y):

    # Diviser les données en entrainement et test
    start_time = time.time()

    # Diviser les données en entrainement et test 90/10
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Effectuer la recherche d'hyperparamètres avec RandomizedSearchCV
    random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_grid, n_iter=10, cv=5, scoring='accuracy')
    random_search.fit(X_train, y_train)

    # Temps d'entrainement
    end_time = time.time()  # Enregistrer le temps à la fin du processus
    elapsed_time = end_time - start_time  # Calculer le temps écoulé

    print("Temps écoulé  pour l'entrainement du RandomizedSearchCV: {:.2f} secondes".format(elapsed_time))
    print("-" * 40)  # Ligne de tirets pour la séparation

    # Obtenir les meilleurs hyperparamètres et le meilleur modèle
    best_params = random_search.best_params_
    best_model = random_search.best_estimator_
    best_score = random_search.best_score_

    # Entrainer le modèle sur l'ensemble d'entrainement
    best_model.fit(X_train, y_train)
    predictions = best_model.predict(X_test)  # Prendre des prédictions sur l'ensemble de test

    # Temps d'entrainement
    end_time2 = time.time()  # Enregistrer le temps à la fin du processus
    elapsed_time = end_time2 - end_time  # Calculer le temps écoulé

    print("Temps écoulé  pour l'entrainement du Best Model: {:.2f} secondes".format(elapsed_time))
    print("-" * 40)  # Ligne de tirets pour la séparation

    # Sauvegardez le modèle dans un fichier
    best_model.save_model('best_xgboost_model.model')

    # Afficher les meilleurs hyperparamètres
    print("Meilleurs hyperparamètres :")
    print(best_params)

    print("-" * 40)  # Ligne de tirets pour la séparation

    print("Score obtenu avec les meilleurs hyperparamètres : {:.3f}".format(best_score))
    print("Score sur les données de test : {:.4f}".format(best_model.score(X_test, y_test)))

    print("-" * 40)  # Ligne de tirets pour la séparation


    # Matrice de confusion
    conf_matrix = confusion_matrix(y_test, predictions)
    print("Matrice de confusion :")
    print(conf_matrix)

    print("-" * 40)  # Ligne de tirets pour la séparation

    # Table de précision, rappel et score F1
    class_report = classification_report(y_test, predictions, output_dict=True)
    class_report_df = pd.DataFrame(class_report).T
    print("Table de précision, rappel et score F1 :")
    print(tabulate(class_report_df, headers='keys', tablefmt='fancy_grid'))
    print("-" * 40)  # Ligne de tirets pour la séparation

    # AUC-ROC (mesure d'efficacité des classifieurs d'entropie)
    auc_roc = roc_auc_score(y_test, predictions)
    print("AUC-ROC :")
    print(auc_roc)

    print("-" * 40)  # Ligne de tirets pour la séparation

    # Tracé de la courbe ROC
    fpr, tpr, thresholds = roc_curve(y_test, best_model.predict_proba(X_test)[:,1])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_roc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


    # Créer un DataFrame pour stocker les résultats
    results = pd.DataFrame(random_search.cv_results_)
    return results, predictions, best_model, class_report, auc_roc, conf_matrix

# @title
results, predictions, best_model, class_report, auc_roc, conf_matrix = xgb_tuning(scaled_x2, y_encoded)

# Normalisation des données
scaler = StandardScaler()
scaled_x = scaler.fit_transform(x)
scaled_x

import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import pandas as pd
from tabulate import tabulate

best_features = ['mean radius', 'mean texture', 'mean compactness',
       'mean concave points', 'radius error', 'worst radius', 'worst texture',
       'worst concavity', 'worst concave points']

# Load the breast cancer dataset
data = load_breast_cancer()
y = pd.DataFrame(data.target, columns=['diagnosis'])  # 0 for Malignant, 1 for Benign
x = pd.DataFrame(data.data, columns=data.feature_names)[best_features]

# Define the ExtraTreesClassifier model
et_model = ExtraTreesClassifier()

param_grid = {
    'n_estimators': [300, 500],
    'max_depth': [15, 20],
    'min_samples_split': [2, 3],
    'min_samples_leaf': [2, 3],
}


performance_list = []

def et_tuning(X, y):
    # Split the data into training and testing
    start_time = time.time()

    # Normalisation des données
    scaler = StandardScaler()

    # Create a DataFrame for scaled data with the same feature names
    scaled_x = pd.DataFrame(scaler.fit_transform(X))

    X_train, X_test, y_train, y_test = train_test_split(scaled_x, y, test_size=0.1, random_state=42)

    # Perform hyperparameter tuning with RandomizedSearchCV
    random_search = RandomizedSearchCV(estimator=et_model, param_distributions=param_grid, n_iter=10, cv=5, scoring='accuracy', verbose=1)
    random_search.fit(X_train, y_train)

    # Training time
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("Time elapsed for RandomizedSearchCV training: {:.2f} seconds".format(elapsed_time))
    print("-" * 40)

    # Get the best hyperparameters and the best model
    best_params = random_search.best_params_
    best_model = random_search.best_estimator_
    best_score = random_search.best_score_

    # Train the model on the training set
    best_model.fit(X_train, y_train)
    predictions = best_model.predict(X_test)

    # Training time
    end_time2 = time.time()
    elapsed_time = end_time2 - end_time

    print("Time elapsed for Best Model training: {:.2f} seconds".format(elapsed_time))
    print("-" * 40)

    # Save the model to a file
    joblib.dump(best_model, 'tuned_extra_trees_model.pkl')

    # Display the best hyperparameters
    print("Best hyperparameters:")
    print(best_params)
    print("-" * 40)

    # Display the score obtained with the best hyperparameters
    print("Score obtained with the best hyperparameters: {:.3f}".format(best_score))
    print("Score on the test data: {:.4f}".format(best_model.score(X_test, y_test)))
    print("-" * 40)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, predictions)
    print("Confusion matrix:")
    print(conf_matrix)
    print("-" * 40)

    # Precision, recall, and F1-score table
    class_report = classification_report(y_test, predictions, output_dict=True)
    class_report_df = pd.DataFrame(class_report).T
    print("Precision, recall, and F1-score table:")
    print(tabulate(class_report_df, headers='keys', tablefmt='fancy_grid'))
    print("-" * 40)

    # AUC-ROC
    auc_roc = roc_auc_score(y_test, predictions)
    print("AUC-ROC:")
    print(auc_roc)
    print("-" * 40)

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_roc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Create a DataFrame to store the results
    results = pd.DataFrame(random_search.cv_results_)
    return results, predictions, best_model, class_report, auc_roc, conf_matrix

results, predictions, best_model, class_report, auc_roc, conf_matrix = et_tuning(x[best_features], y_encoded)

"""`Matrice de Confusion :`

Sur 57 patients, Le modèle a :

1.   Correctement classé 16 exemples comme positifs (classe 1, B) et 39 exemples négatifs (classe 0, M)
2.   Commis 1 erreurs en classant à tort des exemples négatifs (classe 0, M) comme positifs.
3.   Commis 1 erreurs en classant à tort des exemples positifs comme négatifs.

Ces résultats montrent qu'il peut bien distinguer entre les deux classes.

`Table de Précision, Rappel et Score F1 :`

1.     Pour la classe 0, la précision est de 94,11 %, ce qui signifie que la plupart des prédictions positives sont correctes.
2.     Le rappel pour la classe 1 est de 97.5 %, ce qui indique qu'il parvient à identifier la plupart des exemples réellement positifs.
3.     Le score F1 est élevé, ce qui montre qu'il parvient à trouver un équilibre entre la précision et le rappel.

Ces métriques reflètent la capacité du modèle à bien classer les données.

`AUC-ROC :`
  
*     L'AUC-ROC est de 0.9580, ce qui signifie que le modele est capable de bien discriminer entre les classes. Une valeur proche de 1 indique une excellente capacité de discrimination.




En résumé, je montre une performance solide en termes de précision, de rappel et de score F1. Le AUC-ROC confirme la capacité du modèle à bien discriminer les classes.

# Application
"""

import sys
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

best_model = joblib.load('/content/tuned_extra_trees_model.pkl')

best_features = ['mean radius', 'mean texture', 'mean compactness',
       'mean concave points', 'radius error', 'worst radius', 'worst texture',
       'worst concavity', 'worst concave points']

# Load the breast cancer dataset
data = load_breast_cancer()

y = pd.DataFrame(data.target, columns=['diagnosis'])  # 0 for Malignant, 1 for Benign
x = pd.DataFrame(data.data, columns=data.feature_names)[best_features]


# Function to make predictions for patients with a indices
def predict_patients(*patient_indices):

    # Diviser les données en entrainement et test
    start_time = time.time()

    # Normalisation des données
    scaler = StandardScaler()

    # Create a DataFrame for scaled data with the same feature names
    scaled_x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

    # Select only the 'best_features' columns from the DataFrame
    patient_data_df = scaled_x.iloc[list(patient_indices)]

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

# User input as a string
user_input = "17,18,19,20,21,22" # Replace with the user input

# Convert the string to a list of integers
patient_indices = [int(x) for x in user_input.split(',')]

# Call the function with the patient indices
results = predict_patients(*patient_indices)

# Function to make predictions for patients randomely
def random_predict():
    # Générer un nombre aléatoire de patients entre 1 et 20
    num_patients = np.random.randint(1, 26)

    # Sélectionner un échantillon aléatoire de patients à partir de votre jeu de données 'x'
    selected_patients = x.sample(n=num_patients)


    # Select only the best features columns from the DataFrame
    patient_data_df = selected_patients[best_features]

    results = []  # Store results (Benign or Malignant) and probabilities for each patient

    for i in range(len(patient_data_df)):
        patient_data = patient_data_df.iloc[i]  # Get patient data as a Series

        # Predict for the patient
        prediction = best_model.predict([patient_data])
        probabilities = best_model.predict_proba([patient_data])[0]  # Probabilities for each class
        actual_diagnosis = "Malignant" if y.values[selected_patients.index[i]][0] == 0 else "Benign"

        result = {
            "Patient": "Patient {}".format(selected_patients.index[i] + 1),
            "Result": "Malignant" if prediction[0] == 0 else "Benign",
            "Proba_Benign": "{:.2%}".format(probabilities[1]),
            "Proba_Malignant": "{:.2%}".format(probabilities[0]),
            "Actual_Diagnosis": actual_diagnosis
        }
        results.append(result)

    # Create a table from the results
    headers = results[0].keys()
    table = [result.values() for result in results]
    formatted_table = tabulate(table, headers, tablefmt="fancy_grid")
    print(formatted_table)

# Example usage
# random_predict()"""

# https://www.kaggle.com/code/kanncaa1/feature-selection-and-data-visualization
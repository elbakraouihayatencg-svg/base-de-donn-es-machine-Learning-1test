# ZINEB EL MEJDOUBI
*Numéro d'étudiant* : 24010156
*Classe* : CAC2

# Rapport Final d'Analyse Prédictive (Code, Explications et Interprétations)

## Introduction
Ce rapport détaille une analyse prédictive complète réalisée sur un jeu de données économiques simulées. L'objectif était de suivre un pipeline d'analyse standard, du prétraitement des données à l'évaluation et l'interprétation des modèles, afin de comprendre les facteurs influençant le PIB par habitant (`GDP_per_Capita`).

## 1. Préparation et Analyse Exploratoire des Données (EDA)

### 1.1 Chargement et Nettoyage des Données
En raison de problèmes persistants de chargement du jeu de données original depuis l'URL GitHub, un **jeu de données factice** a été créé. Ce jeu de données simulé a permis de poursuivre l'analyse. Les doublons ont été vérifiés (et aucun n'a été trouvé), et les types de données se sont avérés cohérents.

**Code de création du DataFrame factice et vérification initiale:**
```python
import pandas as pd
import numpy as np

data = {
    'Country': ['USA', 'China', 'Japan', 'Germany', 'India', 'UK', 'USA', 'China', 'Japan', 'Germany'],
    'Year': [2010, 2010, 2010, 2010, 2010, 2010, 2011, 2011, 2011, 2011],
    'GDP_Billions': [14.9, 6.0, 5.7, 3.3, 1.7, 2.7, 15.5, np.nan, 6.1, 3.5],
    'Population_Millions': [309.3, 1337.8, 128.1, 81.7, 1234.0, np.nan, 311.6, 1344.1, 127.8, 81.8],
    'Inflation_Rate': [1.6, 3.3, -0.7, 1.2, 10.5, 3.3, 3.2, 5.4, np.nan, 2.1],
    'Unemployment_Rate': [9.6, 4.1, 5.1, 7.0, 3.5, 7.9, 8.9, np.nan, 4.5, 5.9],
    'Trade_Balance_Billions': [-500.0, 180.0, 170.0, 200.0, -100.0, -50.0, -550.0, 200.0, 180.0, 210.0],
    'Continent': ['North America', 'Asia', 'Asia', 'Europe', 'Asia', np.nan, 'North America', 'Asia', 'Asia', 'Europe'],
    'Economic_Status': ['Developed', 'Developing', 'Developed', 'Developed', 'Developing', 'Developed', 'Developed', 'Developing', 'Developed', 'Developed']
}
df = pd.DataFrame(data)

print("Original DataFrame info before encoding:")
df.info()
print("\nFirst 5 rows of the DataFrame:")
display(df.head())
```

**Interprétation :** Le DataFrame a été créé avec des valeurs simulées et quelques valeurs manquantes pour tester les étapes d'imputation. `df.info()` confirme les types de données et le nombre d'entrées (10).

### 1.2 Imputation des Valeurs Manquantes
Les valeurs manquantes ont été imputées en utilisant la médiane pour les colonnes numériques et le mode pour les colonnes catégorielles. Cette approche est robuste et évite la perte de données.

**Code d'imputation des valeurs manquantes :**
```python
print("Missing values before imputation:")
print(df.isnull().sum())

# Impute numerical columns with median
for column in df.select_dtypes(include=['number']).columns:
    if df[column].isnull().any():
        median_value = df[column].median()
        df[column].fillna(median_value, inplace=True)
        print(f"Imputed missing values in numerical column '{column}' with median: {median_value}")

# Impute categorical columns with mode
for column in df.select_dtypes(include=['object']).columns:
    if df[column].isnull().any():
        mode_value = df[column].mode()[0]
        df[column].fillna(mode_value, inplace=True)
        print(f"Imputed missing values in categorical column '{column}' with mode: {mode_value}")

print("\nMissing values after imputation:")
print(df.isnull().sum())
```

**Interprétation :** Toutes les valeurs manquantes ont été traitées. La médiane a été utilisée pour `GDP_Billions`, `Population_Millions`, `Inflation_Rate`, `Unemployment_Rate`, et le mode pour `Continent`. Le résultat montre qu'il n'y a plus de valeurs manquantes dans le DataFrame.

### 1.3 Encodage des Variables Catégorielles
Les colonnes catégorielles (`Country`, `Continent`, `Economic_Status`) ont été encodées via One-Hot Encoding pour les rendre utilisables par les modèles de machine learning.

**Code d'encodage des variables catégorielles :**
```python
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print("\nDataFrame info after encoding:")
df.info()
display(df.head())
```

**Interprétation :** Les colonnes catégorielles ont été transformées en multiples colonnes binaires (True/False). `drop_first=True` a été appliqué pour éviter la multicolinéarité parfaite, réduisant ainsi le nombre total de colonnes booléennes.

### 1.4 Normalisation des Données Numériques
Les caractéristiques numériques ont été mises à l'échelle en utilisant `StandardScaler` pour assurer une échelle comparable entre elles, ce qui est bénéfique pour de nombreux algorithmes.

**Code de normalisation des données numériques :**
```python
from sklearn.preprocessing import StandardScaler

numerical_cols_to_scale = ['GDP_Billions', 'Population_Millions', 'Inflation_Rate', 'Unemployment_Rate', 'Trade_Balance_Billions']
scaler = StandardScaler()
df[numerical_cols_to_scale] = scaler.fit_transform(df[numerical_cols_to_scale])

print("\nDataFrame info after scaling:")
df.info()
display(df.head())
```

**Interprétation :** Les valeurs des colonnes numériques sélectionnées sont maintenant centrées autour de 0 avec un écart type de 1, comme le montrent les premières lignes du DataFrame.

### 1.5 Ingénierie des Caractéristiques
Deux nouvelles caractéristiques ont été créées : `GDP_per_Capita` et `Economic_Burden_Index`.

**Code d'ingénierie des caractéristiques :**
```python
df['GDP_per_Capita'] = df['GDP_Billions'] / df['Population_Millions']
df['Economic_Burden_Index'] = df['Inflation_Rate'] * df['Unemployment_Rate']

print("\nFirst 5 rows of the DataFrame with new engineered features:")
display(df.head())
```

**Interprétation :** `GDP_per_Capita` mesure la production économique par personne, normalisant le PIB par la population. `Economic_Burden_Index` combine l'inflation et le chômage pour évaluer l'instabilité économique générale. Ces caractéristiques enrichissent l'analyse.

### 1.6 Visualisation des Distributions et Analyse de Corrélation
Des visualisations (histogrammes et boîtes à moustaches) ont été créées pour les variables numériques afin d'analyser leurs distributions et de détecter les valeurs aberrantes. Une matrice de corrélation a été utilisée pour identifier les relations entre les variables.

**Code de visualisation (exemple pour une colonne):**
```python
import matplotlib.pyplot as plt
import seaborn as sns

numerical_cols_final = numerical_cols_to_scale # Assuming numerical_cols_to_scale is still relevant for visualization
# Example for GDP_Billions
column = 'GDP_Billions'
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
sns.histplot(df[column], kde=True)
plt.title(f'Distribution of {column}')
plt.subplot(1, 2, 2)
sns.boxplot(y=df[column])
plt.title(f'Box Plot of {column}')
plt.tight_layout()
plt.show()

# Code de la matrice de corrélation
correlation_matrix = df[numerical_cols_final + ['GDP_per_Capita', 'Economic_Burden_Index']].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Numerical Features')
plt.show()
```

**Interprétation :**
*   **Distributions :** `GDP_Billions` et `Population_Millions` présentaient une asymétrie positive. Les autres variables montraient des distributions variées, avec quelques valeurs aberrantes potentielles.
*   **Corrélations :** Forte corrélation positive entre `GDP_Billions` et `Population_Millions`. `Trade_Balance_Billions` était positivement corrélé avec `GDP_per_Capita`, tandis que `Inflation_Rate` montrait une corrélation négative avec `GDP_per_Capita`.

## 2. Modélisation et Évaluation

### 2.1 Division des Données
Le jeu de données a été divisé en ensembles d'entraînement et de test.

**Code de division des données :**
```python
from sklearn.model_selection import train_test_split

y = df['GDP_per_Capita']
X = df.drop(columns=['GDP_per_Capita', 'Year'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")
```

**Interprétation :** Le jeu de données a été divisé en un ensemble d'entraînement de 8 échantillons et un ensemble de test de 2 échantillons. Cette taille extrêmement réduite du jeu de test est une limitation majeure pour la robustesse des évaluations.

### 2.2 Entraînement et Comparaison Initiale des Modèles
Trois modèles de régression ont été entraînés : Régression Linéaire, RandomForestRegressor et SVR.

**Code d'entraînement et de prédiction initiaux :**
```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

lr_model = LinearRegression()
rf_model = RandomForestRegressor(random_state=42)
svr_model = SVR()

lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
svr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)
svr_predictions = svr_model.predict(X_test)

lr_mse = mean_squared_error(y_test, lr_predictions)
lr_r2 = r2_score(y_test, lr_predictions)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)
svr_mse = mean_squared_error(y_test, svr_predictions)
svr_r2 = r2_score(y_test, svr_predictions)

print(f"\nLinear Regression Model Performance: MSE={lr_mse:.4f}, R2 Score={lr_r2:.4f}")
print(f"Random Forest Regressor Model Performance: MSE={rf_mse:.4f}, R2 Score={rf_r2:.4f}")
print(f"SVR Model Performance: MSE={svr_mse:.4f}, R2 Score={svr_r2:.4f}")
```

**Interprétation :** Tous les modèles ont montré des performances très faibles, avec des scores R2 négatifs, indiquant qu'ils sont moins performants qu'un modèle prédisant simplement la moyenne.

### 2.3 Cross-Validation et Optimisation des Hyperparamètres
`GridSearchCV` a été utilisé pour optimiser les hyperparamètres de `RandomForestRegressor` et `SVR`.

**Code d'optimisation des hyperparamètres :**
```python
from sklearn.model_selection import GridSearchCV

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_features': ['sqrt', 'log2', 1.0],
    'min_samples_split': [2, 5, 10]
}

param_grid_svr = {
    'kernel': ['rbf', 'linear'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto']
}

grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

grid_search_svr = GridSearchCV(estimator=svr_model, param_grid=param_grid_svr, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_svr.fit(X_train, y_train)

print("\nBest parameters for RandomForestRegressor:", grid_search_rf.best_params_)
print("Best cross-validation score for RandomForestRegressor:", grid_search_rf.best_score_)
print("\nBest parameters for SVR:", grid_search_svr.best_params_)
print("Best cross-validation score for SVR:", grid_search_svr.best_score_)
```

**Interprétation :** `GridSearchCV` a trouvé les meilleures combinaisons d'hyperparamètres pour chaque modèle en se basant sur une validation croisée. Les scores de validation croisée sont également négatifs, ce qui confirme la difficulté de la tâche.

### 2.4 Évaluation des Modèles Optimisés sur l'Ensemble de Test
Les modèles optimisés ont été réévalués sur l'ensemble de test.

**Code d'évaluation des modèles optimisés :**
```python
best_rf_model = grid_search_rf.best_estimator_
best_svr_model = grid_search_svr.best_estimator_

best_rf_predictions = best_rf_model.predict(X_test)
best_svr_predictions = best_svr_model.predict(X_test)

best_rf_mse = mean_squared_error(y_test, best_rf_predictions)
best_rf_r2 = r2_score(y_test, best_rf_predictions)
best_svr_mse = mean_squared_error(y_test, best_svr_predictions)
best_svr_r2 = r2_score(y_test, best_svr_predictions)

print(f"\nOptimized Random Forest Regressor Model Performance: MSE={best_rf_mse:.4f}, R2 Score={best_rf_r2:.4f}")
print(f"Optimized SVR Model Performance: MSE={best_svr_mse:.4f}, R2 Score={best_svr_r2:.4f}")
print(f"Linear Regression Model Performance (for comparison): MSE={lr_mse:.4f}, R2 Score={lr_r2:.4f}")
```

**Interprétation :** Les scores R2 restent négatifs pour les modèles optimisés, même si le MSE peut varier. Le modèle de régression linéaire initial conserve le meilleur (moins négatif) score R2 et le plus faible MSE, suggérant que l'optimisation n'a pas significativement amélioré la capacité prédictive avec si peu de données.

## 3. Sélection et Interprétation du Meilleur Modèle

Malgré les performances globalement médiocres (scores R2 négatifs), le **modèle de Régression Linéaire** a été sélectionné comme le "meilleur" en raison de son **erreur quadratique moyenne (MSE) la plus faible (0.0231)** et de son **score R2 le moins négatif (-1.6344)**.

**Code pour afficher les coefficients du modèle de Régression Linéaire :**
```python
print("Feature Coefficients for Linear Regression Model:")
for feature, coef in zip(X.columns, lr_model.coef_):
    print(f"  {feature}: {coef:.4f}")
```

**Interprétation des Coefficients du Modèle de Régression Linéaire :**
*   **`Trade_Balance_Billions` (0.8097)** : Fortement associé positivement au PIB par habitant, ce qui est économiquement intuitif.
*   **`Country_UK` (1.4069)** : A le plus grand impact positif parmi les pays, suggérant une association avec un PIB par habitant plus élevé.
*   **`Inflation_Rate` (-0.1790)** : Associé négativement au PIB par habitant, en accord avec la théorie économique.
*   **`Country_Japan` (-0.5842)** et **`Country_India` (-0.4713)** : Associés négativement à un PIB par habitant plus faible.
*   **`GDP_Billions` (-1.1701)** : Coefficient négatif qui semble contre-intuitif. Cela est probablement un artéfact de la forte multicolinéarité avec `Population_Millions` et `GDP_per_Capita` elle-même, ainsi que de la petite taille de l'échantillon.
*   **`Economic_Burden_Index` (0.0256)** : Impact très faible, potentiellement dû au bruit des données ou aux interactions complexes.

## Conclusion et Recommandations

### Bilan des Résultats
L'analyse a permis de traverser toutes les étapes d'un pipeline de machine learning. Cependant, la **limitation majeure** est la **taille extrêmement petite du jeu de données (10 échantillons au total)**. Cela a conduit à des scores R2 négatifs pour tous les modèles, indiquant une incapacité à fournir des prédictions fiables. Le modèle de Régression Linéaire, malgré ses faibles performances, a été identifié comme le meilleur comparé aux autres.

### Limites Critiques
1.  **Taille du Jeu de Données** : La faiblesse prédictive est directement attribuable au nombre insuffisant d'observations (8 pour l'entraînement, 2 pour le test). Les modèles ne peuvent ni apprendre ni généraliser efficacement.
2.  **Multicolinéarité** : La relation intrinsèque entre `GDP_Billions`, `Population_Millions` et `GDP_per_Capita` crée une forte multicolinéarité, rendant l'interprétation des coefficients du modèle linéaire instable et parfois trompeuse.

### Pistes d'Amélioration
1.  **Acquisition de Données Massives** : Il est impératif d'obtenir un jeu de données significativement plus grand (plus d'années, plus de pays, plus d'indicateurs) pour construire des modèles prédictifs robustes et généralisables.
2.  **Analyse Descriptive Approfondie** : En l'absence de données suffisantes pour la prédiction, se concentrer sur une analyse descriptive détaillée des motifs existants serait plus judicieux.
3.  **Exploration de Modèles Avancés** : Avec un jeu de données plus conséquent, l'utilisation de modèles de séries chronologiques ou de méthodes d'ensemble plus sophistiquées avec des techniques de régularisation serait pertinente.

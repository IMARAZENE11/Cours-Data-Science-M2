from pathlib import Path
from IPython.display import display, Markdown

# Génération du plan Markdown du notebook structuré
markdown_sections = [
    "# 🏡 Airbnb Price Prediction Project",
    "## 🎯 Objectif",
    "L’objectif de ce projet est de prédire le **logarithme du prix** des logements Airbnb en utilisant des techniques de machine learning supervisé, sur la base de leurs caractéristiques (type, localisation, équipements, etc.).",
    
    "---",
    "## 1. 📥 Chargement des données",
    "```python\nimport pandas as pd\nimport numpy as np\ntrain = pd.read_csv(\"airbnb_train.csv\")\ntest = pd.read_csv(\"airbnb_test.csv\")\n```",

    "---",
    "## 2. 📊 Analyse exploratoire (EDA)",
    "- Aperçu des colonnes\n- Analyse des valeurs manquantes\n- Distribution de `log_price`\n- Analyse des variables catégorielles et numériques",

    "---",
    "## 3. 🧹 Nettoyage & Feature Engineering",
    "- Imputation des colonnes manquantes (`bathrooms`, `beds`, etc.)\n- Transformation de `cleaning_fee`, `description`, `amenities`, `host_since`\n- Création de nouvelles features utiles",

    "---",
    "## 4. 🔧 Préparation des données pour les modèles",
    "- Séparation colonnes numériques / catégorielles\n- Création d’un `ColumnTransformer` avec pipelines d’imputation, encodage et scaling",

    "---",
    "## 5. 🧠 Modélisation",
    "- Régression linéaire (baseline)\n- Random Forest\n- ✅ XGBoost (RMSE : 0.4089)",
    
    "---",
    "## 6. 🧪 Évaluation",
    "- Split train/test\n- Cross-validation\n- RMSE comme métrique principale",

    "---",
    "## 7. 📤 Génération des prédictions finales",
    "```python\nsubmission = pd.DataFrame({\n    'id': ids,\n    'prediction': final_preds\n})\nsubmission.to_csv(\"prediction.csv\", index=False)\n```",

    "---",
    "## ✅ Conclusion",
    "- XGBoost a fourni les meilleures performances avec une RMSE de 0.4089\n- Le pipeline peut être amélioré avec plus de feature engineering (géolocalisation, texte, etc.)\n- Le fichier `prediction.csv` est conforme et prêt à être soumis 🎯"
]

# Affichage du plan dans un format Notebook
display(Markdown("\n\n".join(markdown_sections)))


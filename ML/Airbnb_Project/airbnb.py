# 🏡 Airbnb Price Prediction Project

# 1. 📥 Chargement des données
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

train = pd.read_csv("airbnb_train.csv")
test = pd.read_csv("airbnb_test.csv")

# 2. 📊 Analyse exploratoire (EDA)
print(train.shape)
print(train.dtypes)
print(train.isnull().sum())
sns.histplot(train['log_price'], bins=40, kde=True)
plt.title("Distribution de log_price")
plt.show()

# 3. 🧹 Nettoyage & Feature Engineering
train.drop(columns=['host_response_rate', 'review_scores_rating', 'first_review', 'last_review'], inplace=True)
test.drop(columns=['host_response_rate', 'review_scores_rating', 'first_review', 'last_review'], inplace=True)

for col in ['bathrooms', 'beds', 'bedrooms']:
    train[col].fillna(train[col].median(), inplace=True)
    test[col].fillna(test[col].median(), inplace=True)

for col in ['host_has_profile_pic', 'host_identity_verified']:
    train[col].fillna('faux', inplace=True)
    test[col].fillna('faux', inplace=True)

def parse_cleaning_fee(x):
    if isinstance(x, str) and '$' in x:
        return float(x.replace('$','').replace(',',''))
    elif x is True:
        return 1.0
    elif x is False:
        return 0.0
    else:
        return np.nan

train['cleaning_fee'] = train['cleaning_fee'].apply(parse_cleaning_fee).fillna(0.0)
test['cleaning_fee'] = test['cleaning_fee'].apply(parse_cleaning_fee).fillna(0.0)

train['desc_length'] = train['description'].fillna('').apply(len)
test['desc_length'] = test['description'].fillna('').apply(len)

train['n_amenities'] = train['amenities'].fillna('').apply(lambda x: len(x.split(',')))
test['n_amenities'] = test['amenities'].fillna('').apply(lambda x: len(x.split(',')))

today = pd.to_datetime('2025-01-01')
train['host_since'] = pd.to_datetime(train['host_since'], errors='coerce')
test['host_since'] = pd.to_datetime(test['host_since'], errors='coerce')
train['host_age_days'] = (today - train['host_since']).dt.days
test['host_age_days'] = (today - test['host_since']).dt.days

# 4. 🔧 Préparation des données
X = train.drop(columns=['log_price', 'id', 'description', 'name', 'amenities'])
y = train['log_price']
X_test = test[X.columns]

num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include='object').columns.tolist()

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# 5. 🧠 Modélisation - XGBoost
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1))
])

xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_val)
rmse = mean_squared_error(y_val, y_pred, squared=False)
print("✅ RMSE sur validation (XGBoost) :", round(rmse, 4))

# 6. 📤 Génération des prédictions finales
xgb_model.fit(X, y)
final_preds = xgb_model.predict(X_test)

ids = pd.read_csv("airbnb_test.csv")['Unnamed: 0']
submission = pd.DataFrame({
    'id': ids,
    'prediction': final_preds
})
submission.to_csv("prediction.csv", index=False)
submission.head()

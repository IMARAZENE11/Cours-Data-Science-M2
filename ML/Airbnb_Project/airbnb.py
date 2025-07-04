# 🏡 Airbnb Price Prediction Project — version finale optimisée + Feature Selection + Target Encoding améliorée

# 1. 🕵️ Chargement des données
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
from math import radians, sin, cos, sqrt, atan2

train = pd.read_csv("airbnb_train.csv")
test = pd.read_csv("airbnb_test.csv")

# 2. 📊 Analyse exploratoire
sns.histplot(train['log_price'], bins=40, kde=True)
plt.title("Distribution de log_price")
plt.show()

# 3. 🩹 Nettoyage & Feature Engineering
cols_to_drop = ['host_response_rate', 'review_scores_rating', 'first_review', 'last_review']
train.drop(columns=cols_to_drop, inplace=True)
test.drop(columns=cols_to_drop, inplace=True)

for col in ['bathrooms', 'beds', 'bedrooms']:
    train[col].fillna(train[col].median(), inplace=True)
    test[col].fillna(train[col].median(), inplace=True)

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
    return 0.0

train['cleaning_fee'] = train['cleaning_fee'].apply(parse_cleaning_fee)
test['cleaning_fee'] = test['cleaning_fee'].apply(parse_cleaning_fee)

train['desc_length'] = train['description'].fillna('').apply(len)
test['desc_length'] = test['description'].fillna('').apply(len)

train['n_amenities'] = train['amenities'].fillna('').apply(lambda x: len(x.split(',')))
test['n_amenities'] = test['amenities'].fillna('').apply(lambda x: len(x.split(',')))

today = pd.to_datetime('2025-01-01')
train['host_since'] = pd.to_datetime(train['host_since'], errors='coerce')
test['host_since'] = pd.to_datetime(test['host_since'], errors='coerce')

train['host_age_days'] = (today - train['host_since']).dt.days
host_age_median = train['host_age_days'].median()
train['host_age_days'] = train['host_age_days'].fillna(host_age_median)

test['host_age_days'] = (today - test['host_since']).dt.days
test['host_age_days'] = test['host_age_days'].fillna(host_age_median)

train['price_per_person'] = train['cleaning_fee'] / (train['accommodates'] + 1)
test['price_per_person'] = test['cleaning_fee'] / (test['accommodates'] + 1)

train['luxury'] = ((train['beds'] >= 3) & (train['bathrooms'] >= 2)).astype(int)
test['luxury'] = ((test['beds'] >= 3) & (test['bathrooms'] >= 2)).astype(int)

train['accom_bedrooms'] = train['accommodates'] / (train['bedrooms'] + 1)
test['accom_bedrooms'] = test['accommodates'] / (test['bedrooms'] + 1)

train['log_reviews'] = np.log1p(train['number_of_reviews'])
test['log_reviews'] = np.log1p(test['number_of_reviews'])

train['bed_bath_ratio'] = train['beds'] / (train['bathrooms'] + 0.1)
test['bed_bath_ratio'] = test['beds'] / (test['bathrooms'] + 0.1)

train['bedroom_x_bathroom'] = train['bedrooms'] * train['bathrooms']
test['bedroom_x_bathroom'] = test['bedrooms'] * test['bathrooms']

train['accom_x_beds'] = train['accommodates'] * train['beds']
test['accom_x_beds'] = test['accommodates'] * test['beds']

# Réduction de cardinalité des variables catégorielles
def reduce_categories(df, col, min_freq=0.01):
    value_counts = df[col].value_counts(normalize=True)
    rare = value_counts[value_counts < min_freq].index
    df[col] = df[col].replace(rare, 'Other')

for col in ['property_type', 'bed_type', 'cancellation_policy', 'city']:
    reduce_categories(train, col)
    test[col] = test[col].where(test[col].isin(train[col].unique()), other='Other')

city_centers = {
    "Paris": (48.8566, 2.3522),
    "New York": (40.7128, -74.0060),
    "London": (51.5074, -0.1278),
    "San Francisco": (37.7749, -122.4194),
    "Los Angeles": (34.0522, -118.2437),
    "Berlin": (52.5200, 13.4050)
}

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

def compute_distance_to_center(row):
    if row['city'] in city_centers:
        center_lat, center_lon = city_centers[row['city']]
        return haversine_distance(row['latitude'], row['longitude'], center_lat, center_lon)
    return np.nan

train['distance_to_center'] = train.apply(compute_distance_to_center, axis=1)
test['distance_to_center'] = test.apply(compute_distance_to_center, axis=1)

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
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# 5. 🧠 Modélisation - XGBoost avec Feature Selection
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

xgb_selector = XGBRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.85,
    gamma=0.2,
    reg_alpha=0.4,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('selector', SelectFromModel(xgb_selector, threshold='mean')),
    ('regressor', XGBRegressor(
        n_estimators=2500,
        learning_rate=0.01,
        max_depth=7,
        subsample=0.9,
        colsample_bytree=0.85,
        gamma=0.2,
        reg_alpha=0.4,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
        verbosity=0
    ))
])

model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_val)
rmse = mean_squared_error(y_val, y_pred, squared=False)
print("✅ RMSE sur validation (XGBoost optimisé+++):", round(rmse, 4))

# 6. 📄 Prédiction finale
model_pipeline.fit(X, y)
final_preds = model_pipeline.predict(X_test)

ids = pd.read_csv("airbnb_test.csv")["Unnamed: 0"]
submission = pd.DataFrame({
    'id': ids,
    'prediction': final_preds
})
submission.to_csv("prediction.csv", index=False)
submission.head()
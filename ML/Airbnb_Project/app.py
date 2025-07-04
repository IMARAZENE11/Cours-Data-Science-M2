import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Fonction pour charger et préparer les données
def preprocess_and_predict(train_df, test_df):
    # Suppression des colonnes inutilisables
    for df in [train_df, test_df]:
        df.drop(columns=['host_response_rate', 'review_scores_rating', 'first_review', 'last_review'], inplace=True, errors='ignore')

    for col in ['bathrooms', 'beds', 'bedrooms']:
        train_df[col].fillna(train_df[col].median(), inplace=True)
        test_df[col].fillna(train_df[col].median(), inplace=True)

    for col in ['host_has_profile_pic', 'host_identity_verified']:
        train_df[col].fillna('faux', inplace=True)
        test_df[col].fillna('faux', inplace=True)

    def parse_cleaning_fee(x):
        if isinstance(x, str) and '$' in x:
            return float(x.replace('$','').replace(',',''))
        elif x is True:
            return 1.0
        elif x is False:
            return 0.0
        else:
            return np.nan

    train_df['cleaning_fee'] = train_df['cleaning_fee'].apply(parse_cleaning_fee).fillna(0.0)
    test_df['cleaning_fee'] = test_df['cleaning_fee'].apply(parse_cleaning_fee).fillna(0.0)

    train_df['desc_length'] = train_df['description'].fillna('').apply(len)
    test_df['desc_length'] = test_df['description'].fillna('').apply(len)

    train_df['n_amenities'] = train_df['amenities'].fillna('').apply(lambda x: len(x.split(',')))
    test_df['n_amenities'] = test_df['amenities'].fillna('').apply(lambda x: len(x.split(',')))

    today = pd.to_datetime('2025-01-01')
    train_df['host_since'] = pd.to_datetime(train_df['host_since'], errors='coerce')
    test_df['host_since'] = pd.to_datetime(test_df['host_since'], errors='coerce')
    train_df['host_age_days'] = (today - train_df['host_since']).dt.days
    test_df['host_age_days'] = (today - test_df['host_since']).dt.days

    # Préparation des colonnes
    X = train_df.drop(columns=['log_price', 'id', 'description', 'name', 'amenities'], errors='ignore')
    y = train_df['log_price']
    X_test = test_df[X.columns]

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include='object').columns.tolist()

    num_pipeline = Pipeline([

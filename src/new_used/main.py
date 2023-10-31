import argparse
import json
import os
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# Project root and data path.
project_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)
DATA_FILENAME = os.path.join(
    project_root, 'data', 'MLA_100k_checked_v3.jsonlines'
)

######################################################
# Build dataset from raw format
######################################################
def build_dataset(filename: str = DATA_FILENAME) -> pd.DataFrame:
    data = [json.loads(x) for x in open(filename)]
    target = lambda x: x.get("condition")
    N = -10000
    X_train = data[:N]
    X_test = data[N:]
    y_train = [target(x) for x in X_train]
    y_test = [target(x) for x in X_test]
    for x in X_test:
        del x["condition"]
    return X_train, y_train, X_test, y_test

######################################################
# Build cleaned dataset
######################################################
def build_dataset_df(dataset: List[dict]) -> pd.DataFrame:
    print("Start cleaning dataset...")
    dataset_df = pd.json_normalize(dataset)
    columns_to_discard = [
        'differential_pricing', 'subtitle', 'catalog_product_id',
        'shipping.dimensions', 'original_price', 'official_store_id',
        'video_id', 'shipping.free_methods', 'sub_status', 'deal_ids',
        'variations', 'attributes', 'coverage_areas', 'listing_source',
        'international_delivery_mode', 'shipping.methods', 'shipping.tags',
        'seller_address.country.id', 'seller_address.state.id', 'seller_address.city.id',
        'seller_id', 'site_id', 'parent_item_id', 'category_id', 'currency_id',
        'descriptions', 'seller_address.country.name', 'seller_address.state.name',
        'seller_address.city.name'
    ]
    dataset_df.drop(columns_to_discard, axis = 1, inplace = True)

    # Build count for list type fields
    counts_df = build_count_fields_from_list(dataset_df)
    dataset_df = pd.concat([dataset_df, counts_df], axis = 1)

    # Handle datetime fields
    datetime_df = transform_datetime_fields(dataset_df)
    dataset_df = pd.concat([dataset_df, datetime_df], axis = 1)

    # Subset columns for post processing
    columns_to_keep = [
        'price',
        'listing_type_id',
        'buying_mode',
        'tags',
        'accepts_mercadopago',
        'automatic_relist',
        'status',
        'initial_quantity',
        'sold_quantity',
        'available_quantity',
        'shipping.local_pick_up',
        'shipping.free_shipping',
        'shipping.mode',
        'non_mercado_pago_payment_methods',
        'non_mercado_pago_payment_methods_count',
        'pictures_count',
        'age_days',
        'duration_days',
    ]
    dataset_df = dataset_df[columns_to_keep]

    # Replace nan with zeros for count fields
    dataset_df['non_mercado_pago_payment_methods_count'].fillna(0, inplace = True)
    dataset_df['pictures_count'].fillna(0, inplace = True)

    print("Finished cleaning dataset...")
    print("Dataset shape: ", dataset_df.shape)

    return dataset_df


def build_count_fields_from_list(dataset_df: pd.DataFrame) -> pd.DataFrame:
    list_type_cols = [
        'non_mercado_pago_payment_methods', 'pictures'
    ]
    # Build the count fields
    list_type_cols_data = {}
    for col in list_type_cols:
        list_type_cols_data[col] = dataset_df[col].apply(
            lambda x: np.nan if len(x) == 0 else len(x)
        )

    df_list_cols = pd.DataFrame(
        data = list_type_cols_data
    )

    # Rename columns with count suffix
    df_list_cols.rename(
        columns = {
            'non_mercado_pago_payment_methods': 'non_mercado_pago_payment_methods_count',
            'pictures': 'pictures_count',
        },
        inplace = True
    )

    return df_list_cols

def transform_datetime_fields(dataset_df: pd.DataFrame) -> pd.DataFrame:
    df = dataset_df.copy()
    # Transform stop_time and start_time to datetime format
    df['start_time_dt'] = df['start_time'].apply(lambda x: datetime.fromtimestamp(x / 1000))
    df['stop_time_dt'] = df['stop_time'].apply(lambda x: datetime.fromtimestamp(x / 1000))

    # Transform date_created and last_updated to datetime format
    df['date_created_dt'] = pd.to_datetime(df['date_created'])
    df['last_updated_dt'] = pd.to_datetime(df['last_updated'])

    df['age_days'] = (df['last_updated_dt'] - df['date_created_dt']).apply(
        lambda x: x.total_seconds() / (60 * 60 * 24)
    )
    df['duration_days'] = (df['stop_time_dt'] - df['start_time_dt']).apply(
        lambda x: x.total_seconds() / (60 * 60 * 24)
    )

    return df[['age_days', 'duration_days']]

######################################################
# Build processed dataset
######################################################
def build_processed_dataset(dataset_df: pd.DataFrame) -> Dict[str, any]:
    """
    Takes a cleaned dataset in pandas dataframe format and returns a dictionary
    with processed data, sacalers and encoders.
    """
    print("Start processing dataset...")
    processed_dataset_df = dataset_df.copy()

    print("One hot encoding categorical variables...")
    # One hot encode categorical variables
    # Single value columns
    ohe_listing_type = OneHotEncoder()
    ohe_buying_mode = OneHotEncoder()
    ohe_status = OneHotEncoder()
    ohe_shipping_mode = OneHotEncoder()
    processed_dataset_df = one_hot_encode(processed_dataset_df, 'listing_type_id', ohe_listing_type, True)
    processed_dataset_df = one_hot_encode(processed_dataset_df, 'buying_mode', ohe_buying_mode, True)
    processed_dataset_df = one_hot_encode(processed_dataset_df, 'status', ohe_status, True)
    processed_dataset_df = one_hot_encode(processed_dataset_df, 'shipping.mode', ohe_shipping_mode, True)
    # Multi value columns
    # 1. Process tags
    processed_dataset_df = one_hot_encode_tags(processed_dataset_df)
    # 2. Process payment methods
    processed_dataset_df = one_hot_encode_payment_methods(processed_dataset_df)

    print("Finished one hot encoding categorical variables...")

    print("Handling boolean variables...")
    # Build boolean variables
    processed_dataset_df['accepts_mercadopago'] = processed_dataset_df['accepts_mercadopago'].astype(int)
    processed_dataset_df['automatic_relist'] = processed_dataset_df['automatic_relist'].astype(int)
    processed_dataset_df['shipping.local_pick_up'] = processed_dataset_df['shipping.local_pick_up'].astype(int)
    processed_dataset_df['shipping.free_shipping'] = processed_dataset_df['shipping.free_shipping'].astype(int)
    print("Finished handling boolean variables...")

    # Build response
    response = {
        'dataset': processed_dataset_df,
        'ohe_listing_type': ohe_listing_type,
        'ohe_buying_mode': ohe_buying_mode,
        'ohe_status': ohe_status,
        'ohe_shipping_mode': ohe_shipping_mode
    }

    print("Finished processing dataset...")
    print("Processed dataset shape: ", processed_dataset_df.shape)

    return response

def build_processed_test_dataset(
    dataset_df: pd.DataFrame,
    encoders: Dict[str, OneHotEncoder]
) -> pd.DataFrame:
    """
    Takes a cleaned test dataset in pandas dataframe format with a scaler and
    encoders and returns a dictionary with processed data.
    """
    print("Start processing dataset...")
    processed_dataset_df = dataset_df.copy()

    print("One hot encoding categorical variables...")
    # One hot encode categorical variables
    # Single value columns
    ohe_listing_type = encoders['ohe_listing_type']
    ohe_buying_mode = encoders['ohe_buying_mode']
    ohe_status = encoders['ohe_status']
    ohe_shipping_mode = encoders['ohe_shipping_mode']
    processed_dataset_df = one_hot_encode(processed_dataset_df, 'listing_type_id', ohe_listing_type, True, False)
    processed_dataset_df = one_hot_encode(processed_dataset_df, 'buying_mode', ohe_buying_mode, True, False)
    processed_dataset_df = one_hot_encode(processed_dataset_df, 'status', ohe_status, True, False)
    processed_dataset_df = one_hot_encode(processed_dataset_df, 'shipping.mode', ohe_shipping_mode, True, False)
    # Multi value columns
    # 1. Process tags
    processed_dataset_df = one_hot_encode_tags(processed_dataset_df)
    # 2. Process payment methods
    processed_dataset_df = one_hot_encode_payment_methods(processed_dataset_df)

    print("Finished one hot encoding categorical variables...")

    print("Handling boolean variables...")
    # Build boolean variables
    processed_dataset_df['accepts_mercadopago'] = processed_dataset_df['accepts_mercadopago'].astype(int)
    processed_dataset_df['automatic_relist'] = processed_dataset_df['automatic_relist'].astype(int)
    processed_dataset_df['shipping.local_pick_up'] = processed_dataset_df['shipping.local_pick_up'].astype(int)
    processed_dataset_df['shipping.free_shipping'] = processed_dataset_df['shipping.free_shipping'].astype(int)
    print("Finished handling boolean variables...")

    print("Finished processing dataset...")
    print("Processed dataset shape: ", processed_dataset_df.shape)

    return processed_dataset_df

######################################################
# Helper functions
######################################################
def one_hot_encode(
    df: pd.DataFrame, column: str, ohe: OneHotEncoder,
    drop: bool = False, is_fit: bool = True
) -> pd.DataFrame:
    """
    This function takes a dataframe and a column name as input and returns a
    new dataframe with the one hot encoded values of the column.
    """
    # Create a copy of the dataframe to avoid modifying the original dataframe
    df_copy = df.copy()

    # One hot encode the column
    if is_fit:
        encoded_data = ohe.fit_transform(df_copy[[column]]).toarray()
    else:
        encoded_data = ohe.transform(df_copy[[column]]).toarray()

    # Get the names of the new columns
    column_names = ohe.get_feature_names_out([column])

    # Create a new dataframe with the one hot encoded values
    encoded_df = pd.DataFrame(
        encoded_data,
        columns = column_names,
        index = df_copy.index
    )

    # Concatenate the new dataframe with the original one
    df_copy = pd.concat([df_copy, encoded_df], axis = 1)

    # Drop the original column
    if drop: df_copy.drop(column, axis = 1, inplace = True)

    return df_copy

def get_tags_dict(row):
    res = {
        'dragged_bids_and_visits': 0,
        'good_quality_thumbnail': 0,
        'dragged_visits': 0,
        'free_relist': 0,
        'poor_quality_thumbnail': 0
    }

    for val in row:
        res[val] = 1

    return res

def one_hot_encode_tags(df: pd.DataFrame, drop: bool = True):
    """
    Takes a tag column and returns a dataframe with the one hot encoded values of the column.
    """
    # Create a copy of the dataframe to avoid modifying the original dataframe
    df_copy = df.copy()

    # One hot encode the column
    encoded_data = pd.json_normalize(df_copy['tags'].apply(get_tags_dict))

    # Concatenate the new dataframe with the original one
    df_copy = pd.concat([df_copy, encoded_data], axis = 1)

    # Drop the original column
    if drop: df_copy.drop('tags', axis = 1, inplace = True)

    return df_copy

def get_payment_methods(x):
    if len(x) == 0:
        return []
    else:
        return [d['id'] for d in x]

def get_payment_methods_dict(row):
    res = {
        'MLATB': 0, 'MLAWC': 0, 'MLAMO': 0, 'MLAOT': 0, 'MLAMC': 0, 'MLAMS': 0, 'MLAVE': 0,
        'MLACD': 0, 'MLAVS': 0, 'MLADC': 0, 'MLAAM': 0, 'MLAWT': 0, 'MLAMP': 0, 'MLABC': 0
    }

    for val in row:
        res[val] = 1

    return res

def one_hot_encode_payment_methods(df: pd.DataFrame, drop: bool = True):
    """
    Takes a payment methods column and returns a dataframe with the one hot encoded values of the column.
    """
    # Create a copy of the dataframe to avoid modifying the original dataframe
    df_copy = df.copy()

    # transform column to list of ids
    df_copy['non_mercado_pago_payment_methods'] = df_copy['non_mercado_pago_payment_methods'].apply(get_payment_methods)

    # One hot encode the column
    encoded_data = pd.json_normalize(df_copy['non_mercado_pago_payment_methods'].apply(get_payment_methods_dict))

    # Concatenate the new dataframe with the original one
    df_copy = pd.concat([df_copy, encoded_data], axis = 1)

    # Drop the original column
    if drop: df_copy.drop('non_mercado_pago_payment_methods', axis = 1, inplace = True)

    return df_copy

# Encode the condition column
class CustomLabelEncoder(LabelEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Define the desired mapping
        self.mapping = {'new': 1, 'used': 0}

    def fit(self, y):
        super().fit(list(self.mapping.keys()))

    def transform(self, y):
        return [self.mapping[item] for item in y]

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--filename', type = str,
        default = None,
        required = False,
        help = 'Name of the file to process'
    )
    args = parser.parse_args()

    print("----" * 5, "Loading dataset", "----" * 5)
    if args.filename:
        X_train, y_train, X_test, y_test = build_dataset(args.filename)
    else:
        X_train, y_train, X_test, y_test = build_dataset()

    print("----" * 5, "Cleaning dataset", "----" * 5)
    X_train_clean_df = build_dataset_df(X_train)
    X_test_clean_df = build_dataset_df(X_test)

    print("----" * 5, "Processing train dataset", "----" * 5)
    X_train_proc_dict = build_processed_dataset(X_train_clean_df)
    X_train_proc_df = X_train_proc_dict['dataset']

    print("----" * 5, "Processing test dataset", "----" * 5)
    X_test_proc_df = build_processed_test_dataset(
        dataset_df = X_test_clean_df,
        encoders = X_train_proc_dict
    )

    print("----" * 5, "Processing labels", "----" * 5)
    le = CustomLabelEncoder()
    y_train_proc = le.fit_transform(y_train)
    y_test_proc = le.transform(y_test)

    print("----" * 5, "Starting XGBoost Training", "----" * 5)

    selected_features = [
        'price', 'accepts_mercadopago', 'automatic_relist', 'initial_quantity',
        'sold_quantity', 'available_quantity', 'shipping.local_pick_up',
        'shipping.free_shipping', 'non_mercado_pago_payment_methods_count',
        'pictures_count', 'age_days', 'duration_days', 'listing_type_id_bronze',
        'listing_type_id_free', 'listing_type_id_gold', 'listing_type_id_gold_special',
        'listing_type_id_silver', 'buying_mode_buy_it_now', 'buying_mode_classified',
        'status_active', 'status_paused', 'shipping.mode_custom', 'shipping.mode_me2',
        'shipping.mode_not_specified', 'dragged_bids_and_visits', 'MLATB', 'MLAWC',
        'MLAMO', 'MLAOT', 'MLAWT'
    ]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(
        X_train_proc_df[selected_features]
    )
    # Train the model
    xgb = XGBClassifier(
        learning_rate = 0.1,
        max_depth = 7,
        n_estimators = 500,
        reg_alpha = 0,
        reg_lambda = 0.1,
        subsample = 0.8,
        colsample_bytree = 0.8,
        gamma = 0,
    )
    xgb.fit(X_train_scaled, y_train_proc)
    print("----" * 5, "Finished XGBoost Training", "----" * 5)

    print("----" * 5, "Starting XGBoost Testing", "----" * 5)
    X_test_scaled = scaler.transform(X_test_proc_df[selected_features])
    y_pred = xgb.predict(X_test_scaled)
    print("** Accuracy score of classifier: ", accuracy_score(y_test_proc, y_pred))
    print("** Precision score of classifier: ", precision_score(y_test_proc, y_pred))
    print("** Recall score of classifier: ", recall_score(y_test_proc, y_pred))
    print("** F1 score of classifier: ", f1_score(y_test_proc, y_pred))

    print("----" * 5, "Done!", "----" * 5)


if __name__ == "__main__":
    main()

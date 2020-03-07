import pickle
import numpy as np
import pandas as pd
import re
from sklearn.impute import SimpleImputer

def pickle_read(path):
    with open(path, "rb") as f:
        pickle_file = pickle.load(f)
    return pickle_file

def pickle_write(item, path):
    with open(path, "wb") as f:
        pickle.dump(item, f)

def extract_column_names(df, term):
    matches = [column for column in df.columns if re.search(term, column)]
    return pd.Index(matches)

def extract_impute_values(df, column, bad_data):
    isolate_good = df[df[column] != bad_data]
    return isolate_good[column].median()

def missing_val_dummies(df, column, bad_data):
     return np.where(df[column].values == bad_data, 1, 0)

def get_imputer_objs(df, impute_dict):
    return [SimpleImputer(val[0], val[1], val[2]).fit(df[[key]]) for key, val in impute_dict.items()]

def impute_vals(df, impute_dict, imputers):
    for i, column in enumerate(impute_dict.keys()):
        df["missing_"+ column] = missing_val_dummies(df, column, impute_dict[column][0])
        index = df[column].index
        df[column] = pd.Series(imputers[i].transform(df[[column]]).ravel(), index=index)
    return df


def simp_datetime_map(df, col, format=None):
    df[col] = df[col].map(lambda x: pd.to_datetime(x, format=format))
    return df
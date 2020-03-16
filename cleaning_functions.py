import pickle
import numpy as np
import pandas as pd
import re
from sklearn.impute import SimpleImputer

#Handles the file opening to expedite the pickle reading process.
def pickle_read(path):
    with open(path, "rb") as f:
        pickle_file = pickle.load(f)
    return pickle_file

#Handles the file opening to expedite the pickle writing process.
def pickle_write(item, path):
    with open(path, "wb") as f:
        pickle.dump(item, f)

#Chains the non imputing data cleaning commands.
def main_data_cleaning(df):
    df = df.drop(columns=["recorded_by"])
    df = simp_datetime_map(df, "date_recorded")
    return df

#Creates a Pandas Index object for column names found via a regex search.
def extract_column_names(df, term):
    matches = [column for column in df.columns if re.search(term, column)]
    return pd.Index(matches)

#Was used to impute values before the sklearn libraries were imported.
def extract_impute_values(df, column, bad_data):
    isolate_good = df[df[column] != bad_data]
    return isolate_good[column].median()

#Creates a new column containing dummy variables indicating what was missing when imputing data.
def missing_val_dummies(df, column, bad_data):
    if np.isnan(bad_data):
        return np.where(df[column].isna().values == True, 1, 0)
    else:
        return np.where(df[column].values == bad_data, 1, 0)

#Loops through a dictionary containing parameters to construct imputer objects. Arguments required by the dictionary
#for the function to work are a key containing the column name followed by a list of values in this order:
#name of missing value, imputation strategy, and value if imputation is a predetermined constant.
def get_imputer_objs(df, impute_dict):
    return [SimpleImputer(val[0], val[1], val[2]).fit(df[[key]]) for key, val in impute_dict.items()]

#Loops through imputer object list and creates dummies for missing values. If the dictionary indicates that
# a new categorical value should be added, dummy creation is skipped.
def impute_vals(df, impute_dict, imputers):
    for i, column in enumerate(impute_dict.keys()):
        if type(impute_dict[column][2]) != str and impute_dict[column][1] != "constant":
            df["missing_"+ column] = missing_val_dummies(df, column, impute_dict[column][0])
        index = df[column].index
        df[column] = pd.Series(imputers[i].transform(df[[column]]).ravel(), index=index)
    return df

#Used in situations when imputing data to replace a specific value and a nan as well. Requires a dictionary that
# contains the column names as keys and the values of the term and it's replacement.
def impute_mult_categorical(df, impute_dict):
    for column, vals in impute_dict.items():
        index = df[column].index
        df[column].fillna(vals[1], inplace=True)
        imputed = np.where(df[column].values == vals[0], vals[1], df[column].values)
        df[column] = pd.Series(imputed.ravel(), index=index)
    return df

#Maps a dataframe's column to datetime values.
def simp_datetime_map(df, col, format=None):
    df[col] = df[col].map(lambda x: pd.to_datetime(x, format=format))
    return df
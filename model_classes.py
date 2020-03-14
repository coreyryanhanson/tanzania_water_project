import pandas as pd
import numpy as np
import math
from copy import copy, deepcopy
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.under_sampling import TomekLinks
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures


class ModelSwitcher(object):
    def __init__(self, data, duplicate=False):
        self.duplicate = duplicate
        self._instantiate_data(data)

    def _instantiate_data(self, data):
        if not self.duplicate:
            self.data = data
        else:
            self.data = deepcopy(data)

class DataPreprocessor(object):
    def __init__(self, df, target, cat_features={}, cont_features={}, transformed_interactions=False, dummy_interactions=False):
        self.df = df
        self.transformed_interactions = transformed_interactions
        self.dummy_interactions = dummy_interactions
        self._set_features(target, cat_features, cont_features)
        self.X = df[self.cols]
        self.y = df[target]

    #Creates various attributes storing column names from specifically structured dictionaries for the
    # categorical and continuous variables.
    def _set_features(self, target, cat_features, cont_features):
        self._get_cat_features(cat_features)
        self._get_cont_features(cont_features)
        self.cols_initial = self.cols_continuous.union(self.cols_categorical, sort=False)
        self.cols = self.cols_initial
        self.target = target

    #Gathers categorical column name information and creates corresponding attributes.
    def _get_cat_features(self, feature_dict):
        self.cols_nominal = self._get_indiv_feature(feature_dict, "nominal_features")
        self.cols_standard_dummies = self._get_indiv_feature(feature_dict, "standard_dummies")
        self.cols_impute_dummies = self._get_indiv_feature(feature_dict, "impute_dummies")
        self.cols_dummies = self.cols_standard_dummies.union(self.cols_impute_dummies, sort=False)
        self.cols_categorical = self.cols_nominal.union(self.cols_dummies, sort=False)

    # Gathers continuous column name information and creates corresponding attributes, calling transformation functions if specified.
    def _get_cont_features(self, feature_dict):
        transformed_dict = self._get_feature_group(feature_dict, "transformed")
        self.cols_linear = self._get_indiv_feature(feature_dict, "untransformed")
        if transformed_dict:
            self._get_trans_features(transformed_dict)
            self.cols_continuous = self.cols_linear.union(self.cols_transformed, sort=False)
        else:
            self.cols_transformed = pd.Index([])
            self.cols_continuous = self.cols_linear

    #Gathers transformed features in the dictionary for continous features. New transformed columns are performed for whatever
    # transformations are specified.
    def _get_trans_features(self, transformed_dict):
        logged = self._get_feature_group(transformed_dict, "logged")
        pow = self._get_feature_group(transformed_dict, "exp")
        self.cols_logged = pd.Index([])
        if logged:
            self._log_features(logged)
        else:
            pass
        self.cols_transformed = self.cols_logged

    #Checks the for the existence of a key in a nested dictionary.
    def _get_feature_group(self, feature_dict, key):
        return feature_dict.get(key)

    #Checks a specific category of features being present in the passed dictionary returning an empty list if no results.
    def _get_indiv_feature(self, feature_dict, key, default=[]):
        return pd.Index(feature_dict.get(key, default))

    #Performs log transformations and gathers the new column names.
    def _log_features(self, logged_dict):
        for column, base in logged_dict.items():
            if base:
                new_col_name = f"{column}_log_b{base}"
            else:
                new_col_name = f"{column}_ln"
            self.df[new_col_name] = self.df[column].map(lambda x: math.log(x, base) if base else math.log(x))
            self.cols_logged = self.cols_logged.append(pd.Index([new_col_name]))

    def _train_test_split(self):
        X, y = self.X, self.y
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=self.test_size, random_state=self.random_state)
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

    def _fit_scale(self):
        if self.scale_type == "standard":
            print("Using standard scaler")
            self.scaler = StandardScaler()
        elif self.scale_type == "minmax":
            print("Using Min/Max scaler")
            self.scaler = MinMaxScaler()
        else:
            print("No scaling specified")
            self.scale_type = False
            return
        self.scaler.fit(self.X_train[self.scaled_columns])

    def _rescale(self):
        if self.scale_type == False:
            print("Skipping scaling")
            return
        else:
            columns = self.scaled_columns

        #Overwrites original train/test dataframes to prevent linkage errors.
        self.X_test, self.X_train = self.X_test.copy(), self.X_train.copy()

        #Calls the transform function on both the train and test data targeting only the relevant columns.
        X_train = self._transform_scale(self.X_train)
        X_test = self._transform_scale(self.X_test)
        self.X_train[columns] = X_train[columns]
        self.X_test[columns] = X_test[columns]

    def _transform_scale(self, data):
        to_transform = data[self.scaled_columns]
        indices = to_transform.index
        scaled = self.scaler.transform(to_transform)
        return pd.DataFrame(scaled, columns=self.scaled_columns, index=indices)

    def _class_imbalance(self):
        df = pd.concat([self.X_train, self.y_train], axis=1)
        if self.balance_class == "upsample":
            print("Performing upsample")
            self._simple_resample(df)
        elif self.balance_class == "downsample":
            print("Performing downsample")
            self._simple_resample(df, down=True)
        elif self.balance_class == "smote":
            print("Performing SMOTE")
            self._smote_data()
        elif self.balance_class == "tomek":
            print("Performing Tomek Links")
            self._tomek_data()
        else:
            print("Skipping class imbalance functions")

    def _simple_resample(self, df, down=False):
        target = self.target
        groups = [item for item in df[target].unique()]
        counts = {group: df[df[target] == group][target].count() for group in groups}
        most, least = max(counts, key=counts.get), min(counts, key=counts.get)
        if down == False:
            goal, samples = most, counts[most]
        else:
            goal, samples = least, counts[least]
        sample_queue = [remaining for remaining in groups if remaining != goal]
        new_df = df[df[target]==goal]
        for sample in sample_queue:
            current = df[df[target]==sample]
            resampled = resample(current, replace=True, n_samples=samples, random_state=self.random_state)
            new_df = pd.concat([new_df, resampled])
        self.X_train, self.y_train = new_df.drop(self.target, axis=1), new_df[self.target]

    def _smote_data(self):
        if self.cols_nominal.size > 0:
            cats = self.X_train.columns.isin(self.cols_nominal)
            sm = SMOTENC(categorical_features=cats, sampling_strategy='not majority', random_state=self.random_state)
        else:
            sm = SMOTE(sampling_strategy='not majority', random_state=self.random_state)
        self.X_train, self.y_train = sm.fit_sample(self.X_train, self.y_train)

    def _tomek_data(self):
        if self.cols_nominal.size > 0:
            print("Skipping Tomek Links. Cannot perform with raw categorical data. Create dummies to use.")
            return
        tl = TomekLinks()
        self.X_train, self.y_train = tl.fit_sample(self.X_train, self.y_train)

    def _poly_features(self):
        if type(self.poly_degree) == int:
            print(f"Getting polynomial features of degree {self.poly_degree}")
            orig_columns = self._choose_poly_columns()
            X_cont = self.X[orig_columns]
            X_cont_index = X_cont.index
            poly = PolynomialFeatures(degree=self.poly_degree, include_bias=False)
            X_poly = poly.fit_transform(X_cont)
            columns = pd.Index(poly.get_feature_names(X_cont.columns))
            poly_df = pd.DataFrame(X_poly, index=X_cont_index, columns=columns)
            self.cols_polynomial = columns.drop(labels=orig_columns)
            self.X = pd.concat([self.X[self.cols_initial], poly_df[self.cols_polynomial]], axis=1)
            self.cols = self.cols_initial.union(self.cols_polynomial, sort=False)
        else:
            print("Skipping polynomial features")
            self.poly_degree = False
            self.cols_polynomial = pd.Index([])
            self.X = self.X[self.cols_initial]

    #Creates a column list for polynomial features including or excluding dummy variables and transformed features depending
    # on arguments.
    def _choose_poly_columns(self):
        if self.transformed_interactions and self.cols_transformed.size > 0:
            columns = self.cols_continuous
        else:
            columns = self.cols_linear
        if self.dummy_interactions:
            return columns.union(self.cols_dummies, sort=False)
        else:
            return columns

    def data_preprocessing(self, balance_class=False, scale_type=False, poly_degree=False, transform_dummies=False):
        self.scaled_columns = self.cols.drop(labels=self.cols_nominal)
        self.random_state = 1
        self.test_size = .2
        self.poly_degree = poly_degree
        self.balance_class = balance_class
        self.scale_type = scale_type
        self._poly_features()
        self._train_test_split()
        self._class_imbalance()
        self._fit_scale()
        self._rescale()

    #Drops unwanted features from the column selection.
    def column_drop(self, columns):
        self.cols = self.cols.drop(labels=columns)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print("F1 Score:", f1)
    print("Accuracy:", accuracy)

'''
Author: Reza Marzban
Date: 6/13/2020
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


preprocess_file_address = "../Data/preprocessed_data.csv"


def preliminary_visualization(df, cat_cols):
    for col in cat_cols:
        df[col].value_counts().plot(kind='bar')
        plt.ylabel('Count')
        plt.title(col + " distribution")
        plt.xticks(rotation=45, ha='right')
        plt.show()


def create_dummy_variables(df, cat_cols):
    for col in cat_cols:
        one_hot = pd.get_dummies(df[col], prefix=col)
        df = df.join(one_hot)
        df = df.drop(col, axis=1)
    return df


def normalize(df, numerical_columns):
    for col in numerical_columns:
        df[col] = (df[col] - df[col].min())/(df[col].max() - df[col].min())
    return df


def categorical_to_binary(df, binary_cols):
    for col in binary_cols:
        df[col] = df[col].astype('category').cat.codes


def pre_process(data_filename):
    df = pd.read_csv(data_filename, sep=';')
    df = df.drop("default", axis=1)
    categorical_columns = ['job', 'marital', 'education', 'housing', 'loan',
                           'contact', 'month', 'day_of_week', 'poutcome', 'y']
    numerical_columns = list(set(df.columns)-set(categorical_columns))

    # preliminary_visualization(df, categorical_columns)
    df.replace('unknown', np.nan, inplace=True)
    df.dropna(inplace=True)

    binary_cols = ['housing', 'loan', 'contact', 'y']
    categorical_columns = list(set(categorical_columns)-set(binary_cols))
    categorical_to_binary(df, binary_cols)
    df = create_dummy_variables(df, categorical_columns)
    df = normalize(df, numerical_columns)
    df.to_csv(preprocess_file_address, index=False)
    print("\nPreprocess phase is complete, the cleaned data is saved in:" + preprocess_file_address + "\n")
    print()


def load_and_split():
    df = pd.read_csv(preprocess_file_address)
    negetive_df = df[df["y"] == 0]
    positive_df = df[df["y"] == 1]
    # balance data
    if len(positive_df) < len(negetive_df):
        negetive_df = negetive_df.sample(n=len(positive_df))
    elif len(negetive_df) < len(positive_df):
        positive_df = positive_df.sample(n=len(negetive_df))
    train_set = pd.concat([negetive_df, positive_df])
    # shuffle
    train_set = train_set.sample(frac=1)
    # split to train and test sets
    train, validation = train_test_split(train_set, test_size=0.2)
    # separate the target variable from features
    validation_y = validation["y"].values
    validation_x = validation.drop("y", axis=1).values
    train_y = train["y"].values
    train_x = train.drop("y", axis=1).values
    print("\nTraining set size: " + str(len(train_y)))
    print("Validation set size: " + str(len(validation_y)))
    return train_x, train_y, validation_x, validation_y


if __name__ == "__main__":
    print("\nPreprocess.py is a supplementary file, please Run main.py\n")

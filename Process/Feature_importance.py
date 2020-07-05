"""
Author: Reza Marzban
Date: 6/21/2020
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle


def important_features_random_forest():
    model_name = "Random Forest"
    model_file_address = "../temp/" + model_name + ".model"
    random_forest_model = pickle.load(open(model_file_address, 'rb'))
    feature_importance = random_forest_model.feature_importances_
    return feature_importance


def important_features_logistic_regression():
    model_name = "Logistic Regression"
    model_file_address = "../temp/" + model_name + ".model"
    logreg_model = pickle.load(open(model_file_address, 'rb'))
    feature_importance = np.absolute(logreg_model.coef_[0, :])
    feature_importance = feature_importance/feature_importance.sum()
    return feature_importance


def visualize_most_important_features(n=20):
    rf_feature_importance = important_features_random_forest()
    lr_feature_importance = important_features_logistic_regression()
    feature_importance = (rf_feature_importance+lr_feature_importance)/2
    df = pd.read_csv("../Data/preprocessed_data.csv")
    df = df.drop(["y"], axis=1)
    variable_names = df.columns.values

    indices = np.argsort(lr_feature_importance)[::-1]
    plt.figure()
    plt.title("Feature importances (Logistic Regression)")
    plt.bar(range(n), lr_feature_importance[indices[:n]], color="m", align="center")
    plt.xticks(range(n), variable_names[indices[:n]], rotation=45, ha='right')
    plt.ylabel("Importance rate")
    plt.xlabel("Features")
    plt.subplots_adjust(bottom=0.3)
    plt.show()
    print()

    indices = np.argsort(rf_feature_importance)[::-1]
    plt.figure()
    plt.title("Feature importances (Random Forest)")
    plt.bar(range(n), rf_feature_importance[indices[:n]], color="m", align="center")
    plt.xticks(range(n), variable_names[indices[:n]], rotation=45, ha='right')
    plt.ylabel("Importance rate")
    plt.xlabel("Features")
    plt.subplots_adjust(bottom=0.3)
    plt.show()
    print()

    indices = np.argsort(feature_importance)[::-1]
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(n), feature_importance[indices[:n]], color="m", align="center")
    plt.xticks(range(n), variable_names[indices[:n]], rotation=45, ha='right')
    plt.ylabel("Importance rate")
    plt.xlabel("Features")
    plt.subplots_adjust(bottom=0.3)
    plt.show()
    print()


if __name__ == "__main__":
    print("\nFeature_importance.py is a supplementary file, please Run main.py\n")


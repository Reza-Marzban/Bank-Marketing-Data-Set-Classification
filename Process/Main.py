"""
Author: Reza Marzban
Date: 6/13/2020
"""

from Process.Preprocess import pre_process, load_and_split
from Process.Models import run_and_compare_models
from Process.Feature_importance import visualize_most_important_features

if __name__ == "__main__":
    # read the raw data, preprocess it and save it.
    data_filename = "../Data/bank-additional-full.csv"
    pre_process(data_filename)

    # load preprocessed data
    train_x, train_y, validation_x, validation_y = load_and_split()

    # run and compare 4 models and visualize results
    run_and_compare_models(train_x, train_y, validation_x, validation_y)

    # calculate the importance rate of each feature
    visualize_most_important_features(n=52)

    print()


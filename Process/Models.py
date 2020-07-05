"""
Author: Reza Marzban
Date: 6/13/2020
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import keras
from keras import layers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_recall_fscore_support


def score_calculator(model_name, label, prediction, prediction_probability):
    accuracy = round(accuracy_score(label, prediction)*100, 2)
    fpr, tpr, _ = roc_curve(label, prediction_probability)
    roc_auc = round(auc(fpr, tpr), 4)
    precision, recall, f_score, _ = precision_recall_fscore_support(label, prediction, average=None)
    precision = round(precision[1]*100, 2)
    recall = round(recall[1]*100, 2)
    f_score = round(f_score[1]*100, 2)
    print(model_name + " - Performance on validation data: ")
    print("\tAccuracy:\t" + str(accuracy))
    print("\tPrecision:\t" + str(precision))
    print("\tRecall:\t\t" + str(recall))
    print("\tF_score:\t" + str(f_score))
    print("\tArea Under Curve in ROC:\t" + str(roc_auc))
    return accuracy, roc_auc, fpr, tpr


def logistic_regression(train_x, train_y, validation_x, validation_y):
    model_name = "Logistic Regression"
    logreg_model = LogisticRegression(solver='lbfgs')
    logreg_model.fit(train_x, train_y)
    validation_pred = logreg_model.predict(validation_x)
    validation_prob = logreg_model.predict_proba(validation_x)[:, 1]
    accuracy, roc_auc, fpr, tpr = score_calculator(model_name, validation_y, validation_pred, validation_prob)
    model_file_address = "../temp/" + model_name+".model"
    pickle.dump(logreg_model, open(model_file_address, 'wb'))
    print("model saved successfully in: " + model_file_address)
    print("_____________________________________________________________")
    return accuracy, roc_auc, fpr, tpr, model_name


def random_forest(train_x, train_y, validation_x, validation_y):
    model_name = "Random Forest"
    random_forest_model = RandomForestClassifier(n_estimators=100, max_depth=None)
    random_forest_model.fit(train_x, train_y)
    validation_pred = random_forest_model.predict(validation_x)
    validation_prob = random_forest_model.predict_proba(validation_x)[:, 1]
    accuracy, roc_auc, fpr, tpr = score_calculator(model_name, validation_y, validation_pred, validation_prob)
    model_file_address = "../temp/" + model_name + ".model"
    pickle.dump(random_forest_model, open(model_file_address, 'wb'))
    print("model saved successfully in: " + model_file_address)
    print("_____________________________________________________________")
    return accuracy, roc_auc, fpr, tpr, model_name


def naive_bayes(train_x, train_y, validation_x, validation_y):
    model_name = "Naive Bayes"
    naive_bayes_model = GaussianNB(var_smoothing=1e-01)
    naive_bayes_model.fit(train_x, train_y)
    validation_pred = naive_bayes_model.predict(validation_x)
    validation_prob = naive_bayes_model.predict_proba(validation_x)[:, 1]
    accuracy, roc_auc, fpr, tpr = score_calculator(model_name, validation_y, validation_pred, validation_prob)
    model_file_address = "../temp/" + model_name + ".model"
    pickle.dump(naive_bayes_model, open(model_file_address, 'wb'))
    print("model saved successfully in: " + model_file_address)
    print("_____________________________________________________________")
    return accuracy, roc_auc, fpr, tpr, model_name


def artificial_neural_network(train_x, train_y, validation_x, validation_y):
    threshold = 0.5
    model_name = "Artificial Neural Network"
    model = keras.Sequential()
    model.add(layers.Dense(32, activation="relu", input_dim=52))
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dense(8, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(train_x, train_y, epochs=20, verbose=0, batch_size=64, validation_data=(validation_x, validation_y))
    validation_prob = model.predict(validation_x)
    validation_pred = np.where(validation_prob >= threshold, 1, 0)
    accuracy, roc_auc, fpr, tpr = score_calculator(model_name, validation_y, validation_pred, validation_prob)
    model_file_address = "../temp/" + model_name + ".model"
    model.save(model_file_address)
    print("model saved successfully in: " + model_file_address)
    print("_____________________________________________________________")
    return accuracy, roc_auc, fpr, tpr, model_name


def create_roc_curve(results):
    plt.title('Receiver Operating Characteristic Curve')
    for r in results:
        accuracy, roc_auc, fpr, tpr, model_name = r
        plt.plot(fpr, tpr, linewidth=2, alpha=0.8, label=model_name + ' (AUC = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Baseline (AUC = 0.50)')
    plt.legend(loc='lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def run_and_compare_models(train_x, train_y, validation_x, validation_y):
    ann_results = artificial_neural_network(train_x, train_y, validation_x, validation_y)
    naive_bayes_results = naive_bayes(train_x, train_y, validation_x, validation_y)
    log_reg_results = logistic_regression(train_x, train_y, validation_x, validation_y)
    random_forest_results = random_forest(train_x, train_y, validation_x, validation_y)

    create_roc_curve([log_reg_results, random_forest_results, ann_results, naive_bayes_results])
    print()


if __name__ == "__main__":
    print("\nModels.py is a supplementary file, please Run main.py\n")

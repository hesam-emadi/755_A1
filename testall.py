import pandas as pd
import numpy as np
import seaborn as sn
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from gridSearch import graph_grid_search
from matplotlib import pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import BernoulliNB


def classify_all(data, data_name):

    X, y = data.drop(data.columns[-1], axis=1), data[data.columns[-1]]
    classify(X, y, data_name)


def classify(X, y, data_name):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    # train_ind, test_ind = None, None
    for train_index, test_index in sss.split(X, y):
        print(max(train_index))
        print(max(test_index))
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    labels = y.unique()
    labels.sort()

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    models = [
                {'model': SVC(), 'title': data_name + ' Linear kernel SVM',
                 'params': {'kernel': ['linear'], 'C': [2 ** x for x in range(0, 6)]}, 'x_param': 'C'},
                {'model': SVC(), 'title': data_name + ' RBF kernel SVM',
                 'params': {'kernel': ['rbf'], 'C': [2 ** x for x in range(0, 6)]}, 'x_param': 'C'},
                {'model': SVC(), 'title': data_name + ' Polynomial kernel SVM',
                 'params': {'kernel': ['poly'], 'C': [2 ** x for x in range(0, 6)]}, 'x_param': 'C'},
                {'model': DecisionTreeClassifier(), 'title': data_name + ' DecisionTree',
                 'params': {'max_leaf_nodes': [2 ** x for x in range(1, 10)]}, 'x_param': 'max_leaf_nodes'},
                {'model': KNeighborsClassifier(), 'title': data_name + ' KNN',
                 'params': {'n_neighbors': [2 ** x for x in range(1, 5)]}, 'x_param': 'n_neighbors'},
                {'model': BernoulliNB(), 'title': data_name + ' Bayes',
                 'params': {'alpha': [2 ** i / 100 for i in range(0, 13)]}, 'x_param': 'alpha'},
                {'model': Perceptron(), 'title': data_name + ' Perceptron',
                 'params': {'max_iter': [2 ** i for i in range(1, 13)]}, 'x_param': 'max_iter'}
        ]

    scoring = {'accuracy': 'accuracy', 'r2': 'r2'}
    for model_type in models:
        grid_search = graph_grid_search(model_type['title'], [model_type['x_param'], 'Score'], model_type['model'], model_type['params'],
                                        scoring, X_train, y_train, model_type['x_param'])
        classifier = grid_search.best_estimator_

        T_predict = classifier.predict(X_test)

        print('-' * 100)
        print('{} Testing accuracy: {:.2f}%.'.format(model_type['title'], 100*accuracy_score(y_test, T_predict)))

        print(grid_search.best_params_)
        print()

        print(confusion_matrix(y_test, T_predict))
        cm = confusion_matrix(y_test, T_predict)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        confusion = pd.DataFrame(cm, labels, labels)
        ax = plt.axes()
        heatmap = sn.heatmap(confusion, cmap=plt.get_cmap('Blues'), annot=True, ax=ax)
        ax.set_title(model_type['title'] + ' Heat Map')
        heatmap = heatmap.get_figure()
        heatmap.savefig(model_type['title'] + ' heat_map.png')
        print()

        print(classification_report(y_test, T_predict))


if __name__ == '__main__':
    classify_all(pd.read_csv("./Data-assignment-1/Landsat/lantsat.csv", header=None), 'LandSat')

import pandas as pd
import numpy as np
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from gridSearch import graph_grid_search
from matplotlib import pyplot as plt


def classify_all(data, data_name):

    X, y = data.drop(data.columns[-1], axis=1), data[data.columns[-1]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
        ]

    scoring = {'accuracy': 'accuracy', 'r2': 'r2'}
    for model_type in models:
        grid_search = graph_grid_search(model_type['title'], ['C', 'Score'], model_type['model'], model_type['params'],
                                        scoring, X_train, y_train, model_type['x_param'])
        classifier = grid_search.best_estimator_

        T_predict = classifier.predict(X_test)

        print('-' * 100)
        print('{} Testing accuracy: {:.2f}%.'.format(model_type['title'], 100*accuracy_score(y_test, T_predict)))

        print(grid_search.best_params_)
        print()

        print(confusion_matrix(T_predict, y_test))
        cm = confusion_matrix(T_predict, y_test)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        confusion = pd.DataFrame(cm, labels, labels)
        sn.heatmap(confusion, cmap=plt.get_cmap('Blues'), annot=True)
        print()

        print(classification_report(y_test, T_predict))


if __name__ == '__main__':
    classify_all(pd.read_csv("./Data-assignment-1/Landsat/lantsat.csv", header=None), 'LandSat')

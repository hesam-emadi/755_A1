import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from preprocessing import preprocessing
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from matplotlib import pyplot as plt
from graph import graph
import warnings
from gridSearch import graph_grid_search

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

if __name__ == '__main__':
    worldcup = pd.read_csv("./Data-assignment-1/World_Cup_2018/2018 worldcup.csv", index_col=0)
    cats = ['Location', 'Phase', 'Team1', 'Team2', 'Team1_Continent', 'Team2_Continent', 'Normal_Time']
    a, features = preprocessing(worldcup, cats)
    labels = features['Match_result']

    S = features.index
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    train_ind, test_ind = None, None
    for train_index, test_index in sss.split(features, labels):
        train_ind, test_ind = S[train_index], S[test_index]

    X_test = features.loc[test_ind].drop(['Match_result'], axis=1)
    y_test = features.loc[test_ind]['Match_result']
    X_train = features.loc[train_ind].drop(['Match_result'], axis=1)
    y_train = features.loc[train_ind]['Match_result']

    X = features.drop(['Match_result'], axis=1)
    y = features['Match_result']
    y = label_binarize(y, classes=['win', 'loss', 'draw'])
    # y = LabelEncoder().fit_transform(y)
    print(X.shape)
    print(y.shape)

    scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
    # ['accuracy', 'adjusted_mutual_info_score', 'adjusted_rand_score', 'average_precision', 'completeness_score',
    #  'explained_variance', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'fowlkes_mallows_score',
    #  'homogeneity_score', 'mutual_info_score', 'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_squared_error',
    #  'neg_mean_squared_log_error', 'neg_median_absolute_error', 'normalized_mutual_info_score', 'precision',
    #  'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro',
    #  'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc', 'v_measure_score']

    param_grid = [
            # {'kernel': ['rbf'], 'C': [2**x for x in range(0, 6)], 'gamma': [1e-3, 1e-4]},
            # {'kernel': ['poly'], 'C': [2 ** x for x in range(0, 6)], 'degree': [1, 2, 3, 4, 5, 6]},
            # {'kernel': ['linear'], 'C': [2 ** x for x in range(0, 6)]},
            # {'kernel': ['sigmoid'], 'C': [2 ** x for x in range(0, 6)]},
            {'estimator__C': [2**x for x in range(0, 6)], 'estimator__tol': [1e-3, 1e-4, 1e-5],
            }
        ]

    inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    a = OneVsRestClassifier(SVC())
    graph_grid_search('title', ['C', 'Score'], a, param_grid, scoring, X_train,
                      y_train, 'estimator__C')
    grid_search = GridSearchCV(a, param_grid, cv=3, scoring=scoring, refit='AUC', verbose=0)
    grid_search.fit(X, y)
    results = grid_search.cv_results_
    graph("a", ['a', 'b'], results, scoring, 'estimator__C')
    plt.figure(figsize=(13, 13))
    plt.title("GridSearchCV evaluating using multiple scorers simultaneously",
              fontsize=16)

    plt.xlabel("min_samples_split")
    plt.ylabel("Score")
    plt.grid()

    ax = plt.axes()
    # ax.set_xlim(0, 402)
    # ax.set_ylim(0.73, 1)

    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(results['param_estimator__C'].data, dtype=float)

    for scorer, color in zip(sorted(scoring), ['g', 'k']):
        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
            sample_score_std = results['std_%s_%s' % (sample, scorer)]
            ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                            sample_score_mean + sample_score_std,
                            alpha=0.1 if sample == 'test' else 0, color=color)
            ax.plot(X_axis, sample_score_mean, style, color=color,
                    alpha=1 if sample == 'test' else 0.7,
                    label="%s (%s)" % (scorer, sample))

        best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
        best_score = results['mean_test_%s' % scorer][best_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([X_axis[best_index], ] * 2, [0, best_score],
                linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score,
                    (X_axis[best_index], best_score + 0.005))

    plt.legend(loc="best")
    plt.grid('off')
    plt.show()

    # classifier = grid_search.best_estimator_
    #
    # T_predict = classifier.predict(X_test)
    #
    # print('-' * 100)
    # print("The prediction accuracy (tuned) for all testing sentence is : {:.2f}%."
    #       .format(100*accuracy_score(y_test, T_predict)))
    #
    # print(grid_search.best_params_)
    # print()
    #
    # print(confusion_matrix(y_test, T_predict))
    # print()
    #
    # print(classification_report(y_test, T_predict))
    #
    # # Trees
    # params = {
    #     'max_leaf_nodes': list(i**3 + 1 for i in range(1, 10, 2)),
    #     'min_samples_split': [i**2 + 1 for i in range(1, 10, 2)],
    #     'max_depth': [i**2 for i in range(1, 10, 5)],
    #     'max_features': ['sqrt', 'log2', None],
    #     'min_impurity_decrease': [0.01 * i for i in range(1, 10, 2)],
    #     'class_weight': ['balanced', None]
    # }
    # grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1, verbose=0)
    # grid_search_cv.fit(X_train, y_train)
    # classifier = grid_search_cv.best_estimator_
    #
    # T_predict = classifier.predict(X_test)
    #
    # print('-' * 100)
    # print("The prediction accuracy using the decision tree is : {:.2f}%."
    #       .format(100 * accuracy_score(y_test, T_predict)))
    # print(grid_search.best_params_)
    #
    #
    # print(confusion_matrix(y_test, T_predict))
    # print(classification_report(y_test, T_predict))
    #
    # # Perceptron
    # params = {}
    # grid_search_cv = GridSearchCV(Perceptron(random_state=42), params, n_jobs=-1, verbose=0)
    # grid_search_cv.fit(X_train, y_train)
    # classifier = grid_search_cv.best_estimator_
    #
    # T_predict = classifier.predict(X_test)
    #
    # print('-' * 100)
    # print("The prediction accuracy using the perceptron is : {:.2f}%."
    #       .format(100 * accuracy_score(y_test, T_predict)))
    # print(grid_search.best_params_)
    #
    #
    # print(confusion_matrix(y_test, T_predict))
    # print(classification_report(y_test, T_predict))
    #
    # # Multinominal Bayes
    # params = {}
    # grid_search_cv = GridSearchCV(GaussianNB(), params, n_jobs=-1, verbose=0)
    # grid_search_cv.fit(X_train, y_train)
    # classifier = grid_search_cv.best_estimator_
    #
    # T_predict = classifier.predict(X_test)
    #
    # print('-' * 100)
    # print("The prediction accuracy using the Multinominal Bayes is : {:.2f}%."
    #       .format(100 * accuracy_score(y_test, T_predict)))
    # print(grid_search.best_params_)
    #
    #
    # print(confusion_matrix(y_test, T_predict))
    # print(classification_report(y_test, T_predict))
    #
    # # Bernoulli Bayes
    # params = {}
    # grid_search_cv = GridSearchCV(BernoulliNB(), params, n_jobs=-1, verbose=0)
    # grid_search_cv.fit(X_train, y_train)
    # classifier = grid_search_cv.best_estimator_
    #
    # T_predict = classifier.predict(X_test)
    #
    # print('-' * 100)
    # print("The prediction accuracy using the Bernoulli Bayes is : {:.2f}%."
    #       .format(100 * accuracy_score(y_test, T_predict)))
    # print(grid_search.best_params_)
    #
    #
    # print(confusion_matrix(y_test, T_predict))
    # print(classification_report(y_test, T_predict))
    #
    # # KNeighbors
    # params = {}
    # grid_search_cv = GridSearchCV(KNeighborsClassifier(), params, n_jobs=-1, verbose=0)
    # grid_search_cv.fit(X_train, y_train)
    # classifier = grid_search_cv.best_estimator_
    #
    # T_predict = classifier.predict(X_test)
    #
    # print('-' * 100)
    # print("The prediction accuracy using the KNeighbors is : {:.2f}%."
    #       .format(100 * accuracy_score(y_test, T_predict)))
    # print(grid_search.best_params_)
    #
    #
    # print(confusion_matrix(y_test, T_predict))
    # print(classification_report(y_test, T_predict))
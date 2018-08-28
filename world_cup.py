import pandas as pd
import numpy as np
from preprocessing import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from preprocessing import preprocessing
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier

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

    Test_Matrix = features.loc[test_ind]
    Train_Matrix = features.loc[train_ind]

    param_grid = [
            {'kernel': ['rbf'], 'C': [2**x for x in range(0, 6)], 'gamma': [1e-3, 1e-4]},
            {'kernel': ['poly'], 'C': [2 ** x for x in range(0, 6)], 'degree': [1, 2, 3, 4, 5, 6]},
            {'kernel': ['linear'], 'C': [2 ** x for x in range(0, 6)]},
            {'kernel': ['sigmoid'], 'C': [2 ** x for x in range(0, 6)]},
        ]

    inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=inner_cv,  n_jobs=1, scoring='accuracy', verbose=0)
    grid_search.fit(Train_Matrix.drop(['Match_result'], axis=1), Train_Matrix['Match_result'])
    clf = grid_search.best_estimator_
    # data testing
    T_predict = clf.predict(Test_Matrix.drop(['Match_result'], axis=1))

    print('*******************************************************************')
    print("The prediction accuracy (tuned) for all testing sentence is : {:.2f}%."
          .format(100*accuracy_score(Test_Matrix['Match_result'], T_predict)))
    print(grid_search.best_params_)
    print(grid_search.param_grid)

    # data training without hyperparameter tuning
    clf = Pipeline([
            ('std_scaler', StandardScaler()),
            ("svm", SVC())
    ])
    clf.fit(Train_Matrix.drop(['Match_result'], axis=1), Train_Matrix['Match_result'])

    # data testing
    T_predict = clf.predict(Test_Matrix.drop(['Match_result'], axis=1))

    print('*******************************************************************')
    print("The prediction accuracy (untuned) for all testing sentence is : {:.2f}%."
          .format(100*accuracy_score(Test_Matrix['Match_result'], T_predict)))

    # Trees
    params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
    grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1, verbose=1)
    grid_search_cv.fit(Train_Matrix.drop(['Match_result'], axis=1), Train_Matrix['Match_result'])
    best = grid_search_cv.best_estimator_

    T_predict = best.predict(Test_Matrix.drop(['Match_result'], axis=1))

    print('*******************************************************************')
    print("The prediction accuracy using the decision tree is : {:.2f}%."
          .format(100 * accuracy_score(Test_Matrix['Match_result'], T_predict)))

    # Perceptron
    params = {}
    grid_search_cv = GridSearchCV(Perceptron(random_state=42), params, n_jobs=-1, verbose=1)
    grid_search_cv.fit(Train_Matrix.drop(['Match_result'], axis=1), Train_Matrix['Match_result'])
    best = grid_search_cv.best_estimator_

    T_predict = best.predict(Test_Matrix.drop(['Match_result'], axis=1))
    print("The prediction accuracy using the perceptron is : {:.2f}%."
          .format(100 * accuracy_score(Test_Matrix['Match_result'], T_predict)))

    # Multinominal Bayes
    params = {}
    grid_search_cv = GridSearchCV(GaussianNB(), params, n_jobs=-1, verbose=1)
    grid_search_cv.fit(Train_Matrix.drop(['Match_result'], axis=1), Train_Matrix['Match_result'])
    best = grid_search_cv.best_estimator_

    T_predict = best.predict(Test_Matrix.drop(['Match_result'], axis=1))
    print("The prediction accuracy using the Multinominal Bayes is : {:.2f}%."
          .format(100 * accuracy_score(Test_Matrix['Match_result'], T_predict)))

    # Bernoulli Bayes
    params = {}
    grid_search_cv = GridSearchCV(BernoulliNB(), params, n_jobs=-1, verbose=1)
    grid_search_cv.fit(Train_Matrix.drop(['Match_result'], axis=1), Train_Matrix['Match_result'])
    best = grid_search_cv.best_estimator_

    T_predict = best.predict(Test_Matrix.drop(['Match_result'], axis=1))
    print("The prediction accuracy using the Bernoulli Bayes is : {:.2f}%."
          .format(100 * accuracy_score(Test_Matrix['Match_result'], T_predict)))

    # KNeighbors
    params = {}
    grid_search_cv = GridSearchCV(KNeighborsClassifier(), params, n_jobs=-1, verbose=1)
    grid_search_cv.fit(Train_Matrix.drop(['Match_result'], axis=1), Train_Matrix['Match_result'])
    best = grid_search_cv.best_estimator_

    T_predict = best.predict(Test_Matrix.drop(['Match_result'], axis=1))
    print("The prediction accuracy using the KNeighbors is : {:.2f}%."
          .format(100 * accuracy_score(Test_Matrix['Match_result'], T_predict)))
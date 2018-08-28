import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    data = pd.read_csv("./Data-assignment-1/Landsat/lantsat.csv", header=None)

    X, y = data.drop(data.columns[-1], axis=1), data[data.columns[-1]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    tuned_parameters = [{'alpha': [0.1, 0.5, 1, 2, 4, 8], 'normalize':[True, False], 'fit_intercept': [True, False]}]

    model = SVC()
    grid = GridSearchCV(model, tuned_parameters)

    param_grid = [
            {'kernel': ['rbf'], 'C': [2**x for x in range(0, 6)], 'gamma': [1e-3, 1e-4]},
            {'kernel': ['poly'], 'C': [2 ** x for x in range(0, 6)], 'degree': [1, 2, 3, 4, 5, 6]},
            {'kernel': ['linear'], 'C': [2 ** x for x in range(0, 6)]},
            {'kernel': ['sigmoid'], 'C': [2 ** x for x in range(0, 6)]},
        ]

    inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=inner_cv,  n_jobs=1, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)
    clf = grid_search.best_estimator_

    T_predict = clf.predict(X_test)

    print('*******************************************************************')
    print("The prediction accuracy (tuned) for all testing sentence is : {:.2f}%."
          .format(100*accuracy_score(y_test, T_predict)))
    print(grid_search.best_params_)
    print(grid_search.param_grid)

    model = SVC()
    model.fit(X_train, y_train)

    T_predict = model.predict(X_test)

    print('*******************************************************************')
    print("The prediction accuracy (untuned) for all testing sentence is : {:.2f}%."
          .format(100*accuracy_score(y_test, T_predict)))

    # Trees
    params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
    grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1, verbose=1)
    grid_search_cv.fit(X_train, y_train)
    best = grid_search_cv.best_estimator_

    T_predict = best.predict(X_test)

    print('*******************************************************************')
    print("The prediction accuracy using the decision tree is : {:.2f}%."
          .format(100 * accuracy_score(y_test, T_predict)))

    # Perceptron
    params = {}
    grid_search_cv = GridSearchCV(Perceptron(random_state=42), params, n_jobs=-1, verbose=1)
    grid_search_cv.fit(X_train, y_train)
    best = grid_search_cv.best_estimator_

    T_predict = best.predict(X_test)
    print("The prediction accuracy using the perceptron is : {:.2f}%."
          .format(100 * accuracy_score(y_test, T_predict)))

    # Multinominal Bayes
    params = {}
    grid_search_cv = GridSearchCV(GaussianNB(), params, n_jobs=-1, verbose=1)
    grid_search_cv.fit(X_train, y_train)
    best = grid_search_cv.best_estimator_

    T_predict = best.predict(X_test)
    print("The prediction accuracy using the Multinominal Bayes is : {:.2f}%."
          .format(100 * accuracy_score(y_test, T_predict)))

    # Bernoulli Bayes
    params = {}
    grid_search_cv = GridSearchCV(BernoulliNB(), params, n_jobs=-1, verbose=1)
    grid_search_cv.fit(X_train, y_train)
    best = grid_search_cv.best_estimator_

    T_predict = best.predict(X_test)
    print("The prediction accuracy using the Bernoulli Bayes is : {:.2f}%."
          .format(100 * accuracy_score(y_test, T_predict)))

    # KNeighbors
    params = {}
    grid_search_cv = GridSearchCV(KNeighborsClassifier(), params, n_jobs=-1, verbose=1)
    grid_search_cv.fit(X_train, y_train)
    best = grid_search_cv.best_estimator_

    T_predict = best.predict(X_test)
    print("The prediction accuracy using the KNeighbors is : {:.2f}%."
          .format(100 * accuracy_score(y_test, T_predict)))
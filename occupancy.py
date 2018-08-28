import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    data = pd.read_csv("./Data-assignment-1/Occupancy_sensor/occupancy_sensor_data.csv")
    data['date'] = pd.to_datetime(data['date'])
    data['date'] = [x.time() for x in data['date']]

    data['hour'] = [x.hour for x in data['date']]
    data['minute'] = [x.minute for x in data['date']]
    data['second'] = [x.second for x in data['date']]
    data.drop(['date'], axis=1, inplace=True)

    X, y = data.drop(['Occupancy'], axis=1), data['Occupancy']
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
    grid_search = GridSearchCV(model, param_grid, cv=inner_cv,  n_jobs=1, scoring='accuracy', verbose=0)
    grid_search.fit(X_train, y_train)
    clf = grid_search.best_estimator_

    T_predict = clf.predict(X_test)

    print('*******************************************************************')
    print("The prediction accuracy (tuned) for all testing sentence is : {:.2f}%."
          .format(100*accuracy_score(y_test, T_predict)))
    print(grid_search.best_params_)
    print(grid_search.param_grid)


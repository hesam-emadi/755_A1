import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import explained_variance_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from graph import graph
import sys


def regression_test(test_data):
    data = pd.read_csv("./Data-assignment-1/Traffic_flow/traffic_flow_data.csv")
    cols = []

    # for i in range((data.shape[1]) // 45):
    #     for j in range(5):
    #         cols.append(45 * i + j + 20)
    # y = data['Segment23_(t+1)']
    # X = data[data.columns[cols]]

    # # get every segment 23
    # y = data['Segment23_(t+1)']
    # data = data.iloc[:, 22::45]
    # X = data

    # print(data.shape[1])
    # new = data.copy()
    # for i in range(45, data.shape[1] - 1):
    #     new[new.columns[i]] = data[data.columns[i]] - data[data.columns[i - 45]]

    # data = new[data.columns[44:]]

    X, y = data.drop('Segment23_(t+1)', axis=1), data['Segment23_(t+1)']
    X_test, y_test = test_data.drop('Segment23_(t+1)', axis=1), test_data['Segment23_(t+1)']
    X_train, a, y_train, a = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    tuned_parameters = [{'alpha': [i**2 / 100 for i in range(1,100,2)]}]

    model = Ridge()
    scoring = {'r2': 'r2'} #''mean_squared_error': 'neg_mean_squared_error'
    grid = GridSearchCV(model, tuned_parameters, scoring=scoring, refit='r2')

    grid.fit(X_train, y_train)
    print(grid.param_grid)

    results = grid.cv_results_

    graph('Traffic Flow', ['alpha', 'Score'], results, scoring, 'alpha')
    best = grid.best_estimator_

    # best = model.fit(X_train, y_train)
    predictions = best.predict(X_test)

    print('*******************************************************************')
    print("Ridge Regression Traffic Flow")
    print("Mean squared error: {}".format(mean_squared_error(y_test, predictions)))
    print("Explained variance: {}".format(explained_variance_score(y_test, predictions)))

    tuned_parameters = {}

    model = LinearRegression()
    grid = GridSearchCV(model, tuned_parameters)

    grid.fit(X_train, y_train)
    best = grid.best_estimator_

    # best = model.fit(X_train, y_train)
    predictions = best.predict(X_test)

    print('*******************************************************************')
    print("Linear Regression Traffic Flow")
    print("Mean squared error: {}".format(mean_squared_error(y_test, predictions)))
    print("Explained variance: {}".format(explained_variance_score(y_test, predictions)))


if __name__ == '__main__':
    regression_test(pd.read_csv(str(sys.argv[1])))
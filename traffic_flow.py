import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import explained_variance_score, mean_squared_error
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
    data = pd.read_csv("./Data-assignment-1/Traffic_flow/traffic_flow_data.csv")

    # get every segment 23
    # y = data['Segment23_(t+1)']
    # data = data.iloc[:, 22::45]
    # X = data

    print(data.shape[1])
    new = data.copy()
    for i in range(45, data.shape[1] - 1):
        new[new.columns[i]] = data[data.columns[i]] - data[data.columns[i - 45]]

    # print(new.head)

    data = new


    X, y = data.drop('Segment23_(t+1)', axis=1), data['Segment23_(t+1)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    tuned_parameters = [{'alpha': [0.1, 0.5, 1, 2, 4, 8], 'normalize':[True, False], 'fit_intercept': [True, False]}]

    model = Ridge()
    # grid = GridSearchCV(model, tuned_parameters)

    # grid.fit(X_train, y_train)
    # best = grid.best_estimator_

    best = model.fit(X_train, y_train)
    predictions = best.predict(X_test)

    print('*******************************************************************')
    print("Ridge Regression Traffic Flow")
    print("Mean squared error: {}".format(mean_squared_error(y_test, predictions)))

    tuned_parameters = [{'normalize':[True, False], 'fit_intercept': [True, False]}]

    model = LinearRegression()
    # grid = GridSearchCV(model, tuned_parameters)

    # grid.fit(X_train, y_train)
    # best = grid.best_estimator_

    best = model.fit(X_train, y_train)
    predictions = best.predict(X_test)

    print('*******************************************************************')
    print("Linear Regression Traffic Flow")
    print("Mean squared error: {}".format(mean_squared_error(y_test, predictions)))
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import explained_variance_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from graph import graph
from testall import classify, testing
import warnings
import sys

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def preprocess(worldcup, classification=True):
    categorical = ['Team1_Continent', 'Team2_Continent']
    categorical.sort()

    reversedcup = worldcup.copy()
    for col in worldcup.columns:
        if col.startswith("Team1"):
            col_2 = col.replace('1', '2')
            reversedcup[col] = worldcup[col_2]
            reversedcup[col_2] = worldcup[col]

    worldcup = worldcup.append(reversedcup, ignore_index=True)

    if classification:
        labels = worldcup['Match_result']
    else:
        labels = worldcup['Total_Scores']

    data = worldcup.drop(['Date', 'Match_result', 'Total_Scores', 'Phase', 'Location', 'Team2', 'Team1', 'Normal_Time'],
                         axis=1)

    cat_locs = [data.columns.get_loc(cat) for cat in categorical]
    # use a label encoder and then onehot encoder as the sategory_encoders package breaks the enviroment with seaborn
    for cat in categorical:
        encoder = LabelEncoder()
        col = data[cat]
        encoded = encoder.fit_transform(col)
        data[cat] = encoded
    encoder = OneHotEncoder(categorical_features=cat_locs, sparse=False)
    encoded = encoder.fit_transform(data)
    data = pd.DataFrame(encoded)

    if classification:
        y = pd.DataFrame(LabelEncoder().fit_transform(labels))[0]
    else:
        y = labels

    return data, y


def regression(worldcup):
    X, y = preprocess(worldcup, False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    tuned_parameters = [{'alpha': [i ** 2 / 100 for i in range(1, 500, 2)]}]

    model = Ridge()
    scoring = {'mean_squared_error': 'neg_mean_squared_error'}
    grid = GridSearchCV(model, tuned_parameters, scoring=scoring, refit='mean_squared_error')

    grid.fit(X_train, y_train)
    print(grid.param_grid)

    results = grid.cv_results_

    graph('World Cup', ['alpha', 'Score'], results, scoring, 'alpha')
    best = grid.best_estimator_

    # best = model.fit(X_train, y_train)
    predictions = best.predict(X_test)

    print('*******************************************************************')
    print("Ridge Regression World Cup")
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
    print("Linear Regression World Cup")
    print("Mean squared error: {}".format(mean_squared_error(y_test, predictions)))
    print("Explained variance: {}".format(explained_variance_score(y_test, predictions)))


def classification(worldcup):
    X, y = preprocess(worldcup, True)
    classify(X, y, 'World Cup 2018')


def preprocess_test(worldcup, test_data, classification=True):
    if classification:
        labels = worldcup['Match_result']
    else:
        labels = worldcup['Total_Scores']

    data = worldcup.drop(['Team1_Continent', 'Team2_Continent', 'Date', 'Match_result', 'Total_Scores', 'Phase', 'Location', 'Team2', 'Team1', 'Normal_Time'],
                         axis=1)
    labels_encoder = LabelEncoder()
    if classification:
        labels_encoder.fit(labels)
        y = pd.DataFrame(labels_encoder.transform(labels))[0]
        y_test = pd.DataFrame(labels_encoder.transform(test_data['Match_result']))[0]
    else:
        y = labels
        y_test = test_data['Total_Scores']

    x_test = test_data.drop(['Team1_Continent', 'Team2_Continent', 'Date', 'Match_result', 'Total_Scores', 'Phase', 'Location', 'Team2', 'Team1', 'Normal_Time'],
                         axis=1)

    return data, y, x_test, y_test


def classification_test(worldcup, test_data):
    X, y, x_test, y_test = preprocess_test(worldcup,test_data, True)
    testing('World cup classification', X, y, x_test, y_test)


def regression_test(worldcup, test_data):
    X, y, X_test, y_test = preprocess_test(worldcup, test_data, False)
    X_train, a, y_train, a = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    tuned_parameters = [{'alpha': [i ** 2 / 100 for i in range(1, 500, 2)]}]

    model = Ridge()
    scoring = {'mean_squared_error': 'neg_mean_squared_error'}
    grid = GridSearchCV(model, tuned_parameters, scoring=scoring, refit='mean_squared_error')

    grid.fit(X_train, y_train)
    print(grid.param_grid)

    results = grid.cv_results_

    graph('World Cup', ['alpha', 'Score'], results, scoring, 'alpha')
    best = grid.best_estimator_

    # best = model.fit(X_train, y_train)
    predictions = best.predict(X_test)

    print('*******************************************************************')
    print("Ridge Regression World Cup")
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
    print("Linear Regression World Cup")
    print("Mean squared error: {}".format(mean_squared_error(y_test, predictions)))
    print("Explained variance: {}".format(explained_variance_score(y_test, predictions)))


if __name__ == '__main__':
    worldcup = pd.read_csv("./Data-assignment-1/World_Cup_2018/2018 worldcup.csv", index_col=0)
    if sys.argv[1] == 'classification':
        classification_test(worldcup, pd.read_csv(str(sys.argv[2]), index_col=0))
    if sys.argv[1] == 'regression':
        regression_test(worldcup, pd.read_csv(str(sys.argv[2]), index_col=0))


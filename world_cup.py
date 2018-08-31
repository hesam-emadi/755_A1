import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import explained_variance_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from graph import graph
from testall import classify
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def preprocess(worldcup, classification=True):
    categorical = ['Team1_Continent', 'Team2_Continent']
    categorical.sort()

    if classification:
        labels = worldcup['Match_result']
    else:
        labels = worldcup['Total_Scores']

    data = worldcup.drop(['Date', 'Match_result', 'Total_Scores', 'Team2', 'Team1', 'Phase', 'Location', 'Normal_Time'],
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

    tuned_parameters = [{'alpha': [i ** 2 / 100 for i in range(50)]}]

    model = Ridge()
    scoring = {'r2': 'r2'}
    grid = GridSearchCV(model, tuned_parameters, scoring=scoring, refit='r2')

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


if __name__ == '__main__':
    worldcup = pd.read_csv("./Data-assignment-1/World_Cup_2018/2018 worldcup.csv", index_col=0)
    regression(worldcup)

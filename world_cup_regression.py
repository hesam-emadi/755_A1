import pandas as pd
import numpy as np
from preprocessing import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from preprocessing import preprocessing
from sklearn.linear_model import Ridge
from sklearn.metrics import explained_variance_score, mean_squared_error

if __name__ == '__main__':
    worldcup = pd.read_csv("./Data-assignment-1/World_Cup_2018/2018 worldcup.csv", index_col=0)
    cats = ['Location', 'Phase', 'Team1', 'Team2', 'Team1_Continent', 'Team2_Continent', 'Normal_Time']
    a, features = preprocessing(worldcup, cats)
    features.drop('Match_result', axis=1, inplace=True)
    labels = pd.DataFrame(features['Total_Scores'])

    X_train, X_test, y_train, y_test = train_test_split(features.drop(['Total_Scores'], axis=1),
                                                        features[['Total_Scores']], test_size=0.3)

    param_grid = [
        ]

    # inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    # grid_search = GridSearchCV(Ridge(random_state=42), param_grid,  n_jobs=1, scoring='accuracy', verbose=0)
    # grid_search.fit(X_train, y_train)
    grid_search = Ridge()
    clf = grid_search.fit(X_train, y_train)
    # clf = grid_search.best_estimator_
    # data testing
    T_predict = clf.predict(X_test)
    print(T_predict.shape)
    print(y_test.shape)
    print('*******************************************************************')
    print(mean_squared_error(y_test, T_predict))

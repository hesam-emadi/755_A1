import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from preprocessing import preprocessing


def extract_features(data_set):
    features = data_set.drop(["Date", "Location", "Phase", "Team1", "Team1_Continent",
                              "Team2", "Team2_Continent", "Total_Scores"], axis=1)
    labels = data_set["Match_result"]
    return features, labels


if __name__ == '__main__':
    data = pd.read_csv("./Data-assignment-1/World_Cup_2018/2018 worldcup.csv", index_col=0)
    categorical = ['Location','Phase','Team1','Team2','Team1_Continent','Team2_Continent','Normal_Time']
    features, results = preprocessing(data, categorical)
    print(results.head)

    labels = results.drop(["Total_Scores"], axis=1)
    # features, labels = extract_features(data)

    S = features.index
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
    # labels = LabelEncoder().fit(labels).transform(labels)
    # features["Normal_Time"] = LabelEncoder().fit(features["Normal_Time"]).transform(features["Normal_Time"])

    for train_index, test_index in sss.split(features, labels):
        train_ind, test_ind = S[train_index],S[test_index]
    Test_Matrix=features.loc[test_ind]
    Train_Matrix=features.loc[train_ind]


    # data training with hyperparameter tuning for C
    clf = Pipeline([
            ('std_scaler', StandardScaler()),
            ("svm", SVC())
    ])
    param_grid = [
            {'svm__kernel': ['rbf', 'sigmoid', 'poly'], 'svm__C': [2**x for x in range(0, 6)]},
        ]
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=1)
    grid_search = GridSearchCV(clf, param_grid, cv=inner_cv,  n_jobs=1, scoring='accuracy', verbose=0)
    grid_search.fit(Train_Matrix.drop(['Match_result'], axis=1), Train_Matrix['Match_result'])
    clf = grid_search.best_estimator_
    # data testing
    T_predict=clf.predict(Test_Matrix.drop(['Match_result'], axis=1))

    print('*******************************************************************')
    print("The prediction accuracy (tuned) for all testing sentence is : {:.2f}%."
          .format(100*accuracy_score(Test_Matrix['Match_result'],T_predict)))

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
    print("The prediction accuracy using the decision tree is : {:.2f}%."
          .format(100 * accuracy_score(Test_Matrix['Match_result'], T_predict)))
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from graph import graph


def graph_grid_search(title, labels,  model, param_grid, scoring, X, y, x_param):
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=inner_cv, n_jobs=1, scoring=scoring, refit='accuracy',
                               verbose=1, return_train_score=True)
    grid_search.fit(X, y)

    results = grid_search.cv_results_

    graph(title, labels, results, scoring, x_param)

    return grid_search

import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import category_encoders as cs
from sklearn.pipeline import FeatureUnion

# Create a class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames in this wise manner yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


def preprocessing(data, categorical):
    data.drop(['Date','Team1_Ball_Possession(%)'], axis=1, inplace=True)
    data.describe()

    #world cup attributes
    w_features = data.iloc[:,np.arange(26)].copy()
    #world cup goal result
    w_goals = data.iloc[:,26].copy()
    #wordl cup match result
    w_results = data.iloc[:,27].copy()

    #  w_features_num: numerical features
    #  w_features_cat: categorical features 
    w_features_num = w_features.drop(categorical, axis=1, inplace=False)
    w_features_cat=w_features[categorical].copy()

    num_pipeline = Pipeline([
            ('selector', DataFrameSelector(list(w_features_num))),
            ('imputer', Imputer(strategy="median")),
            ('std_scaler', StandardScaler()),
        ])

    cat_pipeline = Pipeline([
            ('selector', DataFrameSelector(list(w_features_cat))),
            ('cat_encoder', cs.OneHotEncoder(drop_invariant=True)),
        ])

    full_pipeline = FeatureUnion(transformer_list=[
            ("num_pipeline", num_pipeline),
            ("cat_pipeline", cat_pipeline),
        ])


    feature_prepared = pd.DataFrame(data=full_pipeline.fit_transform(w_features),index=np.arange(1,65))
    cleaned=pd.concat([feature_prepared,w_goals.to_frame(), w_results.to_frame()], axis=1)

    return feature_prepared, cleaned

if __name__ == '__main__':
    worldcup=pd.read_csv("./Data-assignment-1/World_Cup_2018/2018 worldcup.csv", index_col=0)
    categorical = ['Location','Phase','Team1','Team2','Team1_Continent','Team2_Continent','Normal_Time']

    print(preprocessing(worldcup, categorical))

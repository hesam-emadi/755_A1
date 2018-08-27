import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

data = pd.read_csv("./Data-assignment-1/World_Cup_2018/2018 worldcup.csv")
print(data.shape)

def extract_features(data_set):
    features = data_set.drop(["Match_ID", "Date", "Location", "Phase", "Team1", "Team1_Continent",
                              "Team2", "Team2_Continent", "Total_Scores", "Match_result"], axis=1)
    labels = data_set["Match_result"]
    return features, labels


features, labels = extract_features(data)
labels = LabelEncoder().fit(labels).transform(labels)
features["Normal_Time"] = LabelEncoder().fit(features["Normal_Time"]).transform(features["Normal_Time"])

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.10)

# Linear kernel
classifier = SVC(kernel='linear')
classifier.fit(features_train, labels_train)
predictions = classifier.predict(features_test)
print(confusion_matrix(labels_test, predictions))
print(classification_report(labels_test, predictions))
print("The prediction accuracy linear for all testing sentence is : {:.2f}%."
      .format(100*accuracy_score(labels_test, predictions)))

# Linear SVM
classifier = LinearSVC(loss="hinge")
classifier.fit(features_train, labels_train)
predictions = classifier.predict(features_test)
print(confusion_matrix(labels_test, predictions))
print(classification_report(labels_test, predictions))

print("The prediction accuracy linear hinge for all testing sentence is : {:.2f}%."
      .format(100*accuracy_score(labels_test, predictions)))
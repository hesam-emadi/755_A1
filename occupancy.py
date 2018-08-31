import pandas as pd
from testall import classify, testing
import sys

def preprocess(data):
    data['date'] = pd.to_datetime(data['date'])
    data['date'] = [x.time() for x in data['date']]

    data['hour'] = [x.hour for x in data['date']]
    data['minute'] = [x.minute for x in data['date']]
    data['second'] = [x.second for x in data['date']]
    data.drop(['date'], axis=1, inplace=True)

    X, y = data.drop(['Occupancy'], axis=1), data['Occupancy']
    return X, y


if __name__ == '__main__':
    data = pd.read_csv("./Data-assignment-1/Occupancy_sensor/occupancy_sensor_data.csv")
    X, y = preprocess(data)
    X_test, y_test = preprocess(pd.read_csv(str(sys.argv[1])))
    testing("Occupancy", X, y, X_test, y_test)


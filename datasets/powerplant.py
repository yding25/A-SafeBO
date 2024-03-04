import os

import numpy as np
import pandas as pd

# set random seed for reproducibility
np.random.seed(42)

from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor


def load_data(newaxis=False):
    base_dir = "./datasets/"
    excel_file = "Folds5x2_pp.xlsx"
    excel_dir = os.path.join(base_dir, excel_file)

    variables = ["AT", "V", "AP", "RH", "PE"]
    data = pd.DataFrame([])
    sheets = [
        pd.read_excel(
            excel_dir,
            sheet_name=f"Sheet{i}",
            header=0,
            names=variables,
            dtype=dict.fromkeys(variables, float),
            engine="openpyxl",
        )
        for i in range(1, 6, 1)
    ]
    data = pd.concat(sheets, axis=0).reset_index(inplace=False, drop=True)

    # save if necessary
    # data.to_csv('./allCCPP.csv', index=False)

    X, y = data[variables[0:-1]], data[variables[-1]]
    X, y = np.array(X), np.array(y)
    if newaxis:
        y = np.array(y)[:, np.newaxis]

    return X, y


def train(X, y):
    model = BaggingRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=10)
    model.fit(X, y)
    return model


def predict(model, X):
    return model.predict(X)


if __name__ == "__main__":
    X, y = load_data()
    model = train(X, y)

    # sample: [min AT, min V, min AP, min RH]
    # inputs = np.array([1.81, 25.36, 992.89, 25.56]).reshape(1,-1)
    # output = predict(model, inputs)
    # print('sample output:', output)

    # find X (=[AT, V, AP, RH]) corresponding to the min/max y (=PE)
    variables = ["AT", "V", "AP", "RH", "PE"]
    step = 1
    ATs = np.arange(X[:, 0].min(), X[:, 0].max(), step=step)
    Vs = np.arange(X[:, 1].min(), X[:, 1].max(), step=step)
    APs = np.arange(X[:, 2].min(), X[:, 2].max(), step=step)
    RHs = np.arange(X[:, 3].min(), X[:, 3].max(), step=step)

    print("AT:", X[:, 0].min(), X[:, 0].max())
    print("V:", X[:, 1].min(), X[:, 1].max())
    print("AP:", X[:, 2].min(), X[:, 2].max())
    print("RH:", X[:, 3].min(), X[:, 3].max())
    print("total:", len(X[:, 0]))

    max_value = -np.inf
    min_value = np.inf
    min_x, max_x = None, None
    for AT in ATs:
        for V in Vs:
            for AP in APs:
                for RH in RHs:
                    print("AT:", AT, "V:", V, "AP:", AP, "RH:", RH)
                    inputs = np.array([AT, V, AP, RH]).reshape(1, -1)
                    prediction = predict(model, inputs)
                    if prediction < min_value:
                        min_value = prediction
                        min_x = inputs
                    if prediction > max_value:
                        max_value = prediction
                        max_x = inputs
    print(f"min {min_value} at {min_x}")
    print(f"max {max_value} at {max_x}")

    # min [423.083] at [[22.81, 59.36, 1020.89, 90.56]]
    # max [495.281] at [[5.81, 40.36, 1019.89, 25.56]]

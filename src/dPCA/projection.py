import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def calc_projection(
    switch_df: pd.DataFrame, variable: np.ndarray, fr: np.ndarray, window_size: int = 30
) -> np.ndarray:
    hilo_mask = switch_df["subtype"] == 1
    lohi_mask = switch_df["subtype"] == -1
    projection_matrix = []

    for neuron in range(fr.shape[2]):
        # initialize model fit params
        projections = np.zeros((window_size, 2))
        coefficients = np.zeros(window_size)
        intercepts = np.zeros(window_size)

        # fit regression per time point, instance are each switch
        for t in range(window_size):
            X = variable[:, t].reshape(-1, 1)
            y = fr[:, t, neuron].reshape(-1, 1)

            model = LinearRegression()
            model.fit(X, y)

            coefficients[t] = model.coef_[0]
            intercepts[t] = model.intercept_

        hilo_speed_projs = variable[hilo_mask, :].mean(axis=0) * coefficients
        lohi_speed_projs = variable[lohi_mask, :].mean(axis=0) * coefficients
        projections[:, 0] = hilo_speed_projs
        projections[:, 1] = lohi_speed_projs
        projection_matrix.append(projections)

    projection_matrix = np.dstack(projection_matrix).transpose(2, 0, 1)
    return projection_matrix

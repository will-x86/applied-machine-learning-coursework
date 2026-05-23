# pyright: basic
from typing import Any

import numpy as np
import pandas as pd


def load_data_train(path_train: str, path_val: str, path_test: str) -> Any:
    data_train = pd.read_csv(path_train)

    # extract the text
    text_train = data_train["text"].values
    # and the labels
    labels_train = data_train["label"].values
    print(
        f"Train: \n Data shape: {data_train.shape}\nTextShape: {text_train.shape}, LabelsShape{labels_train.shape}"
    )

    data_val = pd.read_csv(path_val)
    # extract the text
    text_val = data_val["text"].values
    # and the labels
    labels_val = data_val["label"].values
    print(
        f"Val: \n Data shape: {data_val.shape}\nTextShape: {text_val.shape}, LabelsShape{labels_val.shape}"
    )

    data_test = pd.read_csv(path_test)
    # extract the text
    text_test = data_test["text"].values
    print(f"Test: \n Data shape: {data_test.shape}\nTextShape: {text_test.shape}")
    print("a" * 60)
    return text_train, labels_train, text_val, labels_val, text_test


def save_as_csv(pred_labels, location="."):
    """
    Save the labels out as a .csv file
    :pred_labels: numpy array of shape (no_test_labels,) to be saved
    :param location: Directory to save results.csv in. Default to current working directory
    """
    assert (
        pred_labels.shape[0] == 1434
    ), "wrong number of labels, should be 1434 test labels"
    np.savetxt(location + "/results_task1.csv", pred_labels, delimiter=",")

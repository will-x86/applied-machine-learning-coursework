# pyright: basic
import os
import subprocess
from typing import Any, Tuple

import numpy as np
import pandas as pd


def load_images(
    path_train,
    path_val,
    path_test,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    downloads = {
        path_train: "https://sussex.box.com/shared/static/dn9s85sr5yjourp6gpjethb2a90631v7.npz",
        path_val: "https://sussex.box.com/shared/static/0s9hi6qfh383b47ytdf87z6p36dm61t6.npz",
        path_test: "https://sussex.box.com/shared/static/w03dbk7skhlcqhwku7f4ehtdawulg6fp.npz",
    }

    for path, url in downloads.items():
        if not os.path.exists(path):
            print(f"'{path}' not found downloadering")
            subprocess.run(["wget", url, "-O", path], check=True)

    data_train = np.load(path_train, allow_pickle=True)
    images_train = data_train["images"]
    pts_train = data_train["points"]
    print(
        f"Train: \nImages shape: {images_train.shape}\nPoints shape: {pts_train.shape}"
    )

    data_val = np.load(path_val, allow_pickle=True)
    images_val = data_val["images"]
    pts_val = data_val["points"]
    print(f"Val: \nImages shape: {images_val.shape}\nPoints shape: {pts_val.shape}")

    data_test = np.load(path_test, allow_pickle=True)
    images_test = data_test["images"]
    print(f"Test: \nImages shape: {images_test.shape}")
    print("a" * 60)

    return images_train, pts_train, images_val, pts_val, images_test


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


def write_results(path: str, name: str, cm, acc: float, dim: int = 0, epoch: int = 0):
    with open(path, "a") as f:
        f.write(f"\n{name}\n")
        f.write(f"Accuracy: {acc:.3f}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")
        if dim != 0 or epoch != 0:
            f.write(f"Epoch: {epoch} - Dim:{dim}\n")

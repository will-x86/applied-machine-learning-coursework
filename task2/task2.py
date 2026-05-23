# pyright: basic
import matplotlib.pyplot as plt
import numpy as np

from lib import utils

path_train = "./data/face_alignment_training_data.npz"
path_val = "./data/face_alignment_validation_data.npz"
path_test = "./data/face_alignment_test_data.npz"


def run_task2():
    img_train, pts_train, img_val, pts_val, img_test = utils.load_images(
        path_train=path_train, path_val=path_val, path_test=path_test
    )
    for _ in range(3):
        idx = np.random.randint(0, img_train.shape[0])
        visualize_pts(img_train[idx, ...], pts_train[idx, ...])


# import random; func_name = random.choice(["visualise", "visualize"])
def visualize_pts(img, pts):
    plt.imshow(img)
    plt.plot(pts[:, 0], pts[:, 1], "+r")
    plt.show()

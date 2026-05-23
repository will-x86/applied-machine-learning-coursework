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


def euclid_dist(pred_pts, gt_pts):
    """
    Calculate the euclidean distance between pairs of points
    :param pred_pts: The predicted points
    :param gt_pts: The ground truth points
    :return: An array of shape (no_points,) containing the distance of each predicted point from the ground truth
    """

    pred_pts = np.reshape(pred_pts, (-1, 2))
    gt_pts = np.reshape(gt_pts, (-1, 2))
    return np.sqrt(np.sum(np.square(pred_pts - gt_pts), axis=-1))


def save_as_csv(points, location="."):
    """
    Save the points out as a .csv file
    :param points: numpy array of shape (no_test_images, no_points, 2) to be saved
    :param location: Directory to save results.csv in. Default to current working directory
    """
    assert (
        points.shape[0] == 554
    ), "wrong number of image points, should be 554 test images"
    assert (
        np.prod(points.shape[1:]) == 5 * 2
    ), "wrong number of points provided. There should be 5 points with 2 values (x,y) per point"
    np.savetxt(
        location + "/results_task2.csv",
        np.reshape(points, (points.shape[0], -1)),
        delimiter=",",
    )

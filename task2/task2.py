# pyright: basic
import cv2
import numpy as np
from PIL import Image, ImageDraw

from lib import utils

path_train = "./data/face_alignment_training_data.npz"
path_val = "./data/face_alignment_validation_data.npz"
path_test = "./data/face_alignment_test_data.npz"


def preprocess_image(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (96, 96), interpolation=cv2.INTER_LINEAR).astype(np.uint8)


def print_images_stats():
    img_train, pts_train, img_val, pts_val, img_test = utils.load_images(
        path_train=path_train, path_val=path_val, path_test=path_test
    )

    print(f"img_train:  {img_train.shape}")  # 2600, 256 256
    print(f"pts_train:  {pts_train.shape}")  # 211, 5, 2
    print(f"img_val:    {img_val.shape}")  # 211, 256, 256, 3
    print(f"pts_val:    {pts_val.shape}")  # 211, 5, 2
    print(f"img_test:   {img_test.shape}")  # 544, 256, 256 ,3

    print(f"dtype:      {img_train.dtype}")  # uint8
    print(f"mean/std:   {img_train.mean():.2f} / {img_train.std():.2f}")  # 99.18/72.73

    print(f"pts dtype:  {pts_train.dtype}")  # float64
    print(
        f"x  min/max: {pts_train[..., 0].min():.3f} / {pts_train[..., 0].max():.3f}"
    )  # 48.891 - 207.775
    print(
        f"y  min/max: {pts_train[..., 1].min():.3f} / {pts_train[..., 1].max():.3f}"
    )  # 62.831 - 210.659
    print(
        f"x  mean/std:{pts_train[..., 0].mean():.2f} / {pts_train[..., 0].std():.2f}"
    )  # 128.60 - 37.68
    print(
        f"y  mean/std:{pts_train[..., 1].mean():.2f} / {pts_train[..., 1].std():.2f}"
    )  # 140.61 - 33.95
    # all 255x255..


def run_task2():
    img_train, pts_train, img_val, pts_val, img_test = utils.load_images(
        path_train=path_train, path_val=path_val, path_test=path_test
    )
    print_images_stats()

    for i in range(3):
        idx = np.random.randint(0, img_train.shape[0])
        visualize_pts(img_train[idx, ...], pts_train[idx, ...], i)


# import random; func_name = random.choice(["visualise", "visualize"])
def visualize_pts(img, pts, i):
    # I hate linux, wayland, nixos, everything.
    # USing PIL over plt as it's broken
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    for x, y in pts:
        r = 5
        draw.line([(x - r, y), (x + r, y)], fill="red", width=2)
        draw.line([(x, y - r), (x, y + r)], fill="red", width=2)
    pil_img.save(f"out_{i}.png")


# def visualize_pts(img, pts, i):
#    plt.imshow(img)
#    plt.plot(pts[:, 0], pts[:, 1], "+r")
#    plt.savefig(f"out_{i}.png")
#    plt.show()


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

# pyright: basic

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lib import utils
from netofneural.pictureoffishingnet import predict, train

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


def save_grid():
    nrows = 10
    ncols = 10
    img_train, pts_train, *_ = utils.load_images(
        path_train=path_train, path_val=path_val, path_test=path_test
    )
    indices = np.linspace(0, len(img_train) - 1, nrows * ncols, dtype=int)
    H, W = img_train.shape[1], img_train.shape[2]
    grid = Image.new("RGB", (ncols * W, nrows * H), (30, 30, 30))
    for i, idx in enumerate(indices):
        cell = Image.fromarray(img_train[idx])
        draw = ImageDraw.Draw(cell)
        for x, y in pts_train[idx]:
            draw.line([(x - 5, y), (x + 5, y)], fill="red", width=2)
            draw.line([(x, y - 5), (x, y + 5)], fill="red", width=2)
        grid.paste(cell, (i % ncols * W, i // ncols * H))
    grid.save("grid.png")


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


def _point_errors(pred_pts, gt_pts):
    # returns (N, 5) euclidean distances in pixels
    return np.sqrt(np.sum((pred_pts - gt_pts) ** 2, axis=-1))


def plot_ced(mean_errors, path="ced.png"):
    thresholds = np.linspace(0, mean_errors.max() * 1.05, 500)
    frac = [(mean_errors <= t).mean() for t in thresholds]

    _, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, frac, linewidth=2)
    ax.set_xlabel("Mean euclidean error (pixels)")
    ax.set_ylabel("Fraction of images")
    ax.set_title("Cumulative Error Distribution")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[ced] saved {path}")


def plot_errors_boxplot(errors, path="errors_boxplot.png"):
    # errors: (N, 5)
    _, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(
        errors, tick_labels=[f"p{i}" for i in range(errors.shape[1])], notch=False
    )
    ax.set_xlabel("Landmark")
    ax.set_ylabel("Euclidean error (pixels)")
    ax.set_title("Per-landmark error distribution (validation set)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[boxplot] saved {path}")


def save_qualitative_grid(img_val, pts_val, pred_pts, n=16, path="qualitative.png"):
    ncols = 8
    nrows = n // ncols
    H, W = img_val.shape[1], img_val.shape[2]
    indices = np.linspace(0, len(img_val) - 1, n, dtype=int)
    grid = Image.new("RGB", (ncols * W, nrows * H), (30, 30, 30))
    for i, idx in enumerate(indices):
        cell = Image.fromarray(img_val[idx])
        draw = ImageDraw.Draw(cell)
        r = 5
        for x, y in pts_val[idx]:
            draw.line([(x - r, y), (x + r, y)], fill="red", width=2)
            draw.line([(x, y - r), (x, y + r)], fill="red", width=2)
        for x, y in pred_pts[idx]:
            draw.line([(x - r, y), (x + r, y)], fill="lime", width=2)
            draw.line([(x, y - r), (x, y + r)], fill="lime", width=2)
        grid.paste(cell, (i % ncols * W, i // ncols * H))
    grid.save(path)
    print(f"[qualitative] saved {path}  (red=GT, green=pred)")


def save_failure_cases(img_val, pts_val, pred_pts, n=16, path="failures.png"):
    errors = _point_errors(pred_pts, pts_val).mean(axis=1)
    worst = np.argsort(errors)[::-1][:n]

    ncols = 8
    nrows = n // ncols
    H, W = img_val.shape[1], img_val.shape[2]
    grid = Image.new("RGB", (ncols * W, nrows * H), (30, 30, 30))
    for i, idx in enumerate(worst):
        cell = Image.fromarray(img_val[idx])
        draw = ImageDraw.Draw(cell)
        r = 5
        for x, y in pts_val[idx]:
            draw.line([(x - r, y), (x + r, y)], fill="red", width=2)
            draw.line([(x, y - r), (x, y + r)], fill="red", width=2)
        for x, y in pred_pts[idx]:
            draw.line([(x - r, y), (x + r, y)], fill="lime", width=2)
            draw.line([(x, y - r), (x, y + r)], fill="lime", width=2)
        grid.paste(cell, (i % ncols * W, i // ncols * H))
    grid.save(path)
    print(f"[failures] saved {path}  worst errors: {errors[worst[:4]].round(1)}")


def robustness_analysis(model, img_val, pts_val, path="robustness.png"):
    noise_levels = [0, 5, 10, 20, 30, 50]
    mean_errors = []

    for sigma in noise_levels:
        if sigma == 0:
            noisy = img_val
        else:
            noise = np.random.normal(0, sigma, img_val.shape)
            noisy = np.clip(img_val.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        pred = predict(model, noisy)
        err = _point_errors(pred, pts_val).mean()
        mean_errors.append(err)
        print(f"[robustness] sigma={sigma:3d} | mean_error={err:.2f}px")

    _, ax = plt.subplots(figsize=(7, 5))
    ax.plot(noise_levels, mean_errors, marker="o", linewidth=2)
    ax.set_xlabel("Gaussian noise σ (pixel intensity)")
    ax.set_ylabel("Mean euclidean error (pixels)")
    ax.set_title("Robustness to Gaussian noise")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[robustness] saved {path}")


def run_task2():
    img_train, pts_train, img_val, pts_val, img_test = utils.load_images(
        path_train=path_train, path_val=path_val, path_test=path_test
    )
    print_images_stats()
    save_grid()

    model = train(img_train, pts_train, img_val, pts_val, epochs=100)

    pred_val = predict(model, img_val)
    errors = _point_errors(pred_val, pts_val)  # (N, 5)
    mean_errors = errors.mean(axis=1)  # (N,)
    print(
        f"\n[eval] mean error: {mean_errors.mean():.2f}px | median: {np.median(mean_errors):.2f}px"
    )

    plot_ced(mean_errors)
    plot_errors_boxplot(errors)
    save_qualitative_grid(img_val, pts_val, pred_val)
    save_failure_cases(img_val, pts_val, pred_val)
    robustness_analysis(model, img_val, pts_val)

    pred_test = predict(model, img_test)
    save_as_csv(pred_test)

    for i in range(3):
        idx = np.random.randint(0, img_train.shape[0])
        visualize_pts(img_train[idx, ...], pts_train[idx, ...], i)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os


def read_img(img_path):
    """
        Read grayscale image
        Inputs:
        img_path: str: image path
        Returns:
        img: cv2 image
    """
    return cv2.imread(img_path, 0)


def padding_img(img, filter_size=3):
    """
    The surrogate function for the filter functions.
    The goal of the function: replicate padding the image such that when applying the kernel with the size of filter_size, the padded image will be the same size as the original image.
    Inputs:
        img: np.ndarray: original image
        filter_size: int: size of square filter
    Return:
        padded_img: np.ndarray: the padded image
    """
    # Calculate padding size for each dimension
    pad_size = filter_size // 2  # Integer division

    # Get the dimensions of the original image
    height, width = img.shape

    # Create a new array with dimensions larger than the original image
    padded_img = np.zeros((height + 2 * pad_size, width + 2 * pad_size), dtype=img.dtype)

    # Copy the original image into the center of the new array
    padded_img[pad_size:pad_size + height, pad_size:pad_size + width] = img

    return padded_img


def mean_filter(img, filter_size=3):
    """
    Smoothing image with mean square filter with the size of filter_size. Use replicate padding for the image.
    Inputs:
        img: np.ndarray: original image
        filter_size: int: size of square filter
    Return:
        smoothed_img: np.ndarray: the smoothed image with mean filter
    """
    padded_img = padding_img(img, filter_size)
    height, width = img.shape
    smoothed_img = np.zeros_like(img, dtype=np.float32)

    for i in range(height):
        for j in range(width):
            # Extract the region of interest
            roi = padded_img[i:i + filter_size, j:j + filter_size]
            # Calculate the mean of the region and assign it to the output image
            smoothed_img[i, j] = np.mean(roi)

    return smoothed_img.astype(np.uint8)


def median_filter(img, filter_size=3):
    """
    Smoothing image with median square filter with the size of filter_size. Use replicate padding for the image.
    Inputs:
        img: np.ndarray: original image
        filter_size: int: size of square filter
    Return:
        smoothed_img: np.ndarray: the smoothed image with median filter
    """
    padded_img = padding_img(img, filter_size)
    height, width = img.shape
    smoothed_img = np.zeros_like(img, dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            # Extract the region of interest
            roi = padded_img[i:i + filter_size, j:j + filter_size]
            # Calculate the median of the region and assign it to the output image
            smoothed_img[i, j] = np.median(roi)

    return smoothed_img


def psnr(gt_img, smooth_img):
    """
    Calculate the PSNR metric
    Inputs:
        gt_img: np.ndarray: groundtruth image
        smooth_img: np.ndarray: smoothed image
    Outputs:
        psnr_score: float: PSNR score
    """
    # Ensure images are of the same data type
    gt_img = gt_img.astype(np.float32)
    smooth_img = smooth_img.astype(np.float32)

    # Calculate the mean squared error (MSE)
    mse = np.mean((gt_img - smooth_img) ** 2)

    # Maximum possible pixel value in the images
    max_pixel = 255.0

    # Calculate PSNR using the formula: PSNR = 20 * log10(MAX) - 10 * log10(MSE)
    psnr_score = 20 * np.log10(max_pixel) - 10 * np.log10(mse)

    return psnr_score


def show_res(before_img, after_img):
    """
        Show the original image and the corresponding smooth image
        Inputs:
            before_img: cv2: image before smoothing
            after_img: cv2: corresponding smoothed image
        Return:
            None
    """
    plt.figure(figsize=(12, 9))
    plt.subplot(1, 2, 1)
    plt.imshow(before_img, cmap='gray')
    plt.title('Before')

    plt.subplot(1, 2, 2)
    plt.imshow(after_img, cmap='gray')
    plt.title('After')
    plt.show()


if __name__ == '__main__':
    img_noise = r"D:\ex1_images\noise.png"  # <- need to specify the path to the noise image
    img_gt = r"D:\ex1_images\noise.png"  # <- need to specify the path to the gt image
    img = read_img(img_noise)
    filter_size = 3

    # Mean filter
    mean_smoothed_img = mean_filter(img, filter_size)
    show_res(img, mean_smoothed_img)
    print('PSNR score of mean filter: ', psnr(img, mean_smoothed_img))

    # Median filter
    median_smoothed_img = median_filter(img, filter_size)
    show_res(img, median_smoothed_img)
    print('PSNR score of median filter: ', psnr(img, median_smoothed_img))

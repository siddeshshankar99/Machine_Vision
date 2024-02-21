#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Automatic hysteresis thresholding

Author: Siddesh Bramarambika Shankar
MatrNr: 12329513
"""

import cv2
import numpy as np
from sobel import sobel
from non_max import non_max
from hyst_thresh import hyst_thresh
from blur_gauss import blur_gauss

def hyst_thresh_auto(edges_in: np.array, low_prop: float, high_prop: float) -> np.array:
    """ Apply automatic hysteresis thresholding.

    Apply automatic hysteresis thresholding by automatically choosing the high and low thresholds of standard
    hysteresis threshold. low_prop is the proportion of edge pixels which are above the low threshold and high_prop is
    the proportion of pixels above the high threshold.

    :param edges_in: Edge strength of the image in range [0., 1.]
    :type edges_in: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param low_prop: Proportion of pixels which should lie above the low threshold
    :type low_prop: float in range [0., 1.]

    :param high_prop: Proportion of pixels which should lie above the high threshold
    :type high_prop: float in range [0., 1.]

    :return: Binary edge image
    :rtype: np.array with shape (height, width) with dtype = np.uint8 and values either 0 or 255
    """

    # Calculate the low and high thresholds based on the given proportions
    low_threshold = np.percentile(edges_in, low_prop * 100)
    high_threshold = np.percentile(edges_in, high_prop * 100)

    # Call the hyst_thresh function with the calculated thresholds
    binary_edge_image = hyst_thresh(edges_in, low_threshold, high_threshold)

    return binary_edge_image

# Test the function
if __name__ == "__main__":
    # Load the input image
    input_image = cv2.imread('image/rubens.jpg', cv2.IMREAD_GRAYSCALE)

    # Define the sigma value for Gaussian blur
    sigma = 1.5

    # Apply Gaussian blur to the image
    img = blur_gauss(input_image, sigma)

    # Convert the loaded image to a NumPy array and normalize it
    img = np.float32(img) / 255

    # Apply Sobel edge detection to the image
    gradient, orientation = sobel(img)

    # Apply non-maximum suppression to get the edges
    edges = non_max(gradient, orientation)

    # Apply automatic hysteresis thresholding to get the binary edge image
    low_prop = 0.8
    high_prop = 0.97
    binary_edge_image = hyst_thresh_auto(edges, low_prop, high_prop)

    # Create a window for the binary edge image
    cv2.namedWindow('Binary Edge Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Binary Edge Image', binary_edge_image)

    # Wait for a key press and close all OpenCV windows on key press
    cv2.waitKey(0)

    # Close the OpenCV windows
    cv2.destroyAllWindows()


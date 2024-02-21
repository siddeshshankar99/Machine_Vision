#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Corner detection with the Harris corner detector

Author: Siddesh Bramarambika Shankar
MatrNr: 12329513
"""
import numpy as np
import cv2
from math import ceil

from typing import List

from helper_functions import non_max


def harris_corner(img: np.ndarray,
                  sigma1: float,
                  sigma2: float,
                  k: float,
                  threshold: float) -> List[cv2.KeyPoint]:
    """ Detect corners using the Harris corner detector

    In this function, corners in a grayscale image are detected using the Harris corner detector.
    They are returned in a list of OpenCV KeyPoints (https://docs.opencv.org/4.x/d2/d29/classcv_1_1KeyPoint.html).
    Each KeyPoint includes the attributes, pt (position), size, angle, response. The attributes size and angle are not
    relevant for the Harris corner detector and can be set to an arbitrary value. The response is the result of the
    Harris corner formula.

    :param img: Grayscale input image
    :type img: np.ndarray with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param sigma1: Sigma for the first Gaussian filtering
    :type sigma1: float

    :param sigma2: Sigma for the second Gaussian filtering
    :type sigma2: float

    :param k: Coefficient for harris formula
    :type k: float

    :param threshold: corner threshold
    :type threshold: float

    :return: keypoints:
        corners: List of cv2.KeyPoints containing all detected corners after thresholding and non-maxima suppression.
            Each keypoint has the attribute pt[x, y], size, angle, response.
                pt: The x, y position of the detected corner in the OpenCV coordinate convention.
                size: The size of the relevant region around the keypoint. Not relevant for Harris and is set to 1.
                angle: The direction of the gradient in degree. Relative to image coordinate system (clockwise).
                response: Result of the Harris corner formula R = det(M) - k*trace(M)**2
    :rtype: List[cv2.KeyPoint]

    """
    
    ######################################################
    # Calculate the kernel width based on sigma1 and sigma2
    kernel_width1 = int(2 * ceil(3 * sigma1) + 1)
    kernel_width2 = int(2 * ceil(3 * sigma2) + 1)

    # Create Gaussian kernels
    gauss1 = cv2.getGaussianKernel(kernel_width1, sigma1)
    gauss2 = cv2.getGaussianKernel(kernel_width2, sigma2)
    gauss1 = np.outer(gauss1, gauss1.transpose())
    gauss2 = np.outer(gauss2, gauss2.transpose())

    # Smooth the image with the first Gaussian
    img_smoothed = cv2.filter2D(img, -1, gauss1)

    # Compute image derivatives
    Iy, Ix = np.gradient(img_smoothed)

    # Compute products of derivatives
    Ixx = Ix ** 2
    Ixy = Ix * Iy
    Iyy = Iy ** 2

    # Convolution with the second Gaussian
    Sxx = cv2.filter2D(Ixx, -1, gauss2)
    Sxy = cv2.filter2D(Ixy, -1, gauss2)
    Syy = cv2.filter2D(Iyy, -1, gauss2)

    # Compute the Harris response for each pixel
    detM = Sxx * Syy - Sxy ** 2
    traceM = Sxx + Syy
    R = detM - k * traceM ** 2

    # Normalize the Harris response
    R /= np.max(R)

    # Thresholding
    R[R < threshold] = 0

    # Apply non-maximum suppression
    corners = non_max(R)

    # Extract coordinates of corners
    keypoint_coordinates = np.argwhere(corners)

    # Convert to OpenCV KeyPoint format
    keypoints_cv = []

    for i, j in keypoint_coordinates:
        keypoints = cv2.KeyPoint(x=float(j), y=float(i), size=1, angle=0, response=R[i, j], octave=0, class_id=-1)
        keypoints_cv.append(keypoints)

    ######################################################
    return keypoints_cv

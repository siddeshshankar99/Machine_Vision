#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Find the projective transformation matrix (homography) between from a source image to a target image.

Author: Siddesh Bramarambika Shankar
MatrNr: 12329513
"""

import numpy as np
from helper_functions import *
import math


def find_homography_ransac(source_points: np.ndarray,
                           target_points: np.ndarray,
                           confidence: float,
                           inlier_threshold: float) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return estimated transforamtion matrix of source_points in the target image given matching points

    Return the projective transformation matrix for homogeneous coordinates. It uses the RANSAC algorithm with the
    Least-Squares algorithm to minimize the back-projection error and be robust against outliers.
    Requires at least 4 matching points.

    :param source_points: Array of points. Each row holds one point from the source (object) image [x, y]
    :type source_points: np.ndarray with shape (n, 2)

    :param target_points: Array of points. Each row holds one point from the target (scene) image [x, y].
    :type target_points: np.ndarray with shape (n, 2)

    :param confidence: Solution Confidence (in percent): Likelihood of all sampled points being inliers.
    :type confidence: float

    :param inlier_threshold: Max. Euclidean distance of a point from the transformed point to be considered an inlier
    :type inlier_threshold: float

    :return: (homography, inliers, num_iterations)
        homography: The projective transformation matrix for homogeneous coordinates with shape (3, 3)
        inliers: Is True if the point at the index is an inlier. Boolean array with shape (n,)
        num_iterations: The number of iterations that were needed for the sample consensus
    :rtype: Tuple[np.ndarray, np.ndarray, int]
    """
    ######################################################
    if source_points.shape[0] < 4 or target_points.shape[0] < 4 or source_points.shape[0] != target_points.shape[0]:
        print("Insufficient point correspondences. Skipping homography calculation.")
        return None

    num_points = source_points.shape[0]
    best_homography = np.eye(3)
    best_inliers = np.full(shape=len(target_points), fill_value=False, dtype=bool)
    best_error = np.inf
    num_iterations = 0
    best_inlier_matches = None

    # Convert points to homogeneous coordinates
    source_points_hom = np.concatenate([source_points, np.ones((num_points, 1))], axis=1)

    num_pixels = len(source_points)
    m = 4  # Size of sample
    epsilon = m/num_pixels
    max_trials = math.log(1 - confidence)/math.log(1 - epsilon**2)  # From formula proposed in Fisher, Bolles 1981
    current_trial = 0

    # RANSAC loop
    while current_trial < max_trials:

            # Randomly sample 4 points
            indices = np.random.choice(num_points, 4, replace=False)
            sampled_source_points = source_points[indices]
            sampled_target_points = target_points[indices]

            # Estimate homography using the sampled points
            sampled_homography = find_homography_leastsquares(sampled_source_points, sampled_target_points)

            # Apply the estimated homography to all source points
            transformed_points = (sampled_homography @ source_points_hom.T).T
            transformed_points /= transformed_points[:, -1][:, np.newaxis]  # Normalize to (x, y, 1)

            # Calculate distances to target points
            distances = np.sqrt(np.sum((transformed_points[:, :2] - target_points) ** 2, axis=1))

            # Determine inliers
            inliers = distances < inlier_threshold
            num_inliers = np.sum(inliers)
            error = np.sum(distances[inliers])

            # Update the best homography if the current one is better
            if num_inliers > np.sum(best_inliers):
                best_homography = sampled_homography
                best_inliers = inliers
                # Update epsilon based on new inliers
                epsilon = np.sum(best_inliers) / num_points
                # Recalculate max_trials based on updated epsilon
                max_trials = math.log(1 - confidence) / math.log(1 - epsilon ** 3)

            current_trial += 1

    # Refine the homography matrix using all inliers
    if best_homography is not None:
        all_inliers_source = source_points[best_inliers]
        all_inliers_target = target_points[best_inliers]

        # Recompute the homography with all inliers
        refined_homography = find_homography_leastsquares(all_inliers_source, all_inliers_target)

        # Apply the refined homography to all source points
        transformed_points = (refined_homography @ source_points_hom.T).T
        transformed_points /= transformed_points[:, -1][:, np.newaxis]  # Normalize to (x, y, 1)

        # Calculate distances to target points
        distances = np.sqrt(np.sum((transformed_points[:, :2] - target_points) ** 2, axis=1))

        # Determine inliers
        inliers = distances < inlier_threshold
        num_inliers = np.sum(inliers)

        # Update the best homography, inliers and error
        if num_inliers > np.sum(best_inliers):
            best_homography = refined_homography
            best_inliers = inliers

        # Extract inlier matches based on the final homography
        best_inlier_matches = (
        source_points[best_inliers], target_points[best_inliers]) if best_homography is not None else None

    ######################################################
    return best_inlier_matches

def find_homography_leastsquares(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    """Return projective transformation matrix of source_points in the target image given matching points

    Return the projective transformation matrix for homogeneous coordinates. It uses the Least-Squares algorithm to
    minimize the back-projection error with all points provided. Requires at least 4 matching points.

    :param source_points: Array of points. Each row holds one point from the source image (object image) as [x, y]
    :type source_points: np.ndarray with shape (n, 2)

    :param target_points: Array of points. Each row holds one point from the target image (scene image) as [x, y].
    :type target_points: np.ndarray with shape (n, 2)

    :return: The projective transformation matrix for homogeneous coordinates with shape (3, 3)
    :rtype: np.ndarray with shape (3, 3)
    """
    ######################################################
    if source_points.shape[0] < 4 or target_points.shape[0] < 4 or source_points.shape[0] != target_points.shape[0]:
        print("Insufficient point correspondences. Skipping homography calculation.")
        return None

    n_points = source_points.shape[0]

    A = np.zeros((n_points * 2, 8))
    b = np.zeros((n_points * 2, 1))

    for i in range(n_points):
        x, y = source_points[i, 0], source_points[i, 1]
        xp, yp = target_points[i, 0], target_points[i, 1]

        A[2 * i] = [x, y, 1, 0, 0, 0, -x * xp, -y * xp]
        A[2 * i + 1] = [0, 0, 0, x, y, 1, -x * yp, -y * yp]

        b[2 * i] = xp
        b[2 * i + 1] = yp

    # Solve the least squares problem
    h, r, s, t = np.linalg.lstsq(A, b, rcond=None)

    # Reshape h into a 3x3 matrix
    homography = np.reshape(np.append(h, 1), (3, 3))

    ######################################################
    return homography

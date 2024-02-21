#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Fit Plane in pointcloud

Author: Siddesh Bramarambika Shankar
MatrNr: 12329513
"""

from typing import Tuple

import copy

import numpy as np
import open3d as o3d
import math


def fit_plane(pcd: o3d.geometry.PointCloud,
              confidence: float,
              inlier_threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """ Find dominant plane in pointcloud with sample consensus.

    Detect a plane in the input pointcloud using sample consensus. The number of iterations is chosen adaptively.
    The concrete error function is given as an parameter.

    :param pcd: The (down-sampled) pointcloud in which to detect the dominant plane
    :type pcd: o3d.geometry.PointCloud (http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html)

    :param confidence: Solution Confidence (in percent): Likelihood of all sampled points being inliers.
    :type confidence: float

    :param inlier_threshold: Max. distance of a point from the plane to be considered an inlier (in meters)
    :type inlier_threshold: float

    :return: (best_plane, best_inliers)
        best_plane: Array with the coefficients of the plane equation ax+by+cz+d=0 (shape = (4,))
        best_inliers: Boolean array with the same length as pcd.points. Is True if the point at the index is an inlier
    :rtype: tuple (np.ndarray[a,b,c,d], np.ndarray)
    """
    ######################################################
    if len(np.asarray(pcd.points)) < 3:
        raise ValueError("There must be at least 3 points.")

    points = np.asarray(pcd.points)
    num_points = points.shape[0]
    best_plane = None
    best_inliers = np.zeros(num_points, dtype=bool)

    # Initial epsilon and number of trials
    epsilon = 3 / num_points
    max_trials = math.log(1 - confidence) / math.log(1 - epsilon ** 3)
    current_trial = 0

    # Vectorize sampling if possible
    all_indices = np.arange(num_points)

    while current_trial < max_trials:
        # Randomly sample 3 points
        indices = np.random.choice(all_indices, 3, replace=False)
        sample_points = points[indices]

        # Use the sampled points to fit a plane using least squares
        plane = find_fit_leastsquares(sample_points)

        # Classify inliers
        inliers = classify_inliers(points, plane, inlier_threshold)

        # Count inliers
        inlier_count = np.sum(inliers)

        # Update the best plane if this one is better
        if inlier_count > np.sum(best_inliers):
            best_inliers = inliers
            best_plane = plane
            # Update epsilon based on new inliers
            epsilon = np.sum(best_inliers) / num_points
            # Recalculate max_trials based on updated epsilon
            max_trials = math.log(1 - confidence) / math.log(1 - epsilon ** 3)

        current_trial += 1

    if best_plane is not None:
        all_inliers = points[best_inliers]
        best_plane = find_fit_leastsquares(all_inliers)

        # Ensure that the normal vector has a positive z-component
        if best_plane[2] < 0:
            best_plane = -best_plane

        return best_plane, best_inliers
    else:
        raise Exception("Failed to detect a plane.")


def find_fit_leastsquares(points: np.ndarray) -> np.ndarray:
    # Check if there are at least 3 points
    if len(np.asarray(points)) < 3:
        raise ValueError("There must be at least 3 points.")

    # Extract points
    points = np.asarray(points)

    # Construct the A matrix with point coordinates and a column of ones for the constant term
    A = np.hstack((points, np.ones((points.shape[0], 1))))

    # Construct the b vector (zero vector in this case since we are assuming d = -1)
    b = -np.ones((points.shape[0], 1))

    # Solve the least squares problem
    plane_parameters, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    # Return the plane parameters
    return plane_parameters.flatten()

def classify_inliers(points: np.ndarray, plane: np.ndarray, threshold: float) -> np.ndarray:
    # Vectorized distance calculation
    normal = plane[:3]
    distances = np.abs(np.dot(points, normal) + plane[3]) / np.linalg.norm(normal)
    return distances < threshold
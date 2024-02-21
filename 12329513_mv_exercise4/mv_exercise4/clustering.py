#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Find clusters of pointcloud

Author: Siddesh Bramarambika Shankar
MatrNr: 12329513
"""

from typing import Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import distance
from scipy.stats import anderson
import matplotlib.pyplot as plt
from helper_functions import plot_clustering_results, silhouette_score


def kmeans(points: np.ndarray,
           n_clusters: int,
           n_iterations: int,
           max_singlerun_iterations: int,
           centers_in: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """ Find clusters in the provided data coming from a pointcloud using the k-means algorithm.

    :param points: The (down-sampled) points of the pointcloud to be clustered.
    :type points: np.ndarray with shape=(n_points, 3)

    :param n_clusters: The number of clusters to form as well as the number of centroids to generate.
    :type n_clusters: int

    :param n_iterations: Number of time the k-means algorithm will be run with different centroid seeds.
        The final results will be the best output of n_iterations consecutive runs in terms of inertia.
    :type n_iterations: int

    :param max_singlerun_iterations: Maximum number of iterations of the k-means algorithm for a single run.
    :type max_singlerun_iterations: int

    :param centers_in: Start centers of the k-means algorithm.  If centers_in = None, the centers are randomly sampled
        from input data for each iteration.
    :type centers_in: np.ndarray with shape = (n_clusters, 3) or None

    :return: (best_centers, best_labels)
        best_centers: Array with the centers of the calculated clusters (shape = (n_clusters, 3) and dtype=np.float32)
        best_labels: Array with a different label for each cluster for each point (shape = (n_points),) and dtype=int)
            The label i corresponds with the center in best_centers[i] and therefore are in the range [0, n_clusters-1]
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    ######################################################
    best_inertia = np.inf
    best_centers = None
    best_labels = None

    for iteration in range(n_iterations):
        # Initialize centers and labels
        centers = np.zeros(shape=(n_clusters, 3), dtype=np.float32)
        labels = np.zeros(shape=(len(points),), dtype=int)

        # Forgy initialization method: choose random samples as initial centers
        if centers_in is None:
            centers = points[np.random.choice(points.shape[0], n_clusters, replace=False)]
        else:
            centers = centers_in

        previous_labels = None

        for i in range(max_singlerun_iterations):
            # Compute distances from points to centers
            distances = distance.cdist(points, centers)

            # Assign labels based on closest center
            labels = np.argmin(distances, axis=1)

            # If labels haven't changed, we've reached convergence
            if np.array_equal(labels, previous_labels):
                break
            previous_labels = labels

            # Compute new centers as mean of points assigned to each cluster
            for j in range(n_clusters):
                if np.any(labels == j):
                    centers[j] = points[labels == j].mean(axis=0)
                else:
                    # If a cluster loses all its points, reinitialize its center
                    centers[j] = points[np.random.choice(points.shape[0], 1, replace=False)]

        # Compute inertia for this iteration
        inertia = np.sum([np.min(distances[np.arange(len(points)), labels]) ** 2])

        # Update best_centers and best_labels if this iteration's inertia is the best so far
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers
            best_labels = labels

        best_centers = best_centers.astype(np.float32)
        best_labels =  best_labels.astype(int)

    return best_centers, best_labels

def iterative_kmeans(points: np.ndarray,
                     max_n_clusters: int,
                     n_iterations: int,
                     max_singlerun_iterations: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Applies the k-means algorithm multiple times and returns the best result in terms of silhouette score.

    This algorithm runs the k-means algorithm for all number of clusters until max_n_clusters. The silhouette score is
    calculated for each solution. The clusters with the highest silhouette score are returned.

    :param points: The (down-sampled) points of the pointcloud that should be clustered
    :type points: np.ndarray with shape=(n_points, 3)

    :param max_n_clusters: The maximum number of clusters that is tested.
    :type max_n_clusters: int

    :param n_iterations: Number of time each k-means algorithm will be run with different centroid seeds.
        The final results will be the best output of n_iterations consecutive runs in terms of inertia.
    :type n_iterations: int

    :param max_singlerun_iterations: Maximum number of iterations of each k-means algorithm for a single run.
    :type max_singlerun_iterations: int

    :return: (best_centers, best_labels)
        best_centers: Array with the centers of the calculated clusters (shape = (n_clusters, 3) and dtype=np.float32)
        best_labels: Array with a different label for each cluster for each point (shape = (n_points),) and dtype=int)
            The label i corresponds with the center in best_centers[i] and therefore are in the range [0, n_clusters-1]
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    ######################################################
    best_score = -1
    best_centers = None
    best_labels = None

    # Loop over the range of cluster numbers
    for n_clusters in range(2, max_n_clusters + 1):
        for i in range(n_iterations):
            # Run k-means for this number of clusters
            centers, labels = kmeans(points, n_clusters, 1, max_singlerun_iterations)

            # Check if any cluster has one or zero points
            if any(np.sum(labels == i) <= 1 for i in range(n_clusters)) :
            # Skip silhouette score calculation for this iteration
                 continue

            # Calculate silhouette score
            score = silhouette_score(points, centers, labels)

            # Update the best score, centers, and labels if this is the best score so far
            if score > best_score:
                best_score = score
                best_centers = centers
                best_labels = labels

    return best_centers, best_labels


def gmeans(points: np.ndarray,
           tolerance: float,
           max_singlerun_iterations: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Find clusters in the provided data coming from a pointcloud using the g-means algorithm.

    The algorithm was proposed by Hamerly, Greg, and Charles Elkan. "Learning the k in k-means." Advances in neural
    information processing systems 16 (2003).

    :param points: The (down-sampled) points of the pointcloud to be clustered
    :type points: np.ndarray with shape=(n_points, 3)

    :param max_singlerun_iterations: Maximum number of iterations of the k-means algorithm for a single run.
    :type max_singlerun_iterations: int

    :return: (best_centers, best_labels)
        best_centers: Array with the centers of the calculated clusters (shape = (n_clusters, 3) and dtype=np.float32)
        best_labels: Array with a different label for each cluster for each point (shape = (n_points,) and dtype=int)
            The label i corresponds with the center in best_centers[i] and therefore are in the range [0, n_clusters-1]
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    ######################################################
    # Initialize centers to an empty array, will grow dynamically
    centers = np.zeros((0, points.shape[1]), dtype=np.float32)
    labels = np.zeros(points.shape[0], dtype=int)

    # Start with one cluster, the center is the mean of all points
    centers = np.vstack([centers, points.mean(axis=0)])
    n_clusters = 1

    # The loop continues until no more splits are made
    while True:
        centers_updated = False

        # Run k-means for current centers and get labels for each point
        centers, labels = kmeans(points, n_clusters, 1, max_singlerun_iterations)

        # Array to hold the new set of centers after potential splits
        new_centers = np.zeros((0, points.shape[1]), dtype=np.float32)

        # Evaluate each cluster formed by k-means
        for i in range(n_clusters):
            cluster_points = points[labels == i]

            # Handle the case where there is only one point in the cluster
            if cluster_points.shape[0] <= 1:
                continue

            # covariance matrix and eigenvalue calculation
            cov_matrix = np.cov(cluster_points, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

            # Principal component is the eigenvector with the largest eigenvalue
            si = eigenvectors[:, -1]
            lambda_i = eigenvalues[-1]

            # Define new potential centers
            ci1 = centers[i] + (si * (np.sqrt(2 * lambda_i) / np.pi))
            ci2 = centers[i] - (si * (np.sqrt(2 * lambda_i) / np.pi))

            # Project all points onto the line between the new centers
            v = ci1 - ci2
            x_projected = np.dot(cluster_points - centers[i], v) / np.linalg.norm(v)**2

            # Perform the Anderson-Darling test for normality
            estimation, critical_values, _ = anderson(x_projected)

            # Check if the cluster passes the test for Gaussian distribution
            if estimation <= critical_values[-1] * tolerance:
                # If Gaussian, keep the original center
                new_centers = np.vstack([new_centers, centers[i]])
            else:
                # If not Gaussian, use the new centers instead
                new_centers = np.vstack([new_centers, ci1, ci2])
                centers_updated = True

        # Update centers and labels for the next iteration
        centers = new_centers
        n_clusters = len(centers)

        # If no centers were updated (no splits), then we're done
        if not centers_updated:
            break

    # Final run of k-means to refine the cluster assignments
    final_centers, final_labels = kmeans(points, n_clusters, 1, max_singlerun_iterations)

    return final_centers, final_labels

def dbscan(points: np.ndarray,
           eps: float = 0.05,
           min_samples: int = 10) -> np.ndarray:
    """ Find clusters in the provided data coming from a pointcloud using the DBSCAN algorithm.

    The algorithm was proposed in Ester, Martin, et al. "A density-based algorithm for discovering clusters in large
    spatial databases with noise." kdd. Vol. 96. No. 34. 1996.

    :param points: The (down-sampled) points of the pointcloud to be clustered
    :type points: np.ndarray with shape=(n_points, 3)

    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :type eps: float

    :param min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core
        point. This includes the point itself.
    :type min_samples: float

    :return: Labels array with a different label for each cluster for each point (shape = (n_points,) and dtype=int)
            The label -1 is assigned to points that are considered to be noise.
    :rtype: np.ndarray
    """
    ######################################################
    labels = np.zeros(len(points), dtype=int) - 1
    cluster_id = 0

    for i in range(len(points)):
        if labels[i] != -1:  # Skip if the point is already processed
            continue

        # Find neighbors
        neighbors = [j for j in range(len(points)) if np.linalg.norm(points[j] - points[i]) < eps]

        # Mark as noise and continue if not enough neighbors
        if len(neighbors) < min_samples:
            continue

        # Else, it's a core point
        labels[i] = cluster_id
        seeds = set(neighbors)
        seeds.remove(i)

        # Process every seed point
        while seeds:
            current_point = seeds.pop()

            # Only process if the point is marked as noise or unvisited
            if labels[current_point] == -1:
                new_neighbors = [j for j in range(len(points)) if
                                 np.linalg.norm(points[j] - points[current_point]) < eps]

                # If it's a core point, add its neighbors to the seeds
                if len(new_neighbors) >= min_samples:
                    labels[current_point] = cluster_id
                    seeds.update(new_neighbors)

                else:
                    labels[current_point] = cluster_id

        cluster_id += 1  # Increment for next cluster
    print(labels)

    return labels

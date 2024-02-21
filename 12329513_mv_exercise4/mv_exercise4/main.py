#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Machine Vision (376.081)
Exercise 4: Clustering
Matthias Hirschmanner 2023
Automation & Control Institute, TU Wien

Tutors: machinevision@acin.tuwien.ac.at

"""

from pathlib import Path

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from fit_plane import fit_plane
from clustering import *
from helper_functions import *
import time
import matplotlib.pyplot as plt



if __name__ == '__main__':

    # Selects which single-plane file to use
    pointcloud_idx = 0

    # Pick which clustering algorithm to apply:
    use_kmeans = False
    use_iterative_kmeans = False
    use_gmeans = False
    use_dbscan = True

    # RANSAC parameters:
    confidence = 0.85
    inlier_threshold = 0.015  # Might need to be adapted, depending on how you implement fit_plane

    # Downsampling parameters:
    use_voxel_downsampling = True
    voxel_size = 0.012599
    uniform_every_k_points = 10

    # Clustering Parameters
    kmeans_n_clusters = 6
    kmeans_iterations = 25
    max_singlerun_iterations = 100
    iterative_kmeans_max_clusters = 10
    gmeans_tolerance = 10
    dbscan_eps = 0.05
    dbscan_min_points = 10
    debug_output = True

    # Read Pointcloud
    current_path = Path(__file__).parent
    pcd = o3d.io.read_point_cloud(str(current_path.joinpath("pointclouds/image00")) + str(pointcloud_idx) + ".pcd",
                                  remove_nan_points=True, remove_infinite_points=True)
    if not pcd.has_points():
        raise FileNotFoundError("Couldn't load pointcloud in " + str(current_path))

    # Down-sample the loaded point cloud to reduce computation time
    if use_voxel_downsampling:
        pcd_sampled = pcd.voxel_down_sample(voxel_size=voxel_size)
    else:
        pcd_sampled = pcd.uniform_down_sample(uniform_every_k_points)

    # Apply your own plane-fitting algorithm
    plane_model, best_inliers = fit_plane(pcd=pcd_sampled,
                                          confidence=confidence,
                                          inlier_threshold=inlier_threshold)
    inlier_indices = np.nonzero(best_inliers)[0]

    # Alternatively use the built-in function of Open3D
    #plane_model, inlier_indices = pcd_sampled.segment_plane(distance_threshold=inlier_threshold,
    #                                                        ransac_n=3,
    #                                                        num_iterations=500)

    # Convert the inlier indices to a Boolean mask for the pointcloud
    #best_inliers = np.full(shape=len(pcd_sampled.points, ), fill_value=False, dtype=bool)
    #best_inliers[inlier_indices] = True

    # Store points without plane in scene_pcd
    scene_pcd = pcd_sampled.select_by_index(inlier_indices, invert=True)

    # Plot detected plane and remaining pointcloud
    if debug_output:
        plot_dominant_plane(pcd_sampled, best_inliers, plane_model)
        o3d.visualization.draw_geometries([scene_pcd])

    # Convert to NumPy array
    points = np.asarray(scene_pcd.points, dtype=np.float32)

    # Initialize runtimes dictionary
    runtimes = {
        'K-means': 0,
        'Iterative K-means': 0,
        'G-means': 0,
        'DBSCAN': 0
    }

    # k-Means
    if use_kmeans:
        start_time = time.time()
        # Apply k-means algorithm
        centers, labels = kmeans(points,
                                 n_clusters=kmeans_n_clusters,
                                 n_iterations=kmeans_iterations,
                                 max_singlerun_iterations=max_singlerun_iterations)
        runtimes['K-means'] = time.time() - start_time
        plot_clustering_results(scene_pcd,
                                labels,
                                "K-means",
                                cmap="tab20")

    # Iterative k-Means
    if use_iterative_kmeans:
        start_time = time.time()
        centers, labels = iterative_kmeans(points,
                                           max_n_clusters=iterative_kmeans_max_clusters,
                                           n_iterations=kmeans_iterations,
                                           max_singlerun_iterations=max_singlerun_iterations)
        runtimes['Iterative K-means'] = time.time() - start_time
        plot_clustering_results(scene_pcd,
                                labels,
                                "Iterative k-means",
                                cmap="tab20")

    # G-Means
    if use_gmeans:
        start_time = time.time()
        centers, labels = gmeans(points,
                                 tolerance=gmeans_tolerance,
                                 max_singlerun_iterations=max_singlerun_iterations)
        runtimes['G-means'] = time.time() - start_time
        plot_clustering_results(scene_pcd,
                                labels,
                                "G-means",
                                cmap="tab20")
    # DBSCAN
    if use_dbscan:
        start_time = time.time()
        labels = dbscan(points,
                        eps=dbscan_eps,
                        min_samples=dbscan_min_points)
        runtimes['DBSCAN'] = time.time() - start_time
        plot_clustering_results(scene_pcd,
                                labels,
                                "DBSCAN",
                                cmap="tab20")

    # Plotting the runtimes
    algorithms = list(runtimes.keys())
    times = [runtimes[algo] for algo in algorithms]

    plt.figure(figsize=(10, 6))
    plt.bar(algorithms, times, color='skyblue')
    plt.xlabel('Clustering Algorithm')
    plt.ylabel('Runtime (s)')
    plt.title('Runtime Comparison of Clustering Algorithms')
    plt.show()
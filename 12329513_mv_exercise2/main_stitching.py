#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Machine Vision (376.081)
Exercise 2: Image Stitching using Harris
Matthias Hirschmanner 2023
Automation & Control Institute, TU Wien

Tutors: machinevision@acin.tuwien.ac.at
"""

from pathlib import Path
import cv2
import numpy as np

from harris_corner import harris_corner
from descriptors import compute_descriptors
from find_homography import find_homography_ransac
from helper_functions import *

if __name__ == '__main__':

    save_image = False  # Enables saving of matches image
    use_matplotlib = False  # Enables saving of matches image
    #img_paths = ['Table/Image-001.jpeg',
    #             'Table/Image-002.jpeg',
    #             'Table/Image-003.jpeg',
    #             'Table/Image-004.jpeg']
    img_paths = ['gusshaus/Image-02.jpg',
                 'gusshaus/Image-00.jpg',
                 'gusshaus/Image-01.jpg',
                 'gusshaus/Image-03.jpg',
                 'gusshaus/Image-04.jpg']
    #img_paths = ['desk/Image-00.jpg',
    #             'desk/Image-01.jpg',
    #             'desk/Image-02.jpg',
    #             'desk/Image-03.jpg']

    # parameters <<< try different settings!
    sigma1 = 0.8
    sigma2 = 1.5
    threshold = 0.01
    k = 0.04
    patch_size = 5
    ransac_confidence = 0.8
    ransac_inlier_threshold = 10.

    # Load the images
    current_path = Path(__file__).parent

    imgs_color = []
    imgs_gray = []
    keypoints_list = []
    descriptors_list = []
    for filename in img_paths:
        img_color = cv2.imread(str(current_path.joinpath(filename)), cv2.IMREAD_COLOR)
        if img_color is None:
            raise FileNotFoundError("Couldn't load image " + str(current_path.joinpath(filename)))
        imgs_color.append(img_color)
        imgs_gray.append(cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY))

    for img_gray in imgs_gray:
        # Harris corner detector
        keypoints_tmp = harris_corner(img_gray.astype(np.float32) / 255.,
                                      sigma1=sigma1,
                                      sigma2=sigma2,
                                      k=k,
                                      threshold=threshold)
        filtered_keypoints_tmp, descriptors_tmp = compute_descriptors(img_gray.astype(np.float32) / 255.,
                                                                      keypoints_tmp,
                                                                      patch_size)
        keypoints_list.append(filtered_keypoints_tmp)
        descriptors_list.append(descriptors_tmp.copy())

    # FLANN (Fast Library for Approximate Nearest Neighbors) parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Create an image which is bigger than the individual images, so we can put the stitched images there
    # We translate the first image a bit towards the middle, so we have enough space to the sides
    start_row = imgs_gray[0].shape[0] // 2
    start_col = imgs_gray[0].shape[1]
    translation = np.eye(3, dtype=np.float32)
    translation[0, 2] = start_col
    translation[1, 2] = start_row

    # Create a transformation matrix that we can later multiply with the homographies
    transformation_matrix = translation.copy()
    img_blend = np.zeros((imgs_gray[0].shape[0] * 2, imgs_gray[0].shape[1] * 4, 3), dtype=np.uint8)
    img_blend[start_row:start_row + imgs_color[0].shape[0],
    start_col:start_col + imgs_color[1].shape[1], :] = imgs_color[0]
    for i in range(1, len(descriptors_list)):
        show_image(img_blend, "Stitched Image", use_matplotlib=use_matplotlib, save_image=False)
        # For each keypoint in the image_n get the 2 best matches in image_n-1
        matches_stitching_tmp = flann.knnMatch(descriptors_list[i].astype(np.float32),
                                               descriptors_list[i - 1].astype(np.float32), k=2)
        filtered_matches_stitching_tmp = filter_matches(matches_stitching_tmp)

        source_points_tmp = np.array([keypoints_list[i][match.queryIdx].pt for match in filtered_matches_stitching_tmp])
        target_points_tmp = np.array(
            [keypoints_list[i - 1][match.trainIdx].pt for match in filtered_matches_stitching_tmp])
        homography_tmp, _, _ = find_homography_ransac(source_points_tmp,
                                                      target_points_tmp,
                                                      confidence=ransac_confidence,
                                                      inlier_threshold=ransac_inlier_threshold)

        # Multiply the found homography with the previous transformation matrix and apply perspective Warping
        transformation_matrix = transformation_matrix @ homography_tmp
        img_transformed = cv2.warpPerspective(imgs_color[i],
                                              transformation_matrix,
                                              dsize=(img_blend.shape[1],
                                                     img_blend.shape[0]))
        img_blend[np.all(img_transformed != 0.0, axis=2)] = img_transformed[np.all(img_transformed != 0.0, axis=2)]

    show_image(img_blend, "Stitched Image", use_matplotlib=use_matplotlib, save_image=save_image)

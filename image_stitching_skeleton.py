import cv2
import sys
import numpy as np
import math
import matplotlib.pyplot as plt

def ex_find_homography_ransac(list_pairs_matched_keypoints, threshold_ratio_inliers=0.85, threshold_reprojtion_error=3, max_num_trial=1000):
    '''
    Apply RANSAC algorithm to find a homography transformation matrix that align 2 sets of feature points, transform the first set of feature point to the second (e.g. warp image 1 to image 2)
    :param list_pairs_matched_keypoints: has the format as a list of pairs of matched points: [[[p1x,p1y],[p2x,p2y]],....]
    :param threshold_ratio_inliers: threshold on the ratio of inliers over the total number of samples, accept the estimated homography if ratio is higher than the threshold
    :param threshold_reprojtion_error: threshold of reprojection error (measured as euclidean distance, in pixels) to determine whether a sample is inlier or outlier
    :param max_num_trial: the maximum number of trials to do take sample and do testing to find the best homography matrix
    :return best_H: the best found homography matrix
    '''
    best_H = None

    # to be completed ...

    return best_H

def ex_extract_and_match_feature(img_1, img_2, ratio_robustness=0.7):
    '''
    1/ extract SIFT feature from image 1 and image 2,
    2/ use a bruteforce search to find pairs of matched features: for each feature point in img_1, find its best matched feature point in img_2
    3/ apply ratio test to select the set of robust matched points
    :param img_1: input image 1
    :param img_2: input image 2
    :param ratio_robustness: ratio for the robustness test
    :return list_pairs_matched_keypoints: has the format as list of pairs of matched points: [[[p1x,p1y],[p2x,p2y]]]
    '''
    # ==============================
    # ===== 1/ extract features from input image 1 and image 2
    # ==============================
    def write_out(im1, keypoints1, im2, keypoints2):
        img_1_output = cv2.drawKeypoints(cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY), keypoints1, im1)
        img_2_output = cv2.drawKeypoints(cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY), keypoints2, im2)
        cv2.imwrite('img_1_keypoints.jpg', img_1_output)
        cv2.imwrite('img_2_keypoints.jpg', img_2_output)

    img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    img_2_gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    img_1_keypoints, img_1_desc = sift.detectAndCompute(img_1_gray, None)
    img_2_keypoints, img_2_desc = sift.detectAndCompute(img_2_gray, None)
    write_out(img_1, img_1_keypoints, img_2, img_2_keypoints)

    # ==============================
    # ===== 2/ use bruteforce search to find a list of pairs of matched feature points
    # ==============================
    list_pairs_matched_keypoints = []
    allpoints = []

    for outerindex, (outerpoint, outerdesc) in enumerate(zip(img_1_keypoints, img_1_desc)):
        diffmatrix = outerdesc - img_2_desc
        distances = np.linalg.norm(diffmatrix, axis=1)
        kpwithdistance = np.column_stack((img_2_keypoints, distances, range(0, len(img_2_keypoints))))
        sortwdist = kpwithdistance[kpwithdistance[:,1].argsort()]
        toptwo = sortwdist[:2]
        if len(toptwo) > 1:
            # firstpoint = toptwo[0][0]
            firstdist = toptwo[0][1]
            seconddist = toptwo[1][1]
            if firstdist < seconddist * ratio_robustness:
                allpoints.append([[outerpoint, outerindex], toptwo[0]])

    # Debugging with cv2
    '''
    bestpairs = []
    for tuple in allpoints:
        queryldx = tuple[0][1]
        trainldx = tuple[1][2]
        matchlist = tuple[1]
        tupdistance = tuple[1][1]
        source = cv2.DMatch(queryldx, trainldx, tupdistance)
        best = [tuple[0], tuple[1]]
        bestpairs.append([source])
        # print("test")
    bf = cv2.BFMatcher()
    opencvpairs = bf.knnMatch(img_1_desc, img_2_desc, k=2)
    bestpairscv = []
    for m, n in opencvpairs:
        if m.distance < ratio_robustness*n.distance:
            bestpairscv.append([m])
    img_match_lines = cv2.drawMatchesKnn(img_1, img_1_keypoints, img_2, img_2_keypoints, bestpairs, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img_match_lines),plt.show()
    '''

    for item in allpoints:
        point1 = item[0][0].pt
        point2 = item[1][0].pt
        list_pairs_matched_keypoints.append([[point1[0], point1[1]],
                                            [point2[0], point2[1]]])
    return list_pairs_matched_keypoints

def ex_warp_blend_crop_image(img_1,H_1,img_2):
    '''
    1/ warp image img_1 using the homography H_1 to align it with image img_2 (using backward warping and bilinear resampling)
    2/ stitch image img_1 to image img_2 and apply average blending to blend the 2 images into a single panorama image
    3/ find the best bounding box for the resulting stitched image
    :param img_1:
    :param H_1:
    :param img_2:
    :return img_panorama: resulting panorama image
    '''
    img_panorama = None
    # =====  use a backward warping algorithm to warp the source
    # 1/ to do so, we first create the inverse transform; 2/ use bilinear interpolation for resampling
    # to be completed ...

    # ===== blend images: average blending
    # to be completed ...

    # ===== find the best bounding box for the resulting stitched image so that it will contain all pixels from 2 original images
    # to be completed ...

    return img_panorama

def stitch_images(img_1, img_2):
    '''
    :param img_1: input image 1. We warp this image to align and stich it to the image 2
    :param img_2: is the reference image. We will not warp this image
    :return img_panorama: the resulting stiched image
    '''
    print('==============================')
    print('===== stitch two images to generate one panorama image')
    print('==============================')

    # ===== extract and match features from image 1 and image 2
    list_pairs_matched_keypoints = ex_extract_and_match_feature(img_1=img_1, img_2=img_2, ratio_robustness=0.7)

    # ===== use RANSAC algorithm to find homography to warp image 1 to align it to image 2
    H_1 = ex_find_homography_ransac(list_pairs_matched_keypoints, threshold_ratio_inliers=0.85, threshold_reprojtion_error=3, max_num_trial=1000)

    # ===== warp image 1, blend it with image 2 using average blending to produce the resulting panorama image
    img_panorama = ex_warp_blend_crop_image(img_1=img_1,H_1=H_1, img_2=img_2)


    return img_panorama

if __name__ == "__main__":
    print('==================================================')
    print('PSU CS 410/510, Winter 2020, HW2: image stitching')
    print('==================================================')

    path_file_image_1 = sys.argv[1]
    path_file_image_2 = sys.argv[2]
    path_file_image_result = sys.argv[3]


    # ===== read 2 input images

    img_1 = cv2.imread(path_file_image_1)
    img_2 = cv2.imread(path_file_image_2)

    # ===== create a panorama image by stitch image 1 to image 2
    img_panorama = stitch_images(img_1=img_1, img_2=img_2)

    # ===== save panorama image
    #cv2.imwrite(filename=path_file_image_result, img=(img_panorama).clip(0.0, 255.0).astype(np.uint8))


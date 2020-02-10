import cv2
import sys
import numpy as np
import math
import matplotlib.pyplot as plt


def homog(point):
    appended = point.copy()
    appended.append(1)
    npvec = np.array(appended)
    return npvec

def linear(homogpoint):
    if homogpoint[2] == 0:
        homogpoint[2] = 0.0000001
    x = homogpoint[0] / homogpoint[2]
    y = homogpoint[1] / homogpoint[2]
    return np.array([x, y])

def ex_find_homography_ransac(list_pairs_matched_keypoints, threshold_ratio_inliers=0.85, threshold_reprojtion_error=3, max_num_trial=1000):
    '''
    Apply RANSAC algorithm to find a homography transformation matrix that align 2 sets of feature points, transform the first set of feature point to the second (e.g. warp image 1 to image 2)
    :param list_pairs_matched_keypoints: has the format as a list of pairs of matched points: [[[p1x,p1y],[p2x,p2y]],....]
    :param threshold_ratio_inliers: threshold on the ratio of inliers over the total number of samples, accept the estimated homography if ratio is higher than the threshold
    :param threshold_reprojtion_error: threshold of reprojection error (measured as euclidean distance, in pixels) to determine whether a sample is inlier or outlier
    :param max_num_trial: the maximum number of trials to do take sample and do testing to find the best homography matrix
    :return best_H: the best found homography matrix
    '''
    def find4homo(pointslist):
        '''
        point1 = pointslist[0]
        point2 = pointslist[1]
        point3 = pointslist[2]
        point4 = pointslist[3]
        x1 = point1[0][0]
        y1 = point1[0][1]
        xp1 = point1[1][0]
        yp1 = point1[1][1]
        x2 = point2[0][0]
        y2 = point2[0][1]
        xp2 = point2[1][0]
        yp2 = point2[1][1]
        x3 = point3[0][0]
        y3 = point3[0][1]
        xp3 = point3[1][0]
        yp3 = point3[1][1]
        x4 = point4[0][0]
        y4 = point4[0][1]
        xp4 = point4[1][0]
        yp4 = point4[1][1]
        Aarray = np.array([
                        [-x1,   -y1,    -1,     0,      0,      0,      x1*xp1,     y1*xp1,     xp1],
                        [0,     0,      0,      -x1,    -y1,    -1,     x1*yp1,     y1*yp1,     yp1],
                        [-x2,   -y2,    -1,     0,      0,      0,      x2*xp2,     y2*xp2,     xp2],
                        [0,     0,      0,      -x2,    -y2,    -1,     x2*yp2,     y2*yp2,     yp2],
                        [-x3,   -y3,    -1,     0,      0,      0,      x3*xp3,     y3*xp3,     xp3],
                        [0,     0,      0,      -x3,    -y3,    -1,     x3*yp3,     y3*yp3,     yp3],
                        [-x4,   -y4,    -1,     0,      0,      0,      x4*xp4,     y4*xp4,     xp4],
                        [ 0,    0,      0,      -x4,    -y4,    -1,     x4*yp4,     y4*yp4,     yp4]])
        U, S, testsvd = np.linalg.svd(Aarray, full_matrices=True)
        values = testsvd[:,8]
        values = np.reshape(values, (3,3))
        '''
        A = []
        for i in range(0, len(pointslist)):
            x1, y1 = pointslist[i][0]
            xp1, yp1 = pointslist[i][1]
            # A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
            # A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
            A.append([-x1, -y1, -1, 0, 0, 0, x1*xp1, y1*xp1, xp1])
            A.append([0, 0, 0, -x1, -y1, -1, x1*yp1, y1*yp1, yp1])
        A = np.asarray(A)
        U, S, Vh = np.linalg.svd(A)
        L = Vh[-1,:]
        H = L.reshape(3, 3)
        return H

    def findinliners(H, keypoints):
        inliernum = 0
        for (point1, point2) in keypoints:
            # Homog coords of point 1
            p1homog = homog(point1)
            # Homog coords of predicted point 2
            predicted = H @ p1homog
            # Standard coords of predicted point
            standardpredicted = linear(predicted)
            difference = point2 - standardpredicted
            error = np.linalg.norm(point2 - standardpredicted)
            if error < threshold_reprojtion_error:
                inliernum += 1
            # print("test")
        return inliernum

    H_array = []
    best_H = None
    for trial in range(0, max_num_trial):
        # Array coord pairs, 2 pair for each keypoint
        kps = np.array(list_pairs_matched_keypoints)
        # 4 random pairs from the list
        randomfour = kps[np.random.choice(kps.shape[0], 4)]
        # Random 4 arranged as nested list for function
        listfour = [randomfour[0].tolist(), randomfour[1].tolist(), randomfour[2].tolist(), randomfour[3].tolist()]
        # Calculate homography matrix from 4 pairs of points
        trialH = find4homo(randomfour.tolist())
        totalinliners = 0
        inliners = findinliners(trialH, list_pairs_matched_keypoints)
        if inliners / len(list_pairs_matched_keypoints) > threshold_ratio_inliers:
            H_array.append([trialH, inliners])
        totalinliners += inliners
    H_arr_np = np.array(H_array)
    H_arraysorted = H_arr_np[H_arr_np[:,1].argsort()]
    best_H = H_arraysorted[-1][0]
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


    #write_out(img_1, img_1_keypoints, img_2, img_2_keypoints)

    # ==============================
    # ===== 2/ use bruteforce search to find a list of pairs of matched feature points
    # ==============================
    list_pairs_matched_keypoints = []
    allpoints = []

    # For each keypoint and description in image 1...
    for outerindex, (outerpoint, outerdesc) in enumerate(zip(img_1_keypoints, img_1_desc)):
        diffmatrix = outerdesc - img_2_desc
        testdist = np.linalg.norm(outerdesc - img_2_desc[0])
        distances = np.linalg.norm(diffmatrix, axis=1)
        kpwithdistance = np.column_stack((img_2_keypoints, distances, range(0, len(img_2_keypoints))))
        # testsort = np.sort(kpwithdistance, axis=0)
        sortwdist = kpwithdistance[kpwithdistance[:,1].argsort()]
        toptwo = sortwdist[:2]
        if len(toptwo) > 1:
            # firstpoint = toptwo[0][0]
            firstdist = toptwo[0][1]
            seconddist = toptwo[1][1]
            if firstdist < seconddist * ratio_robustness:
                allpoints.append([[outerpoint, outerindex], toptwo[0]])



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

    # Debugging with cv2
    '''
    bf = cv2.BFMatcher()
    opencvpairs = bf.knnMatch(img_1_desc, img_2_desc, k=2)
    bestpairscv = []
    for m, n in opencvpairs:
        if m.distance < ratio_robustness*n.distance:
            bestpairscv.append([m])
    
    img_match_lines = cv2.drawMatchesKnn(img_1, img_1_keypoints, img_2, img_2_keypoints, bestpairs, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img_match_lines),plt.show()
    '''
    for keypoint1, keypoint2 in allpoints:
        point1 = keypoint1[0].pt
        point2 = keypoint2[0].pt
        list_pairs_matched_keypoints.append([[point1[0], point1[1]],[point2[0], point2[1]]])
    return list_pairs_matched_keypoints

def ex_warp_blend_crop_image(img_1, H_1, img_2):
    '''
    1/ warp image img_1 using the homography H_1 to align it with image img_2 (using backward warping and bilinear resampling)
    2/ stitch image img_1 to image img_2 and apply average blending to blend the 2 images into a single panorama image
    3/ find the best bounding box for the resulting stitched image
    :param img_1:
    :param H_1:
    :param img_2:
    :return img_panorama: resulting panorama image
    '''
    invH = np.linalg.inv(H_1)
    img_dims = img_1.shape
    img_panorama = np.zeros(shape=(3 * img_dims[0], 3 * img_dims[1], img_dims[2]), dtype=np.uint8)
    img_pano_height = img_panorama.shape[1]
    img_pano_width = img_panorama.shape[0]
    img_pano_1 = img_panorama.copy()
    img_pano_2 = img_panorama.copy()
    img_pano_2[img_dims[0]:img_dims[0]*2, img_dims[1]:img_dims[1]*2] = img_2
    warpedpixels = {}
    def one2fourbilin(pixel, rgb):
        i = math.floor(pixel[0])
        j = math.floor(pixel[1])
        dx = pixel[0] - i
        dy = pixel[1] - j
        ll = (dx * dy) * rgb
        lr = ((1 - dx)*dy) * rgb
        ur = ((1 - dx) * (1 - dy)) * rgb
        ul = (dx * (1 - dy)) * rgb
        locations = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]]
        locint = np.array(locations, dtype=int)
        pixvalues = np.array([ll, lr, ur, ul])
        result = np.column_stack((locint, pixvalues))
        return result

    # Compute pixel's color from it's 4 neighbors
    # via bilinear interpolation
    def four2onebilin(pixel, image):
        i = math.floor(pixel[0])
        j = math.floor(pixel[1])
        dx = pixel[0] - i
        dy = pixel[1] - j
        fll = image[i][j]
        flr = image[i + 1][j]
        fur = image[i + 1][j + 1]
        ful = image[i][j + 1]
        ll = ((1 - dx) * (1 - dy)) * fll
        lr = (dx *(1 - dy)) * flr
        ur = dx * fur
        ul = ((1 - dx) * dy) * ful
        totalcolor = ll + lr + ur + ul
        result = []
        return totalcolor

    img_1_width = img_1.shape[0]
    img_1_height = img_1.shape[1]
    # for xindex, row in enumerate(img_panorama):
    #    for yindex, pixel in enumerate(row):
    for xindex in range(-img_1_width, 2 * img_1_width):
        for yindex in range(-img_1_height, 2 * img_1_height):
            # pano coords in homogenous space
            pixhomo = homog([xindex, yindex])
            # Source coords in img_1 (homogenous)
            pixhomowarped = np.dot(invH, pixhomo)
            # scaling = np.array(([1, 0, 0], [0, 1, 0],[0, 0, 1]))
            # pixhomowarped1 = scaling @ pixhomo
            # Linear coords in img_1
            pixsrc = linear(pixhomowarped)
            # x coord in img_1
            srcx = pixsrc[0]
            # y coord in img_1
            srcy = pixsrc[1]
            #print("test")
            # If source coords are inside area of img_1:
            if 0 < srcx < (img_1_width - 1) and 0 < srcy < (img_1_height - 1):
                # Interpolate value from 4 nearest pixels
                colorvalue = four2onebilin([srcx, srcy], img_1)
                # Target pixel should be empty
                targetcoords = [xindex + img_1_width, yindex + img_1_height]
                targetpixel = img_panorama[xindex + img_1_width][yindex + img_1_height]
                # Alter color of current pixel in pano, + offset
                img_pano_1[xindex + img_1_height][yindex + img_1_width] += colorvalue.astype(np.uint8)
                #print("test")
            # else:
                # print("out of range")
    '''
    for xindex, row in enumerate(img_1):
        for yindex, pixel in enumerate(row):
            pixhomo = homog([xindex, yindex])
            newlochomo = invH @ pixhomo
            newloc = linear(newlochomo)
            bilinto4 = one2fourbilin(newloc, pixel)
            # locpixlist = [newloc, pixel]
            for line in bilinto4:
                loc = line[0:2].astype(int)
                loctup = (loc[0], loc[1])
                rgb = line[2:]
                if loctup not in warpedpixels:
                    warpedpixels[loctup] = [rgb]
                else:
                    warpedpixels[loctup].append(rgb)
            #print("test")
    warpedmean = {}
    for fracpixels in warpedpixels.items():
        unpacked = fracpixels[1]
        if len(unpacked) > 1:
            stacked = np.stack(unpacked, axis=1)
            stacked = stacked.mean(axis=1)
            warpedmean[fracpixels[0]] = stacked
        else:
            aslist = unpacked[0].tolist()
            warpedmean[fracpixels[0]] = np.array(aslist)
        #print("test")
    for coords, pixel in warpedmean.items():
        x = coords[0]
        y = coords[1]
        img_panorama[x,y] = pixel
        print("test")
    '''
    # Masking
    def mask(image):
        result = np.zeros(image.shape)
        for xindex, row in enumerate(image):
            for yindex, pixel in enumerate(row):
                if np.any(np.greater(pixel, 0)):
                    result[xindex][yindex] = [1, 1, 1]
        return result

    def crop(image):
        xvals = []
        yvals = []
        for xindex, row in enumerate(image):
            for yindex, pixel in enumerate(row):
                if np.any(np.greater(pixel, 0)):
                    xvals.append(xindex)
                    yvals.append(yindex)
        xvals.sort()
        yvals.sort()
        xmin, xmax = xvals[0], xvals[-1]
        ymin, ymax = yvals[0], yvals[-1]
        result = image[xmin:xmax, ymin:ymax]
        return result

    mask1 = mask(img_pano_1)
    mask2 = mask(img_pano_2)
    maskboth = mask1 + mask2
    lefthalf = np.divide(img_pano_1, maskboth)
    lefthalf = np.nan_to_num(lefthalf)
    righthalf = np.divide(img_pano_2, maskboth)
    righthalf = np.nan_to_num(righthalf)
    img_pano_masked = (lefthalf + righthalf).astype("uint8")
    #plt.imshow(cv2.cvtColor(lefthalf, cv2.COLOR_BGR2RGB)), plt.show()
    cropped = crop(img_pano_masked)
    plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)), plt.show()
    # =====  use a backward warping algorithm to warp the source
    # 1/ to do so, we first create the inverse transform; 2/ use bilinear interpolation for resampling
    # to be completed ...

    # ===== blend images: average blending
    # to be completed ...

    # ===== find the best bounding box for the resulting stitched image so that it will contain all pixels from 2 original images
    # to be completed ...

    return cropped

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
    # plt.imshow(cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)),plt.show()

    img_2 = cv2.imread(path_file_image_2)

    # ===== create a panorama image by stitch image 1 to image 2
    img_panorama = stitch_images(img_1=img_1, img_2=img_2)

    # ===== save panorama image
    cv2.imwrite(filename=path_file_image_result, img=(img_panorama).clip(0.0, 255.0).astype(np.uint8))


import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
from statistics import mean
from PIL import Image
import math


def get_rel_therm_fns(rgb_img, therm_path, depth_path):
    '''
    The function detects related files from different directories

    :param rgb_img: rgb target image
    :param therm_path: raw thermal aligned directory
    :param depth_path: depth matrix directory
    :return: img_name_therm, image_code, img_name_depth, img_name_therm_aligned
            # thermal image fn
            # image identifier
            # related depth path/filename
            # related aligned therm path/filename
    '''

    # image name string
    a = rgb_img
    len_ = len(a)

    # loop
    for i in range(0, len_):
        loc = a[len_ - i:len_]

        if loc.find("\\") != -1:  # if != "-1" the chr was found
            index = len_ - i
            stop_idx = a.find('_img_')
            # get image identifier
            image_code = str(a[index + 1:-4])
            # get thermal image fn
            img_name_therm = therm_path + '\\' + \
                             str(a[index + 1:stop_idx]) + \
                             '_thermal_' + \
                             str(a[stop_idx+5:-4]) + ".PNG"

            # get depth image fn
            img_name_depth = depth_path + '\\' + \
                             str(a[index + 1:stop_idx]) + \
                             '_depth_' + \
                             str(a[stop_idx+5:-4]) + ".PNG"

            # get thermal aligned image fn
            img_name_therm_aligned = therm_path +'\\'+ \
                                     str(a[index + 1:stop_idx]) + \
                                     '_thermal_' + \
                                     str(a[stop_idx+5:-4]) + "_aligned.PNG"

            return img_name_therm, image_code, img_name_depth, img_name_therm_aligned
        else:
            pass


def blob_detector_col(im_path):
    '''
    Color blob detector for RGB images 1280x720
    :param im_path: rgb image matrix
    :return: color detected blobs
    '''

    # Read and open the image + convert to grayscale
    img = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)

    # get image shape
    shape = img.shape

    if shape[1] == 1920:
        # if 1920x1080 convert to 1280x720
        img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)
    else:
        pass

    # https://learnopencv.com/blob-detection-using-opencv-python-c/

    # Set up the detector
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 150
    params.maxThreshold = 255
    params.thresholdStep = 20
    params.filterByArea = True
    params.minArea = 40
    params.maxArea = 260
    params.blobColor = 255
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.minCircularity = 0.7
    params.maxCircularity = 1
    params.filterByInertia = False
    params.minInertiaRatio = 0.5
    params.maxInertiaRatio = 1

    detector = cv2.SimpleBlobDetector_create(params)
    return detector.detect(img)


def blob_detector_therm(im_path):
    '''
    Thermal blob detector for RGB images 1280x720
    :param im_path: normalised thermal image 320x240 matrix
    :return: thermal detected blobs
    '''

    # Read image + convert to grayscale
    img = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)

    # https://learnopencv.com/blob-detection-using-opencv-python-c/

    # Set up the detector
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 30  # minimum level for thresholding
    params.maxThreshold = 255
    params.thresholdStep = 20  # value to increment thresholding value up to thresh max
    params.filterByArea = True
    params.minArea = 1
    params.maxArea = 60
    params.blobColor = 255  # blob for lighter pixels
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.minCircularity = 0.7  # square
    params.maxCircularity = 1  # circle
    params.filterByInertia = False
    params.minInertiaRatio = 0.5
    params.maxInertiaRatio = 1  # means a perfect circle

    detector = cv2.SimpleBlobDetector_create(params)
    return detector.detect(img)


def kpt2_coord(keypoints):
    '''
    The function converts simpleBlobdetector kpts into cartesian coordinates
    :param keypoints: simpleBlobdetector kpts
    :return: list of cartesian kpts coordinates
    '''

    point_list = []
    coord_list = []

    # convert from keypoints format to numeric format
    for i in keypoints:
        # i = list(i)
        # print(i.size)
        point_list.append((i.pt[0],
                        i.pt[1]))

    # convert to integers coordinates
    for pt in point_list:
        x = int(pt[0])
        y = int(pt[1])
        coord_list.append((x, y))

    return coord_list


def show_dynamic_keypoints(image, kpt_coord_list):
    '''
    The function plots kpts on the reference image
    :param image: image matrix
    :param kpt_coord_list: image related keypoints
    :return: None
    '''

    for pt in kpt_coord_list:
        cv2.circle(image, pt, radius=0,
                   color=(255, 0, 0),  # draw points in Blue (B,G,R)
                   thickness=10)
        cv2.imshow("Image", image)
        cv2.moveWindow("Image", 50, 30)
        cv2.waitKey(20)


def distance_matrix(kpt_coord_list):
    '''
    The function computes fot the cartesian keypoints the distance matrix
    :param kpt_coord_list: list of cartesian kpts coordinates
    :return: list of cleaned cartesian kpts coordinates
    '''

    from sklearn.metrics import DistanceMetric
    # get distance from points
    dist = DistanceMetric.get_metric('euclidean')

    # compute the euclidean distance matrix for each point in col_kpt
    distance_mtxs = dist.pairwise(kpt_coord_list)

    # convert numpy element into list to modify it
    distance_mtxs = distance_mtxs.tolist()

    return distance_mtxs


def clean_outliers_kpts(distance_mtxs, kpt2_clean):
    '''
    The function cleans the cartesian keypoints list according to distance matrix
    :param distance_mtxs: distance matrix of the cartesian keypoints
    :param kpt2_clean: list of cartesian kpts coordinates
    :return: list of cleaned cartesian kpts coordinates
    '''

    from statistics import mode

    min_dist = []
    max_dist = []

    for element in distance_mtxs:
        a = element
        a.sort()

        min_dist.append(a[1])
        max_dist.append(a[-10])

    # get list mode distance
    mode_dist = mode(min_dist)

    # create a list containing the index of the outliers to be removed
    to_remove = []
    for element in min_dist:
        idx = min_dist.index(element)

        if element < mode_dist-4 or element > mode_dist+4:
            to_remove.append(idx)

    # clean the input list from outliers
    count = 0
    for index in to_remove:
        # print("idx to remove ", index)

        kpt2_clean.pop(index - count)
        count += 1

    return kpt2_clean


def sort_kpts_coord(kpt_list):
    '''
    sorting function
    :param kpt_list: keypoint list
    :return: sorted  keypoint list
    '''
    kpt_list.sort()

    return kpt_list


def get_ScaleFactor_x_y(kpt_list1, kpt_list2, h, w):
    '''
    the function calculates scaling factors on x and y
    :param kpt_list1: keypoint list image A (RGB)
    :param kpt_list2: keypoint list image B (therm)
    :param h: thermal image height
    :param w: thermal image width
    :return: Sf_x, Sf_y, coord
            # scale factor x
            # scale factor y
            # alignment coordinates: xmin, xmax, ymin, ymax
    '''
    # list initialisation for storing RGB (A) and thermal (B) coordinates
    xA = []
    yA = []
    xB = []
    yB = []

    for element in kpt_list1:
        xA.append(element[0])
        yA.append(element[1])

    for element in kpt_list2:
        xB.append(element[0])
        yB.append(element[1])

    # min, max coord matrix A (RGB)
    min_A_x = min(xA)
    max_A_x = max(xA)

    min_A_y = min(yA)
    max_A_y = max(yA)

    # min, max coord matrix B (therm)
    min_B_x = min(xB)
    max_B_x = max(xB)

    min_B_y = min(yB)
    max_B_y = max(yB)

    f_x = 1  # 1280/1920
    f_y = 1  # 720/1080

    Sf_x = (max_A_x * f_x - min_A_x * f_x) / (max_B_x - min_B_x)
    Sf_y = (max_A_y * f_y - min_A_y * f_y) / (max_B_y - min_B_y)

    # coordinates for rgb clipping, therm resizing
    min_x = int(min_A_x * f_x - (min_B_x * Sf_x))
    max_x = int(max_A_x * f_x + ((w - max_B_x) * Sf_x))

    min_y = int(min_A_y * f_y - (min_B_y * Sf_y))
    max_y = int(max_A_y * f_y + ((h - max_B_y) * Sf_y))

    # alignment coords
    coord = (min_x, max_x, min_y, max_y)

    return Sf_x, Sf_y, coord


def resize_therm(im, coord):
    '''
    the function resizes a matrix
    :param im: matrix
    :param coord: alignment coords
    :return: resized matrix
    '''
    # new image size
    h2 = coord[3] - coord[2] # y2 - y1
    w2 = coord[1] - coord[0] # x2 - x1

    im = Image.fromarray(im)
    im_resized = im.resize((w2, h2))

    return im_resized


def crop_col(im, coord):
    '''
    the function crops a matrix
    :param im: matrix
    :param coord: alignment coords
    :return: cropped matrix
    '''
    # crop the image at specific AOI
    crop_img = im[coord[2]:coord[3], coord[0]:coord[1]]  # im[y1:y2, x1:x2]

    return crop_img


def get_correct_coord(csv_file1):
    '''
    the function identifies the average alignment coordinates from the alignment pannel output file
    :param csv_file1: txt file path to alignment coordinates data (xmin, xmax, ymin, ymax)
    :return: avg coords (xmin, xmax, ymin, ymax)
    '''

    df = pd.read_csv(csv_file1, sep=",")

    x_min = int(df.mean()[0])
    x_max = int(df.mean()[1])
    y_min = int(df.mean()[2])
    y_max = int(df.mean()[3])

    return x_min, x_max,y_min, y_max


def alignment(coords, col_path, therm_path):
    '''
    The fucntion alignes the thermal matrix 320x240 to 1280x720

    :param coords: alignment coords: x1, x2, y1, y2
    :param col_path: single color image directory
    :param therm_path: related thermal image directory
    :return: path to the color image, 1280-aligned thermal matrix
    '''

    from PIL import Image
    # define the alignment function
    def align_images(im_col, im_therm, coord):
        '''
        :param im_col: color matrix
        :param im_therm: thermal matrix
        :param coord: alignment coords xmin, xmax, ymin, ymax
        :return: color matrix cropped to AOI, resized and aligned-to-AOI thermal matrix
        '''

        colore = crop_col(im_col, coord)
        termica = resize_therm(im_therm, coord)
        return colore, termica

    # averaged alignment coords
    coord = (coords[0], coords[2], coords[1], coords[3])

    ### in field images
    # open the color image
    im_col = Image.open(col_path)
    im_col = np.asarray(im_col)
    plt.imshow(im_col)
    print(im_col.shape)
    # im_col = cv2.imread(col_path)


    # open related therm
    im_therm = Image.open(therm_path) # images_therm[count*1]

    # convert to np
    im_therm = np.asarray(im_therm, np.dtype('uint16'))

    # get the rgb matrix shape
    shape = im_col.shape

    # resize the color matrix to the lower resolution
    if shape[1] == 1920:
        # if field images are 1920 x 1080 resize to 1280 x 729, otherwise not
        im_col = cv2.resize(im_col, (1280, 720), interpolation=cv2.INTER_AREA)
    else:
        pass

    # align images to 1280 - AOI
    im_col, termica = align_images(im_col, im_therm, coord)


    return col_path, termica


def align_1920(termica, coord_txt, therm_fn):
    '''
    The fucntion alignes the thermal matrix 1280-aligned to 1920x1080 and saves the output at a specific location (fn)

    :param termica: 1280-aligned thermal matrix
    :param coord_txt: directory to coords txt file from "alignment process"
    :param sf_txt: directory to scale factors txt file from "alignment process"
    :param therm_fn: thermal image filename
    :return: None
    '''

    # thermal 1280
    im = termica
    # np 1280 thermal with AOI and thermal data to overlap to a larger frame
    im_overlap = np.asarray(im, np.dtype('uint16'))

    # get average alignment coordinates and scale factors from the output files of the panel alignment process
    csv_file = coord_txt
    # get UL alignment corner
    x_min, y_min, _, _ = get_calib_x_y_min(csv_file)

    # thermal 1280x720 - aligned shapes
    h, w = im_overlap.shape

    # matrix conversion factors from 1280 x 720 to 1920 x 1080

    f_x = 1 / (1280 / 1920)
    f_y = 1 / (720 / 1080)

    # resize the thermal aligned to take in consideration the shift factor(f) to 1920x1080
    # If you are enlarging the image, you should prefer to use INTER_LINEAR or INTER_CUBIC interpolation.
    # If you are shrinking the image, you should prefer to use INTER_AREA interpolation.

    h2 = int(h * f_y) # resized matrix height to fit with 1920x1080 dimension
    w2 = int(w * f_x) # resized matrix width to fit with 1920x1080 dimension

    # resize the thermal image to fit a 1920 x 1080 background

    im_overlap = Image.fromarray(im_overlap) # image with thermal data
    im_overlap_2 = im_overlap.resize((w2, h2)) # resizing the 1280 AOI to 1920

    # generating a zeros background canva with 1920x1080 dimension
    canva = np.zeros((1080, 1920), dtype='int16')

    # # the 1920 - aligned thermal AOi is copied on the zeros canva according to:
    # #       the average UL corner alignment coordinates
    # #       new matrix shape (h2, w2)

    # report x, y alignment errors from get_RMSE_avg_coord() from "Get_alignment_factor.py"
    x_bias = 4.17
    y_bias = 0.17

    # add the UL corners with mean error to correct the alignment
    # TODO: to be removed in future versions? does it have sense?
    x_min += x_bias
    y_min += y_bias

    # convert the thermal aligned to 1920 np array
    im_overlap_2 = np.asarray(im_overlap_2, dtype='int16')

    # the 1920 - aligned thermal AOi is copied onto the zeros canva
    #       [y min: y max, x min: x max] # rows, cols
    canva[int(y_min * f_y):int(y_min * f_y + h2), int(x_min * f_x):int(x_min * f_x + w2)] = im_overlap_2

    # canva 1920x1080 with thermal data
    canva = Image.fromarray(canva)
    # save the aligned to 1920 thermal matrix to destination filename
    canva.save(therm_fn)


def get_calib_x_y_min(txt_coord_path):
    '''
    get average coordinates from alignment panel output file
    :param txt_coord_path: path to txt coordinate output file from the panel alignment process
    :return: average x_min, y_min, x_max, y_max for field alignment
    '''

    # store the file path to all the coordinates from the alignment process
    csv_file = txt_coord_path

    # open the file
    df = pd.read_csv(csv_file, sep=",", names=["xmin", "xmax", "ymin", "ymax"], dtype=int)

    # calculate the mean and st. dev before cleaning the data related to x_min
    std = df["xmin"].std()
    mean = df["xmin"].mean()
    list_ = df["xmin"].index.tolist()

    row = 0  # indexer
    for i in df["xmin"]:

        # remove from the dataset the values higher/lower than a st dev from mean value
        if i < int(mean) - int(std) or i > int(mean) + int(std):
            df = df.drop(list_[row])
            row += 1
        else:
            row += 1

    # to clean according to y_min
    std = df["ymin"].std()
    mean = df["ymin"].mean()

    # list of the remaining row indexes after x_min cleaning
    list_ = df["ymin"].index.tolist()
    # print(list_)

    row = 0  # indexer
    for i in df["ymin"]:

        if i < int(mean) - int(std) or i > int(mean) + int(std):
            df = df.drop(list_[row])
            row += 1
        else:
            row += 1

    x_min = int(df["xmin"].mean())
    y_min = int(df["ymin"].mean())
    x_max = int(df["xmax"].mean())
    y_max = int(df["ymax"].mean())

    return x_min, y_min, x_max, y_max


def get_calib_Sfx_Sfy(txt_sf_path):
    '''
    get average SFx, SFy from alignment panel output file
    :param txt_sf_path: path to txt scale factors' output file from the panel alignment process
    :return: average Sf_x, Sf_y for field alignment
    '''

    # store the file path to all the scaling factors X, Y from the panel alignment process
    csv_file = txt_sf_path

    # open the file
    df = pd.read_csv(csv_file, sep=",", names=["Sfx", "Sfy"], dtype=float)

    # calculate the mean
    Sf_x = df["Sfx"].mean()
    Sf_y = df["Sfy"].mean()

    return Sf_x, Sf_y


def get_RMSE_avg_coord(kpt_list1, kpt_list2, im, coord, Sf,num, name):
    '''
    METHOD 2 - RMSE calculation with the in-field adopted avg coordinates and scale factors
    The function computes the RMSE error between thermal projection and actual blob locations

    :param kpt_list1: keypoints coords list from colour image
    :param kpt_list2: keypoints coords list from normalized thermal image
    :param im: color image
    :param coord: tuple of average alignment coords (xmin, ymin, xmax, ymax)
    :param Sf: tuple of average scale factors (sfx, sfy)
    :param num: tracking value for plotting results
    :param name: normalised 320x240 thermal image path&filename
    :return: rmse_dist, rmse_x, rmse_y, bias_x, bias_y
    '''

    # TODO: NEXT-YEAR TRY TO COMPUTE rmse ON WHOLE KPTS DATASET (n=30) instead of 4 CORNERS
    #   try to sort them properly on x and y direction

    plt.clf() # clean plot

    # get avg alignment coords
    xmin, ymin, xmax, ymax = coord[0], coord[1], coord[2], coord[3]

    # get avg scale factors
    Sf_x = Sf[0]
    Sf_y = Sf[1]

    # list initialisation to store RGB (A), thermal (B) coords
    xA = []
    yA = []
    xB = []
    yB = []

    for element in kpt_list1:  # RGB keypoints
        xA.append(element[0])
        yA.append(element[1])

    for element in kpt_list2:  # Therm keypoints
        xB.append(element[0])
        yB.append(element[1])

    #  min, max coord matrix A (RGB)
    min_A_x = min(xA)
    max_A_x = max(xA)

    min_A_y = min(yA)
    max_A_y = max(yA)

    # assembled points (P1, P2, P4, P3)
    A_coord = (min_A_x, max_A_x, min_A_y, max_A_y)
    A_coord_rmse = [(A_coord[0], A_coord[2]), (A_coord[0], A_coord[3]), (A_coord[1], A_coord[2]),
                    (A_coord[1], A_coord[3])]

    # min, max coord matrix B (therm)
    min_B_x = (xmin - min_A_x) / -Sf_x
    max_B_x = -((xmax - max_A_x) / Sf_x - 320)

    min_B_y = (ymin - min_A_y) / -Sf_y
    max_B_y = -((ymax - max_A_y) / Sf_y - 240)

    # computing for all thermal - 4 points the therm  projection (method 1 does it only on image corners)
    min_B_x = min_B_x * Sf_x + min_A_x * 1280 / 1920
    max_B_x = max_B_x * Sf_x + min_A_x * 1280 / 1920

    min_B_y = min_B_y * Sf_y + min_A_y * 720 / 1080
    max_B_y = max_B_y * Sf_y + min_A_y * 720 / 1080

    # assembled projected points (P1, P2, P4, P3)
    B_coord = (min_B_x, max_B_x, min_B_y, max_B_y)
    B_coord_rmse = [(B_coord[0], B_coord[2]), (B_coord[0], B_coord[3]), (B_coord[1], B_coord[2]),
                    (B_coord[1], B_coord[3])]

    #  x & y values used to calculate RMSE
    dist_rmse = []
    x_rmse = []
    y_rmse = []
    x_bias = []
    y_bias = []

    count = 0
    for coord in B_coord_rmse:
        img = im.copy()
        count += 1

        # RGB-projected thermal keypoints coord.
        # the coefficients 1280/1920 & 720/1080 (=0.66[...])
        # are used to take in account for the RGB resizing, which
        # affects the offset on x and y values

        x_b = coord[0]  # *1280/1920
        y_b = coord[1]  # *720/1080

        idx = B_coord_rmse.index(coord)

        # RGB kpts coord
        x_a = A_coord_rmse[idx][0]
        y_a = A_coord_rmse[idx][1]

        plt.imshow(img, cmap='Greys')
        if count == 0 and coord == B_coord_rmse[0]:
            plt.plot(x_a, y_a, "rs", markersize=1)
            plt.plot(x_b, y_b, "bo", markersize=1)


        elif coord == B_coord_rmse[-1]:
            plt.plot(x_a, y_a, "rs", markersize=1, label="RGB")
            plt.plot(x_b, y_b, "bo", markersize=1, label="therm proj")
            plt.legend(loc="center right")
            plt.savefig((name[-5:] + str(num) + ".png"), dpi=600)
        else:
            plt.plot(x_a, y_a, "rs", markersize=1)
            plt.plot(x_b, y_b, "bo", markersize=1)

        num += 1

        dist = int(math.sqrt((x_b - x_a) ** 2 + (y_b - y_a) ** 2))
        rmse_x = int((x_b - x_a) ** 2)
        rmse_y = int((y_b - y_a) ** 2)

        bias_x = int((x_b - x_a))
        bias_y = int((y_b - y_a))

        x_rmse.append(rmse_x)
        y_rmse.append(rmse_y)
        dist_rmse.append(dist)

        x_bias.append(bias_x)
        y_bias.append(bias_y)

    num +=1

    rmse_dist = int(mean(dist_rmse))
    rmse_x = int(math.sqrt(mean(x_rmse)))
    rmse_y = int(math.sqrt(mean(y_rmse)))

    bias_x = int(sum(x_bias) / len(x_bias))
    bias_y = int(sum(y_bias) / len(y_bias))

    return rmse_dist, rmse_x, rmse_y, bias_x, bias_y

def get_dir_fn(file_path):
    '''
    :param file_path: - filename
    :return: the file directory
    '''
    a = file_path
    len_ = len(a)

    # loop
    for i in range(0, len_):

        loc = a[len_ - i:len_]

        if loc.find("\\") != -1:  # se diverso da "-1" vuol dire che ha trovato il carattere
            index = len_ - i

            aligned_dir = str(a[0:index] + "_aligned")

            try:
                os.mkdir(aligned_dir)
            except:
                pass

            img_name = str(a[index + 1:-4])

            return aligned_dir, img_name
        else:
            pass


''''
### to convert  Raw Thermal APP to Celsius APP (from lightbulbs datasets):
  
distance	    Slope(a coeff.) 	Intercept (b coeff.)
	
General		
0.5m - 3.0m 	0.01286527470000000	 -28.62827940000000000
			
Raw <3000	
0.5m	        0.01790580184650830	-41.62511947512870000
1.0m	        0.01709801137150510	-39.87904853972370000
1.5m	        0.01671930703052060	-38.82596994946430000
2.0m	        0.01648162214344660	-37.84663166550710000
2.5m	        0.01636317146858520	-37.77315677927730000
3.0m	        0.01592340420569740	-36.11466921652480000
			
Raw 3000-7000	
0.5m	        0.01122734467909650	-22.07632178280650000
1.0m	        0.01121187326341270	-22.00337506467050000
1.5m	        0.01112760151137210	-21.64788385102760000
2.0m	        0.01112956235348940	-21.55103524359960000
2.5m	        0.01106067709941020	-21.37977980029620000
3.0m	        0.01111150330406360	-21.29391749734170000
			
Raw >7000	
0.5m	        0.00802669863747835	-0.37960504788257500
1.0m	        0.00810378210333579	-1.07049399213111000
1.5m	        0.00794082388563919	0.21424855976091400
2.0m	        0.00785087065611661	1.01781066231241000
2.5m	        0.00785149354366968	0.93286503219869300
3.0m	        0.00400799716597930	29.31449127197260000

### to convert  Raw Thermal APP to Celsius APP (from steel bottles datasets):

{'Generale': (0.0123896633, -22.9001998),
 'Raw <2655': {'0.5m': (0.01916293450358146, -39.804689225382454),
  '1.0m': (0.01905148541488177, -39.688738847082874),
  '1.5m': (0.01934585931445127, -40.5132537331243),
  '2.0m': (0.019348790412194544, -40.50448688735466),
  '2.5m': (0.019247944915376018, -40.25002072060615),
  '3.0m': (0.019214346836511745, -40.020404777398)},
 'Raw 2655-6000': {'0.5m': (0.011902342953571303, -21.010625590861586),
  '1.0m': (0.01178822876435835, -20.658997872766488),
  '1.5m': (0.01168269775404037, -20.229585346054904),
  '2.0m': (0.011652903863122864, -20.097671443987412),
  '2.5m': (0.011624608568243778, -19.975095861289343),
  '3.0m': (0.011610997989096192, -19.807642938527408)},
 'Raw >6000': {'0.5m': (0.01100191799354364, -16.3762502078527),
  '1.0m': (0.01101393479404235, -16.662159103716082),
  '1.5m': (0.011034945032283226, -16.834598763075267),
  '2.0m': (0.011197814149822334, -17.806635947915602),
  '2.5m': (0.010749495822503913, -15.026483476506266),
  '3.0m': ('a', 'b')}}
'''

def get_max_temp_quantiles(img_numpy_raw_thermal):
    '''
    Th function identifies the warmer spots in the thermal matrix thanks to quantile filtering
    :param img_numpy_raw_thermal: raw thermal matrix
    :return: min_Celsius_temp, mean_Celsius_temp, max_Celsius_temp, Celsius_filtered_matrix
    '''

    # compute the 70th quantile on raw values != from 0
    quantile = np.quantile(img_numpy_raw_thermal[img_numpy_raw_thermal!=0], q=0.7)
    # assign 0 to pixel values lower than the quantile
    img_numpy = np.where(img_numpy_raw_thermal < quantile, 0, img_numpy_raw_thermal)

    # Ros-to-app raw data conversion
    img_numpy = np.where(img_numpy != 0,
                         img_numpy * 1.16550762 - 13529.4453,
                         0)

    # app-converted raw data to Celsius
    #   dist 2.5m & Raw <= 3000
    img_numpy = np.where((img_numpy != 0) & (img_numpy <= 3000),
                         img_numpy * 0.01636317146858520 -37.77315677927730000,
                         img_numpy)

    #   dist 2.5m & Raw > 3000 & Raw <= 7000
    img_numpy = np.where((img_numpy != 0) & (img_numpy > 3000) & (img_numpy <= 7000),
                         img_numpy * 0.01106067709941020 -21.37977980029620000,
                         img_numpy)

    #   dist 2.5m & Raw > 7000
    img_numpy = np.where((img_numpy != 0) & (img_numpy > 7000),
                         img_numpy * 0.00785149354366968 + 0.93286503219869300,
                         img_numpy)

    # get warmer spots':
    #   min temp [°C]
    min_temp = np.min(img_numpy[img_numpy != 0])
    #   mean temp [°C]
    mean_temp = np.mean(img_numpy[img_numpy != 0])
    #   max temp [°C]
    max_temp = np.max(img_numpy[img_numpy != 0])
    return min_temp, mean_temp, max_temp, img_numpy


# THE FOLLOWING SCRIPT IS ASSUMED TO BE USED FOR THERMAL CAMERA ALIGNMENT WITH RGBD AND RMSE CALCULATION


import glob
import cv2
# per fare statistiche su liste
from statistics import mean
import Thermal_sheet


''''
THE FOLLOWING CODE IS ENCHARGED FOR THE THERMAL ALIGNMENT
'''

# where to get the images to align
images_col = glob.glob(r"colore\*.png")
images_therm = glob.glob(r"termico\*.png")

# where to save the output .txt files with calibration data
csv_file1 = r"colore\alignment_coords_SHEET_deliverable.txt"
csv_file2 = r"colore\alignment_scaleFactors_SHEET_deliverable.txt"

# where to store the average coordinates for in-field alignment
csv_file = r'colore\alignment_coords_SHEET_deliverable.txt'


# initialise lists
count = 0
coord_list = []  # list with coords to convert into txt
scale_list = []  # list with scale factors to convert into txt

num = 0
print("Images successfully used for alignment:")
for im in images_col:
    # get related image
    therm = images_therm[count]
    print(im)
    count += 1

    im_col = cv2.imread(im)
    # get color image shape
    shape = im_col.shape

    if shape[1] == 1920:
        # se in campo prendi le immagini in 1920 e 1280 fai il resize, altrimenti no
        im_col = cv2.resize(im_col, (1280, 720), interpolation=cv2.INTER_AREA)
    else:
        pass

    # open thermal image
    im_therm = cv2.imread(therm)

    # blob detection
    #   color
    keypoints_col = Thermal_sheet.blob_detector_col(im)
    #   thermal
    keypoints_therm = Thermal_sheet.blob_detector_therm(therm)

    # from the keypoint list get the respective coordinates in pixel (row, col)
    #   color
    col_kpt = Thermal_sheet.kpt2_coord(keypoints_col)
    #   therm
    therm_kpt = Thermal_sheet.kpt2_coord(keypoints_therm)

    # show detected keypoints on related images
    im1 = im_col.copy()
    im2 = im_therm.copy()
    # Thermal_sheet.show_dynamic_keypoints(im1, col_kpt)
    # Thermal_sheet.show_dynamic_keypoints(im2, therm_kpt)

    # check if both images have at least 30 kpts, otherwise don't consider them in the alignment
    if len(col_kpt) < 30 or len(therm_kpt) < 30:
        num += 1
        pass
    else:

        # compute distance matrix
        col_dist_mtx = Thermal_sheet.distance_matrix(col_kpt)
        therm_dist_mtx = Thermal_sheet.distance_matrix(therm_kpt)

        # remove outliers thanks to distance matrix (false keypoints)
        try:
            #   color
            col_kpt = Thermal_sheet.clean_outliers_kpts(col_dist_mtx, col_kpt)
        except:
            pass

        try:
            #   therm
            therm_kpt = Thermal_sheet.clean_outliers_kpts(therm_dist_mtx, therm_kpt)
        except:
            pass

        # sort keypoints
        #   color
        col_kpt = Thermal_sheet.sort_kpts_coord(col_kpt)
        #   therm
        therm_kpt = Thermal_sheet.sort_kpts_coord(therm_kpt)

        # show keypoints to visualise the effect of the filtering
        im1 = im_col.copy()
        im2 = im_therm.copy()
        # Thermal_sheet.show_dynamic_keypoints(im1, col_kpt)
        # Thermal_sheet.show_dynamic_keypoints(im2, therm_kpt)

        # check if both images have 30 cleaned kpts, otherwise don't consider them in the alignment
        if len(col_kpt) != 30 or len(therm_kpt) != 30:
            num += 1
            pass
        else:
            # show keypoints
            im1 = im_col.copy()
            # Thermal_sheet.show_dynamic_keypoints(im1, col_kpt)

            im2 = im_therm.copy()
            # Thermal_sheet.show_dynamic_keypoints(im2, therm_kpt)

            # get thermal image size
            h, w, c = im_therm.shape

            # get, coords for the overlap
            Sf_x, Sf_y, coord = Thermal_sheet.get_ScaleFactor_x_y(col_kpt, therm_kpt, h, w)

            # add coordinates to the storing list
            coord_list.append(coord)
            scale_list.append((Sf_x, Sf_y))

            # resized color image to adapt with the resized thermal
            im3 = im_col.copy()
            crop_img = Thermal_sheet.crop_col(im3, coord)

            # resized thermal to 1280 x 720
            im_therm = Thermal_sheet.resize_therm(im_therm, coord)
            print(num)
            num += 1

print("# coord: ", len(coord_list))

# export list to .txt
file = open(csv_file1, "w")
file.close()
with open(csv_file1, "a") as f:
    for i in coord_list:
        f.write((str(i[0]) + "," + str(i[1]) + "," +
                 str(i[2]) + "," + str(i[3]) + "\n"))
f.close()

file = open(csv_file2, "w")
file.close()

with open(csv_file2, "a") as f:
    for i in scale_list:
        f.write((str(i[0]) + "," + str(i[1]) + "\n"))
f.close()

# cv2.destroyAllWindows()

''''
THE FOLLOWING CODE IS NEEDED FOR RMSE CALCULATION
'''
print("\n")

# get average coordinates from alignment panel output file
xmin, ymin, xmax, ymax = Thermal_sheet.get_calib_x_y_min(csv_file)
avg_coord = (xmin, ymin, xmax, ymax)

# get average SFx, SFy from alignment panel output file
sfx, sfy = Thermal_sheet.get_calib_Sfx_Sfy(csv_file2)
SF = (sfx, sfy)

# initialise lists
count = 0
coord_list = []
scale_list = []

dist_rmse = []
x_rmse = []
y_rmse = []

x_bias = []
y_bias = []

for im in images_col:
    # get rgb related therm matrix
    therm = images_therm[count]
    count += 1

    # open the rgb image in the rgb colorspace
    im_col = cv2.cvtColor(cv2.imread(im), cv2.COLOR_BGR2RGB)
    # get matrix shape
    shape = im_col.shape

    if shape[1] == 1920:
        # if not 1920x1080 convert to 1280x720
        im_col = cv2.resize(im_col, (1280, 720), interpolation=cv2.INTER_AREA)
    else:
        pass
    # open thermal image
    im_therm = cv2.imread(therm)

    # blob detection
    #  color
    keypoints_col = Thermal_sheet.blob_detector_col(im)
    #  thermal
    keypoints_therm = Thermal_sheet.blob_detector_therm(therm)

    # from the keypoint list get the respective coordinates in pixel
    #  color
    col_kpt = Thermal_sheet.kpt2_coord(keypoints_col)
    #  THERMAL
    therm_kpt = Thermal_sheet.kpt2_coord((keypoints_therm))

    # if less than 30 kpts for both detections, don't consider
    if len(col_kpt) < 30 or len(therm_kpt) < 30:
        pass
    else:

        # compute distance matrix
        col_dist_mtx = Thermal_sheet.distance_matrix(col_kpt)
        therm_dist_mtx = Thermal_sheet.distance_matrix(therm_kpt)

        # remove outliers (false keypoints)
        try:
            # color
            col_kpt = Thermal_sheet.clean_outliers_kpts(col_dist_mtx, col_kpt)
        except:
            pass

        try:
            # thermal
            therm_kpt = Thermal_sheet.clean_outliers_kpts(therm_dist_mtx, therm_kpt)
        except:
            pass

        # sort keypoints
        #   color
        col_kpt = Thermal_sheet.sort_kpts_coord(col_kpt)
        #   thermal
        therm_kpt = Thermal_sheet.sort_kpts_coord(therm_kpt)

        # if not 30 kpts for both images, don't consider in the alignment
        if len(col_kpt) != 30 or len(therm_kpt) != 30:
            pass
        else:

            # RMSE from average coords
            num = 0
            rmse_dist, rmse_x, rmse_y, bias_x, bias_y = Thermal_sheet.get_RMSE_avg_coord(col_kpt,
                                                                                         therm_kpt,
                                                                                         im_col,
                                                                                         avg_coord,
                                                                                         SF,
                                                                                         num,
                                                                                         therm)
            dist_rmse.append(rmse_dist)
            x_rmse.append(rmse_x)
            y_rmse.append(rmse_y)
            x_bias.append(bias_x)
            y_bias.append(bias_y)

# print(rmse)
rmse_dist = mean(dist_rmse)
rmse_x = mean(x_rmse)
rmse_y = mean(y_rmse)
bias_x = mean(x_bias)
bias_y = mean(y_bias)
print("RMSE:\n RAGGIO " + str(rmse_dist) + ", X: " + str(rmse_x) + ", Y: " + str(rmse_y) + " BIAS X/Y: "
      + str(bias_x) + " / " + str(bias_y))

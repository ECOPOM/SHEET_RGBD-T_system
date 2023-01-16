import glob
from PIL import Image
import Thermal_sheet
from XYZ_functions import *
import INPUTS as c
import Thermal_correction as tc


def yolo_coords(df):
    '''
    The function extracts the YOLO bbox coordinates from label df rows
    :param df: YOLO txt df row with bbox info
    :return: x_c, y_c, w, h, lab
    '''

    x_c = df['x_c']
    y_c = df['y_c']
    w = df['w']
    h = df['h']
    lab = df['label']

    return x_c, y_c, w, h, lab


# INITIALISED PROCESS
#       import needed data from INPUTS.py
rgb_imgs = glob.glob(c.RGB_PATH + '\\*.' + c.IMAGE_FORMAT)
img_path_term = c.RAW_THERM_PATH +'_aligned'
df_labels= glob.glob(c.FRUIT_DETECT_YOLO_LABELS_PATH + '\\*' + '.txt')
therm_imgs = glob.glob(c.RGB_PATH + '\\*.' + c.IMAGE_FORMAT)
image_cardinal_side = c.IMAGE_CARDINAL_FACE
trunk_set_dist_from_camera = c.TRUNK_DIST*1000
depth_path = c.DEPTH_PATH
trunk_detect_labels_path = c.TRUNK_DETECT_YOLO_LABELS_PATH
out_df = c.OUTPUT_DF_PATH

# Import Thermal correction function (error adding)
min_temp_correction_func = tc.poly1d_min_err # requires distance in meters
max_temp_correction_func = tc.poly1d_max_err # requires distance in meters
avg_temp_correction_func = tc.poly1d_avg_err # requires distance in meters
print(f'{min_temp_correction_func= }')
print(f'{max_temp_correction_func= }')
print(f'{avg_temp_correction_func= }')

# initialise the database for storing info
df = pd.DataFrame(columns=[
    'name', 'label', 'ID_tree',
    'sec1', 'Tmin_SEEK', 'Tmean_SEEK', 'Tmax_SEEK',
    'sec2', 'Tmin_SEEK_corr_avg', 'Tmean_SEEK_corr_avg', 'Tmax_SEEK_corr_avg',
                'Tmin_SEEK_corr_max', 'Tmean_SEEK_corr_max', 'Tmax_SEEK_corr_max',
    'sec3', 'X_mm', 'Y_mm', 'Y_cam_mm', 'Z_mm', 'X_trunk_mm', 'Y_trunk_mm', 'Y_trunk_cam_mm', 'Z_trunk_mm',
    'sec4', 'X_relativo', 'Y_relativo', 'Z_relativo', 'X_trunk_rel', 'Y_trunk_rel', 'Z_trunk_rel',
    'sec5', 'X_rel_range_mm', 'Y_rel_range_mm', 'Z_rel_range_mm', 'X_trunk_rel_mm', 'Y_trunk_rel_mm', 'Z_trunk_rel_mm',
    'sec6', 'Tree_wall', 'Card_dir', 'Estim_Fruit_diam_mm'])

id_track = 0  # for building the df
id_tree = 0  # for tree tracking

for lab in df_labels:
    # get related files
    _, fn, depth, therm_image = Thermal_sheet.get_rel_therm_fns(lab, img_path_term, depth_path=depth_path)

    # open raw aligned THERMAL
    therm_raw = Image.open(therm_image)

    # conversion to np array
    t_img = np.asarray(therm_raw)

    # open related DEPTH MAP
    depth_img = Image.open(depth)

    # conversion to np array
    depth_img = np.asarray(depth_img)
    print("\n", f'{depth= }')

    # positioning info are not needed, just thermal data are required
    # open related TRUNK COORDS
    trunk_lab = trunk_detect_labels_path +'\\'+fn+'.txt'

    # get trunk bbox's centre x and y coords
    df_trunk = pd.read_csv(trunk_lab, sep=" ", names=["label", "x_c", "y_c", "w", "h"])
    xc_trunk = int(df_trunk.loc[0]['x_c']*depth_img.shape[1])
    yc_trunk = int(df_trunk.loc[0]['y_c']*depth_img.shape[0])
    w_trunk = int(df_trunk.loc[0]['w']*depth_img.shape[1])
    h_trunk = int(df_trunk.loc[0]['h']*depth_img.shape[0])
    x_trunk = int(xc_trunk + w_trunk/2)
    y_trunk = yc_trunk

    # store trunk coords into a dict
    trunk_coords = {'x_c': x_trunk, 'y_c': y_trunk}

    # open fruit yolo labels
    df_labels = pd.read_csv(lab, sep=" ", names=["label", "x_c", "y_c", "w", "h"])

    for idx, row in df_labels.iterrows():
        # from yolo get coords
        x_c, y_c, w, h, obj_lab = yolo_coords(row)

        # compute bbox position with np format
        x_min = int((x_c - w / 2) * t_img.shape[1])
        x_max = int((x_c + w / 2) * t_img.shape[1])
        y_min = int((y_c - h / 2) * t_img.shape[0])
        y_max = int((y_c + h / 2) * t_img.shape[0])

        # clip image according to bbox coords in np format
        img = t_img[y_min:y_max, x_min:x_max]

        # temperature bbox filtering
        if np.mean(img) == 0 or np.mean(img) < 14500:  # 14500 filters bboxes not fully representing objects
            print('passed bbox - not a full apple')
            pass
        else:
            # TEMPERATURE CALCULATION
            #   warmer spots detection
            min_temp, mean_temp, max_temp, img_numpy = Thermal_sheet.get_max_temp_quantiles(img)

            # FRUIT 3D POSITIONING
            #   xc = relative fruit centre X coord in millimetres with respect to the trunk
            #   yc = relative fruit centre Y coord in millimetres with respect to the trunk
            #   zc = fruit distance in millimetres with respect to 'z_trunk_real' detected from the Dbbox
            #   z_trunk = trunk distance from camera
            #   xc_relativo = relative fruit centre X coord in 0-1 values with respect to the trunk
            #   yc_relativo = relative fruit centre Y coord in 0-1 values with respect to the trunk
            #   zc_relativo = relative fruit centre Z coord in 0-1 values with respect to the trunk
            #   d_linear = linear distance between fruit centre and trunk position (pixels)
            #   a = fruit radious on X: ellipsoid r/2
            #   b = fruit radious on Y: ellipsoid r/2
            #   c = fruit radious on Z: ellipsoid r/2
            #   zc2 = fruit distance in millimetres alligned to 'z_trunk'
            #   x2_trunk = trunk distance detected from the Dbbox

            xc, yc, zc, z_trunk, \
            xc_relativo, yc_relativo, zc_relativo, \
            d_linear, a, b, c, zc2, x2_trunk = positioning_occurrence(depth_img, row, trunk_coords)

            # Objects cardinal orientation according to relative coordinates with respect to shooting tree side
            if xc_relativo < 0 and image_cardinal_side.lower() == 'e':
                card_dir = 'N'
            elif xc_relativo > 0 and image_cardinal_side.lower() == 'e':
                card_dir = 'S'
            elif xc_relativo < 0 and image_cardinal_side.lower() == 'w':
                card_dir = 'S'
            elif xc_relativo > 0 and image_cardinal_side.lower() == 'w':
                card_dir = 'N'
            elif xc_relativo < 0 and image_cardinal_side.lower() == 's':
                card_dir = 'E'
            elif xc_relativo > 0 and image_cardinal_side.lower() == 's':
                card_dir = 'W'
            elif xc_relativo < 0 and image_cardinal_side.lower() == 'n':
                card_dir = 'W'
            elif xc_relativo > 0 and image_cardinal_side.lower() == 'n':
                card_dir = 'E'

            # TEMPERATURE CORRECTION FOR DISTANCE ERRORS
            #   imported functions are used
            min_temp_corr_avg = float(min_temp + avg_temp_correction_func(zc2 / 1000))
            min_temp_corr_max = float(min_temp + max_temp_correction_func(zc2 / 1000))
            max_temp_corr_avg = float(max_temp + avg_temp_correction_func(zc2 / 1000))
            max_temp_corr_max = float(max_temp + max_temp_correction_func(zc2 / 1000))
            mean_temp_corr_avg = float(mean_temp + avg_temp_correction_func(zc2 / 1000))
            mean_temp_corr_max = float(mean_temp + max_temp_correction_func(zc2 / 1000))

            print(f'{mean_temp_corr_max = }')
            # filling the database
            df.loc[id_track] = ([fn, int(obj_lab), id_tree,
                     'Raw_Temperatures [Celsius]', min_temp, mean_temp, max_temp,
                     'Corrected_Temperatures [Celsius]', min_temp_corr_avg, mean_temp_corr_avg, max_temp_corr_avg,
                                 min_temp_corr_max, mean_temp_corr_max, max_temp_corr_max,
                     'Abs_coords [mm]', float(yc+id_tree), float(zc2 - trunk_set_dist_from_camera), float(zc2),
                                 float(xc), id_tree, z_trunk - trunk_set_dist_from_camera, z_trunk, 0,
                     'Rel_coords',  yc_relativo, float(zc_relativo), xc_relativo, 0, 0, 0,
                     'rel_coords_mm', 'A', 'A', 'A', 0, 0, 0,
                     'other_info', image_cardinal_side.upper(), card_dir,  2*a])
            id_track += 4
    print('end tree')
    id_tree += 1000

# compute each fruit relative coordinates in mm with respect to trunk
df['X_rel_range_mm'] = df['X_mm'] - df['X_trunk_mm']
df['Y_rel_range_mm'] = df['Y_mm']
df['Z_rel_range_mm'] = df['Z_mm']

# save the dataset
df.to_csv(out_df, sep=' ', index=False)


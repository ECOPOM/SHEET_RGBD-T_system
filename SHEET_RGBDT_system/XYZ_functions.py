import numpy as np
import pandas as pd
import INPUTS as c

# DISTANCE IN METERS
camera_trunk_dist = c.TRUNK_DIST

def positioning_occurrence(depth, row, trunk_coords):
    '''
    The function computes the 3D positioning of detected objects
    :param depth: depth image
    :param row: fruit Dbbox coordinates (i.e., AOI)
    :param trunk_coords: trunk coords dict {'x_c': x_trunk, 'y_c': y_trunk}
    :return:
            xc2 = relative fruit centre X coord in millimetres with respect to the trunk
            yc2 = relative fruit centre Y coord in millimetres with respect to the trunk
            zc = fruit distance in millimetres with respect to 'z_trunk_real' detected from the Dbbox
            z_trunk = trunk distance from camera
            xc_relativo = relative fruit centre X coord in 0-1 values with respect to the trunk
            yc_relativo = relative fruit centre Y coord in 0-1 values with respect to the trunk
            zc_relativo = relative fruit centre Z coord in 0-1 values with respect to the trunk
            d_linear =  linear distance between fruit centre and trunk position (pixels)
            a = fruit radious on X: ellipsoid r/2
            b = fruit radious on Y: ellipsoid r/2
            c = fruit radious on Z: ellipsoid r/2
            zc2 = fruit distance in millimetres alligned to 'z_trunk'
            z_trunk_real = trunk distance detected from the Dbbox
            '''

    trunk_tree_dist_mm = camera_trunk_dist * 1000

    # processing
    depth = np.array(depth)
    im_np = np.array(depth).astype('float64')

    # fruit
    x_c = int(row['x_c'] * 1920)
    y_c = int(row['y_c'] * 1080)
    w = int(row['w'] * 1920)
    h = int(row['h'] * 1080)

    # computing the Dbbox coords
    x1 = x_c - w // 2
    x2 = x_c + w // 2
    y1 = y_c - h // 2
    y2 = y_c + h // 2

    # clip the fruit on the depth map
    im_trunk = im_np.copy()
    im_np = im_np[y1:y2, x1:x2]

    # get trunk X, Y coords
    x_trunk = trunk_coords['x_c']
    y_trunk = trunk_coords['y_c']
    print(f'{x_trunk= }')


    # extract from 5x20 matrix the trunk distance from camera
    im_trunk = im_trunk[y_trunk - 10:y_trunk + 10, x_trunk - 5:x_trunk]
    im_trunk = np.where(im_trunk == 0, np.nan, im_trunk)
    # print(im_trunk)
    z_trunk = np.nanmean(im_trunk)  # mm
    # print(z_trunk)

    x_coords = []
    y_coords = []
    z_coords = []

    # transform the 2D array into 3D:
    for x in range(0, im_np.shape[1]):
        for y in range(0, im_np.shape[0]):
            x_coords.append(x)
            y_coords.append(y)
            z = im_np[y, x]
            z_coords.append(z)

    x = np.asarray(x_coords)
    y = np.asarray(y_coords)
    z = np.asarray(z_coords)
    # print(z)

    # distance cleaning
    z = np.where(z == 0, np.nan, z)
    z = np.where(z > 3000, np.nan, z)

    # get min and max distance values
    z_list = z_coords.copy()
    z_list.sort()

    # position [0:2]
    np3d = np.vstack([x, y, z])
    print(np.nanmin(np3d[2]))

    # get non NAN values as array
    non_nan_dist = np3d[2][~np.isnan(np3d[2])]

    print(np.min(non_nan_dist), " min distance")

    # find all uniques values
    unique, cc = np.unique(non_nan_dist.astype('int'), return_inverse=True)
    print(unique)

    # find pixel value occurrence from minimum value to maximum
    occurrencies = np.bincount(non_nan_dist.astype('int'))

    # pack occurrencies with distances

    list_values = []
    count = 0
    for value in range(occurrencies[unique[0]], np.max(occurrencies) + 1):
        d = np.where(occurrencies == value)
        list_values.append((value, d))
        count += 1

    # clean extraction
    count_dist_list = []
    for i in list_values:
        if len(i[1][0]) == 0:
            pass
        else:
            count_dist_list.append((i[0], i[1][0]))

    # create a dataframe from extracted pairs of occurrence and distance
    df_dist = pd.DataFrame(columns=['count', 'dist'])

    count = 0
    for i in count_dist_list:

        for j in range(0, len(i[1])):
            f = np.array(i[1])

            df_dist.loc[count] = [i[0], f[j]]
            count += 1

    df_dist = df_dist.loc[df_dist['count'] >= 20]
    # TODO: use dynamic % instead of fixed value (20) for data cleaning

    print('\n', df_dist)

    # get minimum cleaned distance and average most occurrent distance (probable apple)
    #       minimum dist: average distance - 40mm
    #       average distance: considered as the one with max occurrence/count (or average stat. is better?)
    grouped_df = df_dist.groupby('count').max()
    max_count = df_dist['count'].max()
    z_avg = np.array(grouped_df.loc[grouped_df.index == max_count]['dist'])
    z_min = z_avg - 40
    # TODO: (40) which value ? or IEEE method 50% size :
    #   D. Mengoli, G. Bortolotti, M. Piani, and L. Manfrini,
    #   ‘On-line real-time fruit size estimation using a depth-camera sensor’,
    #   in 2022 IEEE Workshop on Metrology for Agriculture and Forestry (MetroAgriFor),
    #   Nov. 2022, pp. 86–90. doi: 10.1109/MetroAgriFor55389.2022.9964960.
    z_max = z_avg + 40
    print(f'{z_avg= }')
    print(f'{z_min= }')

    # bbox depth cleaning
    x = np.asarray(x_coords)
    y = np.asarray(y_coords)
    z = np.asarray(z_coords)

    # distance cleaning
    z = np.where(z == 0, np.nan, z)
    z = np.where(z > z_max, np.nan, z)
    z = np.where(z < z_min, np.nan, z)

    # position [0:2] - PACKING DATA
    # np3d = [[x coords],
    #         [y coords],
    #         [z coords]] --> shape = ( 3 info, variable number of columns (i.e., pixels) )
    np3d_fruit_clean = np.vstack([x, y, z])

    # get distance values from all non NaN pixels
    filtered_dist = np.where(~np.isnan(np3d_fruit_clean[2]))

    # get min, max X coords from min and max distance among all non NaN distance pixels
    x_min = np3d_fruit_clean[0, min(filtered_dist[0])]
    x_max = np3d_fruit_clean[0, max(filtered_dist[0])]

    a = int((x_max - x_min) / 2) # TODO: if improved 'a' could be used for fruit sizing: ellipsoid r/2
    xc = x_c  # fruit bbox centre
    # relative position (0:1) of the fruit centre with respect to trunk position
    xc_relativo = 0 - (xc - x_trunk) / x_trunk

    # get min, max Y coords from min and max distance among all non NaN distance pixels
    y_min = np3d_fruit_clean[1, min(filtered_dist[0])]
    y_max = np3d_fruit_clean[1, max(filtered_dist[0])]
    b = int((y_max - y_min) / 2) # TODO: if improved 'b' could be used for fruit sizing: ellipsoid r/2

    yc = y_c  # fruit bbox centre
    # relative position (0:1) of the fruit centre with respect to trunk position
    yc_relativo = 0 - (yc - y_trunk) / y_trunk

    z_min = np.nanmin(np3d_fruit_clean[2])
    z_max = np.nanmax(np3d_fruit_clean[2])
    c = (z_max - z_min) / 2 # TODO: if improved 'c' could be used for fruit sizing: ellipsoid r/2
    zc = z_avg # most occurrent fruit distance in Dbbox

    FOV_h = 69  # FOV gradi
    FOV_v = 42  # FOV gradi

    risoluzione = (1920, 1080)
    distanza_oggetto_mm = zc  # millimetri

    # horizontal
    IFOV_h_mrad = FOV_h / risoluzione[0] * ((3.14 / 180) * 1000)  # milliradianti
    IFOV_h_mm = IFOV_h_mrad / 1000 * distanza_oggetto_mm  # millimetri

    # vertical
    IFOV_v_mrad = FOV_v / risoluzione[1] * ((3.14 / 180) * 1000)  # milliradianti
    IFOV_v_mm = IFOV_v_mrad / 1000 * distanza_oggetto_mm  # millimetri

    # new apple radious in millimetres
    a_mm = a * IFOV_h_mm
    b_mm = b * IFOV_v_mm

    print('PIXEL COORDS: ', x_max, x_min, y_max, y_min, z_max, z_min)  # PIXEL COORDS
    print('RADII: ', a, b, c)  # RADII
    print('RADII IN MM: ', a_mm, b_mm, c)  # RADII IN MM
    print('APPLE CENTER COORDS: ', xc, yc, zc)  # APPLE CENTER COORDS

    # linear distance between fruit centre and trunk position
    d_linear = np.sqrt(((xc - x_trunk) ** 2 + (yc - y_trunk) ** 2))
    # relative position (0:1) of the fruit centre with respect to trunk position
    zc_relativo = 0 - (z_trunk - zc) / z_trunk  # mm 0-(z_trunk - zc)/z_trunk

    # shifting all depth coordinates to align to fixed camera - tree distance (i.e., 2.8 m)
    z_shift = -(z_trunk - trunk_tree_dist_mm)
    z_trunk_real = z_trunk
    z_trunk = trunk_tree_dist_mm
    zc2 = zc + z_shift

    # xc --> z axis
    # relative coordinates with respect to the trunk (millimetres)
    xc2 = (x_trunk - xc) * IFOV_h_mm
    yc2 = ((y_trunk - yc) * -1) * IFOV_v_mm

    return xc2, yc2, zc, z_trunk, xc_relativo, yc_relativo, zc_relativo, d_linear, a, b, c, zc2, z_trunk_real


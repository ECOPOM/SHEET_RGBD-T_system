## this script runs images alignment

import Thermal_sheet
import glob
import os
import INPUTS as c


# from csv files get the corrected average coordinates for the alignment
csv_file1 = c.ALIGNMENT_COORDS
coords_avg = Thermal_sheet.get_calib_x_y_min(csv_file1)
print(coords_avg)

# raw thermal images fns
img_path_term = c.RAW_THERM_PATH
# get all rgb images fns in the directory
images_col = glob.glob(c.RGB_PATH + '\\*.' + c.IMAGE_FORMAT) # glob.glob(col_path)

print('ALIGNMENT STARTED')

count = 0
for i in images_col:
    print(i)

    # find related image
    img_therm, _, _, _ = Thermal_sheet.get_rel_therm_fns(i, img_path_term, ' ')
    print(img_therm)

    # align to 1280
    rgb_path, termica = Thermal_sheet.alignment(coords_avg, i, img_therm)
    # align to 1920
    dir_img, fn = Thermal_sheet.get_dir_fn(img_therm)
    therm_fn = os.path.join(dir_img,(fn+'_aligned.PNG'))
    # aliognment function
    Thermal_sheet.align_1920(termica, csv_file1, therm_fn)

    count += 1


import glob
from Thermal_sheet import *
import INPUTS as c

# import the produced dataset
df = pd.read_csv(c.OUTPUT_DF_PATH, sep=" ", names=['name', 'label', 'Tmin_SEEK', 'Tmed_SEEK', 'Tmax_SEEK'])

count = 0
for image in glob.glob(c.RGB_PATH + '\\*.' + c.IMAGE_FORMAT):
    # find related depth map
    _, fn, depth, _ = get_rel_therm_fns(image,therm_path=' ', depth_path=c.DEPTH_PATH)

    # find related fruit labels
    fruit_label = c.FRUIT_DETECT_YOLO_LABELS_PATH+'\\'+fn+'.txt'

    # find related trunk labels
    trunk_label = c.TRUNK_DETECT_YOLO_LABELS_PATH + '\\' + fn + '.txt'

    # find related trunk coordinates
    print(trunk_label,'\n')
    count += 1





# specify the specie ['apple' or 'grape']
SPECIE = 'apple'

# pictures where taken from EAST (e) or WEAST (w)?
IMAGE_CARDINAL_FACE = 'e'

# shooting distance between RGBD/T cameras and trees
TRUNK_DIST = 2.8  # m

# from csv files get the corrected average coordinates for the cameras alignment
ALIGNMENT_COORDS = r"C:\Users\mirko.piani2\OneDrive - Alma Mater Studiorum Università di Bologna\Documents\SHEET\SHEET_repository\CAMERAS_ALIGNMENT\calibration_coord_tentativo2.txt"

# field COLOUR images
IMAGE_FORMAT = 'PNG'
RGB_PATH = r"C:\Users\mirko.piani2\OneDrive - Alma Mater Studiorum Università di Bologna\Documents\SHEET\APPLE_reduced_dataset\colore_fuji_gala"

# PATH TO DEPTH IMAGES
DEPTH_PATH = r"C:\Users\mirko.piani2\OneDrive - Alma Mater Studiorum Università di Bologna\Documents\SHEET\DATASET\APPLE_reduced_dataset\depth_fuji_gala"

# PATH SEEK RAw Allineate
RAW_THERM_PATH = r"C:\Users\mirko.piani2\OneDrive - Alma Mater Studiorum Università di Bologna\Documents\SHEET\DATASET\APPLE_reduced_dataset\termico_fuji_gala"

# path to yolo detected fruits labels
FRUIT_DETECT_YOLO_LABELS_PATH = r"C:\Users\mirko.piani2\OneDrive - Alma Mater Studiorum Università di Bologna\Documents\SHEET\DATASET\APPLE_reduced_dataset\fruit_detection_yolo_format_MP4_model\labels"

# path to yolo detected trunk labels
TRUNK_DETECT_YOLO_LABELS_PATH = r"C:\Users\mirko.piani2\OneDrive - Alma Mater Studiorum Università di Bologna\Documents\SHEET\DATASET\APPLE_reduced_dataset\trunk detection_yolo_format\labels\train"

# final database containing desired information
OUTPUT_DF_PATH = r'C:\Users\mirko.piani2\OneDrive - Alma Mater Studiorum Università di Bologna\Documents\SHEET\DATASET\APPLE_reduced_dataset\output_dataset.csv'
DELIVERABLE_OUTPUT_DF_PATH = r'C:\Users\mirko.piani2\OneDrive - Alma Mater Studiorum Università di Bologna\Documents\SHEET\DATASET\APPLE_reduced_dataset\DELIVERABLE_output_dataset.csv'

# calibration file containing the thermal loss with increasing distance
TH_CALIBRATION = r"C:\Users\mirko.piani2\OneDrive - Alma Mater Studiorum Università di Bologna\Documents\SHEET\SHEET_repository\datasets\distances_effect_BOTTLES_degrees_celsius.txt"



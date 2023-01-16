import INPUTS as c
import pandas as pd


df = pd.read_csv(c.OUTPUT_DF_PATH, sep=" ")

# get columns names
col = df.columns
print(df.columns)

# switch Y with Z in columns names
col2 = ['name', 'label', 'ID_tree', 'sec1', 'Tmin_SEEK', 'Tmean_SEEK',
       'Tmax_SEEK', 'sec2', 'Tmin_SEEK_corr_avg', 'Tmean_SEEK_corr_avg',
       'Tmax_SEEK_corr_avg', 'Tmin_SEEK_corr_max', 'Tmean_SEEK_corr_max',
       'Tmax_SEEK_corr_max', 'sec3', 'X_mm', 'Z_mm', 'Z_cam_mm', 'Y_mm',
       'X_trunk_mm', 'Z_trunk_mm', 'Z_trunk_cam_mm', 'Y_trunk_mm', 'sec4',
       'X_relativo', 'Z_relativo', 'Y_relativo', 'X_trunk_rel', 'Z_trunk_rel',
       'Y_trunk_rel', 'sec5', 'X_rel_range_mm', 'Z_rel_range_mm',
       'Y_rel_range_mm', 'X_trunk_rel_mm', 'Z_trunk_rel_mm', 'Y_trunk_rel_mm',
       'sec6', 'Tree_wall', 'Card_dir', 'Estim_Fruit_diam_mm']

# positioning Y coords after X and before Z
col3 = ['name', 'label', 'ID_tree', 'sec1', 'Tmin_SEEK', 'Tmean_SEEK',
       'Tmax_SEEK', 'sec2', 'Tmin_SEEK_corr_avg', 'Tmean_SEEK_corr_avg',
       'Tmax_SEEK_corr_avg', 'Tmin_SEEK_corr_max', 'Tmean_SEEK_corr_max',
       'Tmax_SEEK_corr_max', 'sec3', 'X_mm', 'Y_mm', 'Z_mm', 'Z_cam_mm',
       'X_trunk_mm', 'Y_trunk_mm', 'Z_trunk_mm', 'Z_trunk_cam_mm',  'sec4',
       'X_relativo', 'Y_relativo', 'Z_relativo', 'X_trunk_rel', 'Y_trunk_rel', 'Z_trunk_rel',
        'sec5', 'X_rel_range_mm', 'Y_rel_range_mm', 'Z_rel_range_mm',
        'X_trunk_rel_mm', 'Y_trunk_rel_mm', 'Z_trunk_rel_mm',
       'sec6', 'Tree_wall', 'Card_dir', 'Estim_Fruit_diam_mm']

# intermediate df: switch Y with Z in columns names
df2 = pd.DataFrame(columns= col2)

count = 0
for item in df2.columns:
    df2[item] = df[col[count]]
    count +=1

df3 = df2[col3]

df3.to_csv(c.DELIVERABLE_OUTPUT_DF_PATH, sep=" ")

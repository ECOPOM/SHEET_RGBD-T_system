
import numpy as np
import pandas as pd
import INPUTS as c


# import needed data
THERMAL_CORRECTION_FILE = c.TH_CALIBRATION

df = pd.read_csv(THERMAL_CORRECTION_FILE, sep= " ")
df['Raw min error'] = df.iloc[0,1] - df['Raw min']
df['Raw max error'] = df.iloc[0,2] - df['Raw max']
df['Raw avg error'] = df.iloc[0,3] - df['Raw avg']
df['C min error'] = df.iloc[0,4] - df['C min']
df['C max error'] = df.iloc[0,5] - df['C max']
df['C avg error'] = df.iloc[0,6] - df['C avg']

# linear regression between raw data error and distance to get a linear conversion model
raw_avg_coef_err = np.polyfit(df["dist"], df["Raw avg error"], 1) # x values, y values, function polynomial degree
raw_min_coef_err = np.polyfit(df["dist"], df["Raw min error"], 1) # x values, y values, function polynomial degree
raw_max_coef_err = np.polyfit(df["dist"], df["Raw max error"], 1) # x values, y values, function polynomial degree

# regression functions for raw data error (y variable) according to distance (x variable)
raw_poly1d_avg_err = np.poly1d(raw_avg_coef_err) # <-- it is a function not a vector
raw_poly1d_min_err = np.poly1d(raw_min_coef_err)
raw_poly1d_max_err = np.poly1d(raw_max_coef_err)

# regression functions for celsius data error (y variable) according to distance (x variable)
cels_avg_coef_err = np.polyfit(df["dist"], df["C avg error"], 1) # x values, y values, function polynomial degree
cels_min_coef_err = np.polyfit(df["dist"], df["C min error"], 1) # x values, y values, function polynomial degree
cels_max_coef_err = np.polyfit(df["dist"], df["C max error"], 1) # x values, y values, function polynomial degree

# regression functions for celsius data error (y variable) according to distance (x variable)
poly1d_avg_err = np.poly1d(cels_avg_coef_err) # <-- it is a function not a vector
poly1d_min_err = np.poly1d(cels_min_coef_err)
poly1d_max_err = np.poly1d(cels_max_coef_err)


print(poly1d_max_err)
print(poly1d_min_err)
print(poly1d_avg_err, "\n")

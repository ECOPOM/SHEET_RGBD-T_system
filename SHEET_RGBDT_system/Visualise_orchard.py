import matplotlib.pyplot as plt
import pandas as pd
import INPUTS as c

grape_image_fn_to_filter_data_for = '2022-08-03-12-06-54_img_0001'
apple_image_fn_to_filter_data_for = 'gala_2022-08-04-10-43-57_img_01'

df = pd.read_csv(c.OUTPUT_DF_PATH, sep= ' ')

fig = plt.figure(figsize=(15,9))  # Square figure
ax = fig.add_subplot( projection='3d')

x = df.X_mm
y = df.Y_mm
z = df.Z_mm

size= df.Estim_Fruit_diam_mm
t = df['Tmax_SEEK_corr_max']
names = df.name.unique().tolist()

x_trunk = df.X_trunk_mm
y_trunk = df.Y_trunk_mm
x_trunk_ticks = df.X_trunk_mm.unique()
z_trunk = df.Z_trunk_mm


if c.SPECIE.lower() == 'grape':
    p = ax.scatter3D(x,y,z, marker='v', s=size, c=t, cmap='jet', label='fruits')
elif c.SPECIE.lower() == 'apple':
    p = ax.scatter3D(x, y, z, marker='o', s=size, c=t, cmap='jet', label='fruits')
ax.scatter3D(x_trunk,y_trunk,z_trunk, marker='1', s=1000, c='k', label='trunk')

ax.set_xlabel('X: ${obj_x \ [millimeters]}$')
ax.set_ylabel('Z: ${obj_{dist} \ from\ the\ trunk\ [mm]}$')
ax.set_zlabel('Y: ${obj_y \ [millimeters]}$')
# Hide grid lines
ax.grid(False)
# Hide axes ticks
ax.set_xticks(ticks=x_trunk_ticks, fontsize=7)
ax.yaxis.labelpad = 20  # moves the y-axis label away from the axe line
ax.xaxis.labelpad = 20
ax.zaxis.labelpad = 5
fig.colorbar(p, ax=ax, label='Celsius degrees', anchor=(1,1.0))
ax.legend(loc='upper left', ncol=2)
# adjust plot stretching
# x,y,z - orchard 20,6,5  - tree 2,2,5
ax.set_box_aspect(aspect = (20,6,5))

plt.show()


#### PLOT RELATIVE

if c.SPECIE.lower() == 'grape':
    df=df.loc[(df['name'] == grape_image_fn_to_filter_data_for)]
elif c.SPECIE.lower() == 'apple':
    df=df.loc[(df['name'] == apple_image_fn_to_filter_data_for)]

fig = plt.figure(figsize=(15,9))  # Square figure
ax = fig.add_subplot( projection='3d')

x = df.X_relativo
y = df.Y_relativo
z = df.Z_relativo

size= df.Estim_Fruit_diam_mm
t = df['Tmax_SEEK_corr_max']

if c.SPECIE.lower() == 'grape':
    p = ax.scatter3D(x,y,z, marker='v', s=size, c=t, cmap='jet', label='fruits')
elif c.SPECIE.lower() == 'apple':
    p = ax.scatter3D(x, y, z, marker='o', s=size, c=t, cmap='jet', label='fruits')
ax.scatter3D(0,0,0, marker='1', s=1000, c='k', label='trunk')

ax.set_xlabel('${obj_x \ [relative]}$')
ax.set_ylabel('${obj_z \ [relative]}$')
ax.set_zlabel('${obj_y \ [relative]}$')
# Hide grid lines
fig.colorbar(p, ax=ax, label='Celsius degrees', anchor=(1,1.0))
ax.legend(loc='upper left', ncol=2)
# adjust plot stretching
if c.SPECIE.lower() == 'grape':
    ax.set_box_aspect(aspect = (4,5,3))
elif c.SPECIE.lower() == 'apple':
    ax.set_box_aspect(aspect = (2,2,5))

plt.show()



#### PLOT RELATIVE in mm
fig = plt.figure(figsize=(15,9))  # Square figure
ax = fig.add_subplot( projection='3d')

x = df.X_rel_range_mm
y = df.Y_rel_range_mm
z = df.Z_rel_range_mm


size= df.Estim_Fruit_diam_mm
t = df['Tmax_SEEK_corr_max']

if c.SPECIE.lower() == 'grape':
    p = ax.scatter3D(x,y,z, marker='v', s=size, c=t, cmap='jet', label='fruits')
    ax.set_xlabel('X relative (mm)')
    ax.set_ylabel('Z relative (mm)')
    ax.set_zlabel('Y relative (mm)')

elif c.SPECIE.lower() == 'apple':
    p = ax.scatter3D(x, y, z, marker='o', s=size, c=t, cmap='jet', label='fruits')
    ax.set_xlabel('Z relative (mm)')
    ax.set_ylabel('X relative (mm)')
    ax.set_zlabel('Y relative (mm)')

ax.scatter3D(0,0,0, marker='1', s=1000, c='k', label='trunk')

# Hide grid lines
fig.colorbar(p, ax=ax, label='Celsius degrees', anchor=(1,1.0))
ax.legend(loc='upper left', ncol=2)
# adjust plot stretching
if c.SPECIE.lower() == 'grape':
    ax.set_box_aspect(aspect = (4,5,3))
elif c.SPECIE.lower() == 'apple':
    ax.set_box_aspect(aspect = (2,2,5))

plt.show()
import numpy as np
import seaborn as sb
import matplotlib


color_undist = np.array([0,105,170])/255
color_dist = np.array([200,80,60])/255

# for yaw, pitch and roll
colors_ypr = [ np.array([210,150,0])/255,
               np.array([80,170,200])/255,
               np.array([175,110,150])/255]

# for undistorted, add 1 add 3
color_add1 = np.array([50,110,30])/255
color_add3 = np.array([200,80,60])/255

colors_add1 = sb.blend_palette(['#214017','#326e1e','#4ad41c'],13)
colors_add3 = sb.blend_palette(['#994738','#ee5c43','#ed8e7d'],13)

color_dynobs = np.array([130,185,160])/255
color_statobs = np.array([215,180,105])/255


font = {'family': 'sans-serif',
        'sans-serif': ['Helvetica', 'Fira Sans', 'Liberation Sans', 'DejaVu Sans'],
        'weight': 'normal',
        'size': 8}

def set_style():
    matplotlib.rc('font', **font)
    matplotlib.rc('axes.spines', right=False)
    matplotlib.rc('axes.spines', top=False)
    matplotlib.rc('axes', edgecolor='black')
    matplotlib.rc('xtick', color='black', bottom=True)
    matplotlib.rc('ytick', color='black', left=True)
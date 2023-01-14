# Import libraries
import os
import pandas as pd
import numpy as np
pd.options.display.max_rows = 500
import math
import quaternion
from scripts import *
# for tools.py functions
from scipy.interpolate import interp1d
from math import sqrt
from read import *
import matplotlib.pyplot as plt
from tools import *

def find_timebounds(arr,index):
    h, b = arr.shape
    avg = 0.0
    min = 1000.0 
    max = -1.0 
    for i in range(b):
        if (i == index):
            for j in range(h):
                avg += arr[j,i]
                if (arr[j,i] < min):
                    min = arr[j,i]
                elif (arr[j,i] > max):
                    max = arr[j,i]
    avg = avg/h
    min_margin = avg - min
    max_margin = max - min
    print("min --- avg --- max")    
    print(min,"---",avg,"---",max)
    return min_margin, avg, max_margin

def find_delays_all_modules():
    # finding out the files/folders in the current directory
    for directory in os.listdir(os.getcwd()):  
        if ('hz' in directory):
            file_name = directory + '/traj_timing.txt'
            df = pd.read_csv(file_name, delimiter=',')
            np_table = df.to_numpy()
            print("For the frequency of ",directory)
            # tracking_min, tracking_avg, tracking_ = find_timebounds(np_table,1)
            for i in range(1,8):
                print("For ",df.columns[i]," module") 
                find_timebounds(np_table,i)
                print("\n")

def find_distance_error(gt_np,np_table):
    # gt_np -> GT(groundtruth)
    # np_table -> state_estimate
    gt_h, ref_b = gt_np.shape
    np_table_h, np_table_b = np_table.shape
    gt_tx, gt_ty, gt_tz = gt_np[gt_h-1,1], gt_np[gt_h-1,2], gt_np[gt_h-1,3]
    switch = 1 # 1 -> for better calculation
    if (switch == 0):
        # print("Executing bad pose error")        
        table_tx, table_ty, table_tz = np_table[np_table_h-1,1], np_table[np_table_h-1,2], np_table[np_table_h-1,3]
        error_euclidean_dist = math.sqrt(np.square(abs(table_tx - gt_tx)) + np.square(abs(table_ty - gt_ty)) + np.square(abs(table_tz - gt_tz)))
        print("The deviation in VIO path and the groundtruth is:-", error_euclidean_dist)
    else:
        # print("Executing better pose error")
        gt_timestamp = int(gt_np[gt_h-1,0]) # truncate the decimal values 
        index = 0
        for i in reversed(range(np_table_h)) :
            np_timestamp = int(np_table[np_table_h-1,0])
            index = i 
            if (gt_timestamp == np_timestamp):
                break
        table_tx, table_ty, table_tz = np_table[index,1], np_table[index,2], np_table[index,3]
        error_euclidean_dist = math.sqrt(np.square(abs(table_tx - gt_tx)) + np.square(abs(table_ty - gt_ty)) + np.square(abs(table_tz - gt_tz)))
        print("The deviation in VIO path and the groundtruth is:-", error_euclidean_dist)
          

def find_rotational_error(ref_np,np_table):
    q1 = np.quaternion(ref_np[0,8],ref_np[0,5],ref_np[0,6],ref_np[0,7])
    q2 = np.quaternion(np_table[0,8],np_table[0,5],np_table[0,6],np_table[0,7])
    print("quaternion")
    print(q2.w, q2.x, q2.y, q2.z)
    # print(np_table[0,5])

def pose_error(file):
    # finding out the files/folders in the current directory
    for directory in os.listdir(os.getcwd()): 
        ref_file = file
        ref_df = pd.read_csv(ref_file, delimiter=' ')
        ref_np = ref_df.to_numpy()
        if ('hz' in directory):
            file_name = directory + '/traj_estimate.txt'    
            df = pd.read_csv(file_name, delimiter=' ')
            np_table = df.to_numpy()
            print("For the frequency of ",directory)
            find_distance_error(ref_np,np_table)
            # find_rotational_error(ref_np,np_table)

# from math import sqrt
def rms(a, b):
    tmp = a - b
    rms = sqrt(np.sum(tmp*tmp)/a.shape[0])
    
    tmp = np.sqrt(np.sum(tmp*tmp, axis=1))
    mean = np.mean(tmp)
    std = np.std(tmp)
    
    return mean, std

def ate(gt_file,euroc_file):
    gt = read_3d(gt_file)
    euroc_3d = read_3d(euroc_file)

    gt, euroc_3d, euroc_3d_rms = analyze(gt, euroc_3d)

    euroc_3d, s = scale(gt, euroc_3d)
    # slam2 = slam2 / s

    # slam2_mean, slam2_std = rms(gt[:,1:], slam2[:,1:])
    euroc_3d_mean, euroc_3d_std = rms(gt[:,1:], euroc_3d[:,1:])  
    euroc_rms_error = [ [euroc_3d_rms] ]

    euroc_rms_error = np.asarray(euroc_rms_error, dtype=np.float64)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.yaxis.grid(True)

    error_config = {'ecolor': 'r', 'capsize': 5, 'elinewidth': 3}

    opacity = 0.6
    ax.set_ylabel('Mean absolute trajectory error')
    #ax.set_title('Mean Absolute Trajectory Error for ORB-SLAM2 and Edge-SLAM')

    line = plt.bar([0], [euroc_3d_mean], 1, alpha=opacity, yerr=(euroc_3d_std), error_kw=error_config, color='b')

    ax.set_xlabel('SLAM System')
    # plt.xticks([0], ('OPEN_VINS'))
    fig.tight_layout()

    return euroc_rms_error, plt


######################## DRONESLAB FUNCTIONS ########################


# # from plot.py

# def plot_trajectories_2d(title, *trajectories):
#     import matplotlib.pyplot as plt
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_title(title)

#     for ii, trajectory in enumerate(trajectories):
#         xs, ys = trajectory[:,1].tolist(), trajectory[:,2].tolist()
#         line, = ax.plot(xs, ys, label=str(ii+1))
#         ax.legend()

#     plt.show()

# # from read.py
# def read_gt(filename):
#     result = []
#     with open(filename, 'r') as f:
#         lines = f.readlines()
#         for line in lines[1:]:
#             words = line.split(' ')
#             result.append([ float(words[1].strip()), float(words[2].strip()), float(words[6].strip()), 0.0 ])
#     return np.asarray(result, dtype=np.float64)
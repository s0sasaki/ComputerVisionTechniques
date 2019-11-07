import numpy as np

dataset = 21
giffilename = 'map_test.gif'
tmap_giffilename = 'tmap_test.gif'
#tstart = 1298881994
tstart = 1298881990
#tend   = 1298882114
tend   = 1298882115
#1298881994.957439
#1298881994.982633
#1298881995.438197
#1298881998.282973
#1298881998.441944
#1298882114.36065 
#1298882114.350636
#1298882112.795965
#1298882111.044915
#1298882111.030917
  
with np.load("Encoders%d.npz"%dataset) as data:
    encoder_counts = data["counts"] # 4 x n encoder counts
    encoder_stamps = data["time_stamps"] # encoder time stamps

with np.load("Hokuyo%d.npz"%dataset) as data:
    lidar_angle_min = data["angle_min"] # start angle of the scan [rad]
    lidar_angle_max = data["angle_max"] # end angle of the scan [rad]
    lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad]
    lidar_range_min = data["range_min"] # minimum range value [m]
    lidar_range_max = data["range_max"] # maximum range value [m]
    lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
    lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans
    
with np.load("Imu%d.npz"%dataset) as data:
    imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
    imu_linear_acceleration = data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling)
    imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements
  
with np.load("Kinect%d.npz"%dataset) as data:
    disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
    rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images

if __name__ == '__main__':
    print(encoder_stamps[0])             #1298881994.957439
    print(lidar_stamps[0])               #1298881994.982633
    print(imu_stamps[0])                 #1298881995.438197
    print(disp_stamps[0])                #1298881998.282973
    print(rgb_stamps[0])                 #1298881998.441944
    print(encoder_stamps[-1])            #1298882114.36065 
    print(lidar_stamps[-1])              #1298882114.350636
    print(imu_stamps[-1])                #1298882112.795965
    print(disp_stamps[-1])               #1298882111.044915
    print(rgb_stamps[-1])                #1298882111.030917











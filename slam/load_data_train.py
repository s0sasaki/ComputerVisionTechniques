import numpy as np

dataset = 20
giffilename = 'map_train.gif'
tmap_giffilename = 'tmap_train.gif'
tstart = 1298445270
tend = 1298445400
  
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
    print(encoder_counts.shape)             #(4, 4956)
    print(encoder_stamps.shape)             #(4956,)
    print(lidar_angle_min)                  #-2.356194490192345
    print(lidar_angle_max)                  #2.356194490192345
    print(lidar_angle_increment.shape)      #(1, 1)
    print(lidar_range_min)                  #0.1
    print(lidar_range_max)                  #30
    print(lidar_ranges.shape)               #(1081, 4962)
    print(lidar_stamps.shape)               #(4962,)
    print(imu_angular_velocity.shape)       #(3, 12187)
    print(imu_linear_acceleration.shape)    #(3, 12187)
    print(imu_stamps.shape)                 #(12187,)
    print(disp_stamps.shape)                #(2407,)
    print(rgb_stamps.shape)                 #(2289,)

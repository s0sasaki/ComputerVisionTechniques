import numpy as np

dataset = 23
giffilename = 'map_test2.gif'
tmap_giffilename = 'tmap_test2.gif'

tstart = 1299175880
tend   = 1299175979 
#1299175883.167897
#1299175883.177606
#1299175882.174842
#1299175978.130368
#1299175978.126778
#1299175976.379891

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
  
#with np.load("Kinect%d.npz"%dataset) as data:
#    disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
#    rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images

if __name__ == '__main__':
    print(encoder_stamps[0])     #1299175883.167897
    print(lidar_stamps[0])       #1299175883.177606
    print(imu_stamps[0])         #1299175882.174842
    #print(disp_stamps[0])       
    #print(rgb_stamps[0])        
    print(encoder_stamps[-1])    #1299175978.130368
    print(lidar_stamps[-1])      #1299175978.126778
    print(imu_stamps[-1])        #1299175976.379891
    #print(disp_stamps[-1])      
    #print(rgb_stamps[-1])       











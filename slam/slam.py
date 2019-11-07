import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import map_utils

## Parameters

#mode = "train"
#mode = "test"
mode = "test2"
if mode == "train":
    import load_data_train as data
elif mode == "test":
    import load_data_test as data
elif mode == "test2":
    import load_data_test2 as data
output_data = 'map'
#output_data = 'tmap'
dt = 0.1
pixsize = 300
physicalsize = 40
nparticle = 10
n_thresh = 9
predict_sigma_position = 0.00001
predict_sigma_angle = 0.005
update_sigma_position = 0.00001
update_sigma_angle = 0.001
update_n_dposition = 2
update_n_dangle = 4
logodds_thresh = 100
lidar_max = 8
lidar_min = 0.1

def mapping_particle(logodds, z, x,y,theta):
    wTb = np.matrix([[np.cos(theta),-np.sin(theta), x],
                     [np.sin(theta), np.cos(theta), y],
                     [            0,             1, 0]])
    zb = np.matrix((z*np.cos(angles), z*np.sin(angles), np.ones(angles.shape)))
    zw = wTb*zb
    x  =  x/MAP['res'] + MAP['sizex']/2
    y  =  y/MAP['res'] + MAP['sizey']/2
    zw[0] = zw[0]/MAP['res'] + MAP['sizex']/2
    zw[1] = zw[1]/MAP['res'] + MAP['sizey']/2
    for i in range(len(angles)):
        x1, y1 = map_utils.bresenham2D(x, y, zw[0,i], zw[1,i])
        for j,k in zip(x1[:-1],y1[:-1]):
            logodds[int(j), int(k)] -= 2 
        logodds[int(x1[-1]), int(y1[-1])] += 2  #If this fails, enlarge the map size. (physicalsize)
    np.clip(logodds, -logodds_thresh, logodds_thresh, out=logodds)
    return logodds

def mapping(mus, logoddss, maps, z):
    for i in range(len(mus)):
        x,y,theta = mus[i]
        logoddss[i] = mapping_particle(logoddss[i], z, x, y, theta)
        maps[i] = 1-1/(1+np.exp(logoddss[i]))
        maps[i][maps[i]<=0.5] = 0 #binary
        maps[i][maps[i]>0.5] = 1
    return logoddss, maps

def predict(mus, i):     
    vr = (encoder_counts[0, i]+encoder_counts[2,i])/2 * 0.0022
    vl = (encoder_counts[1, i]+encoder_counts[3,i])/2 * 0.0022
    v = (vr+vl)/2
    omega = imu_angular_velocity[2,i] 
    mus[:,0] += v*np.cos(mus[:,2]) + np.random.normal(0,predict_sigma_position,len(mus))
    mus[:,1] += v*np.sin(mus[:,2]) + np.random.normal(0,predict_sigma_position,len(mus))
    mus[:,2] += omega*dt + np.random.normal(0,predict_sigma_angle,len(mus))
    return mus

def update(mus, alphas, z, maps):
    map_corr = np.zeros(len(mus))
    for i in range(len(mus)):
        x_old,y_old,theta_old = mus[i]
        d_position = 2*update_sigma_position/update_n_dposition
        d_angle    = 2*update_sigma_angle/update_n_dangle
        x_range = np.arange(-update_sigma_position, update_sigma_position+d_position, d_position)
        y_range = np.arange(-update_sigma_position, update_sigma_position+d_position, d_position)
        theta_range = np.arange(-update_sigma_angle,update_sigma_angle   +d_angle,    d_angle) + theta_old
        for theta in theta_range:
            wTb = np.matrix([[ np.cos(theta),-np.sin(theta), x_old],
                             [ np.sin(theta), np.cos(theta), y_old],
                             [            0,              0,     1]])
            xs0w = wTb*np.vstack((z*np.cos(angles), z*np.sin(angles), np.ones(angles.shape)))
            Y = xs0w[:2]
            x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-positions of each pixel of the map
            y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-positions of each pixel of the map
            c = map_utils.mapCorrelation(maps[i],x_im,y_im,Y,x_range,y_range)
            if map_corr[i] < c.max():
                j,k = np.unravel_index(c.argmax(), c.shape)
                mus[i][0] = x_range[j] + x_old
                mus[i][1] = y_range[k] + y_old
                mus[i][2] = theta
                map_corr[i] = float(np.max(c))
    alphas *= np.exp(map_corr - map_corr.max())
    alphas /= alphas.sum()
    return alphas, mus

def resample(mus, alphas, n_thresh):
    n_eff = 1/np.linalg.norm(alphas)**2
    new_mus = np.zeros(mus.shape)
    if n_eff>n_thresh:
        return mus, alphas
    #print("resample:", n_eff)
    j = 0
    c = alphas[0]
    u = np.random.uniform(low=0,high=1/len(mus))
    for k in range(len(mus)):
        beta = u + k/len(mus)
        while beta>c:
            j = j+1
            c = c+alphas[j]
        new_mus[k] = mus[j]
    return new_mus, np.array([1/nparticle for _ in range(nparticle)])

def plotting(trajectories, maps):
    avgmap = np.zeros(maps[0].shape)
    for i in range(len(mus)):
        x,y = trajectories[i]
        x = x/MAP['res'] + MAP['sizex']/2
        y = y/MAP['res'] + MAP['sizey']/2
        for p,q in zip(x,y):
            maps[i][int(p-1):int(p+1),int(q-1):int(q+1)] = 0.3
        avgmap += maps[i]/len(mus)
    plotmap = plt.imshow(1-avgmap, cmap='gray',  animated=True)
    ims.append([plotmap])

from scipy.misc import imread, imresize
def texturemapping(tmap, mus, t):
    time = tstart + t*dt
    target_disp_stamp = np.median(data.disp_stamps[np.logical_and(time<=data.disp_stamps,data.disp_stamps<time+dt)])
    target_disp_index = np.where(data.disp_stamps > target_disp_stamp-0.000001)[0][0]
    filename_disp = "dataRGBD/Disparity"+str(data.dataset)+"/disparity"+str(data.dataset)+"_"+str(target_disp_index+1)+".png"
    target_rgb_stamp = np.median(data.rgb_stamps[np.logical_and(time<=data.rgb_stamps,data.rgb_stamps<time+dt)])
    target_rgb_index = np.where(data.rgb_stamps > target_rgb_stamp-0.000001)[0][0]
    filename_rgb = "dataRGBD/RGB"+str(data.dataset)+"/rgb"+str(data.dataset)+"_"+str(target_rgb_index+1)+".png"
    rgb_img = imread(filename_rgb)
    disp_img = imread(filename_disp)
    Pcam=np.matrix([[0.18,0.005,0.36]]).T
    roll=0     #[rad]
    pitch=0.36 #[rad]
    yaw=0.021  #[rad]
    fsu=585.05108211
    fsv=585.05108211
    fstheta=0
    cu=242.94140713
    cv=315.83800193
    K = np.matrix([[fsu, fstheta, cu],
                   [  0,     fsv, cv],
                   [  0,       0,  1]])
    Kinv = np.linalg.inv(K)
    Roc = np.matrix([[0,-1, 0],
                     [0, 0,-1],
                     [1, 0, 0]])
    xw, yw, thetaw = mus.mean(axis=0)
    wPb = np.matrix([[xw,yw,0]]).T
    wRb = np.matrix([[ np.cos(thetaw),-np.sin(thetaw), 0],
                     [ np.sin(thetaw), np.cos(thetaw), 0],
                     [            0,                0, 1]])
    d = disp_img
    dd = -0.00304*d+3.31
    depth = 1.03/dd
    i = np.arange(d.shape[0])
    j = np.arange(d.shape[1])
    rgbi = (i*526.37 + dd.T*(-4.5*1750.46) + 19276.0)/585.051
    rgbj = (j*526.37 + 16662.0)/585.051
    for k in range(d.shape[0]):
        u = np.int32(rgbi[:,k])
        v = np.int32(rgbj)
        validindex1 = np.logical_and(0<=u,u<rgb_img.shape[0])
        validindex1 = np.logical_and(validindex1, 0<=v,v<rgb_img.shape[1])
        u = u[validindex1]
        v = v[validindex1]
        rgb_pos = np.vstack((u,v,np.ones(u.shape)))
        zo = depth[k, validindex1]
        X0 = zo * np.array(Kinv * rgb_pos)
        X0 = np.matrix(X0)
        Xw = wRb * Roc.T * X0 + wPb
        Xw = Xw + Pcam
        x_tmap = np.int32(Xw[0]/MAP['res'] + MAP['sizex']/2)
        y_tmap = np.int32(Xw[1]/MAP['res'] + MAP['sizey']/2)
        validindex = Xw[2] < 0.5
        validindex = np.logical_and(validindex, x_tmap>=0, x_tmap<pixsize)
        validindex = np.logical_and(validindex, y_tmap>=0, y_tmap<pixsize)
        x_tmap = x_tmap[validindex]
        y_tmap = y_tmap[validindex]
        validindex = np.array(validindex)[0]
        u = u[validindex]
        v = v[validindex]
        tmap[x_tmap, y_tmap] = rgb_img[u,v]/255
    plotmap = plt.imshow(tmap, animated=True)
    ims_tmap.append([plotmap])

def init_data():
    lidar_ranges = np.zeros((1081, nsteps))
    encoder_counts = np.zeros((4, nsteps))
    imu_angular_velocity = np.zeros((3, nsteps))
    for i in range(0, nsteps):
        t = tstart+i*dt
        lidar_ranges[:,i]         = data.lidar_ranges[:,         np.logical_and(t<=data.lidar_stamps,   data.lidar_stamps<t+dt)].mean(axis=1)
        encoder_counts[:,i]       = data.encoder_counts[:,       np.logical_and(t<=data.encoder_stamps, data.encoder_stamps<t+dt)].sum(axis=1)
        imu_angular_velocity[:,i] = data.imu_angular_velocity[:, np.logical_and(t<=data.imu_stamps,     data.imu_stamps<t+dt)].mean(axis=1)
    return lidar_ranges, encoder_counts, imu_angular_velocity

fig = plt.figure()
ims = []
ims_tmap = []
tstart = data.tstart
tend = data.tend
nsteps = int((tend-tstart)/dt)
lidar_ranges, encoder_counts, imu_angular_velocity = init_data()
MAP = {}
MAP['res']   = 2*physicalsize/pixsize
MAP['xmin']  = -physicalsize  #meters
MAP['ymin']  = -physicalsize
MAP['xmax']  =  physicalsize
MAP['ymax']  =  physicalsize 
MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
mus = np.array([[0,0,np.pi*0] for _ in range(nparticle)],dtype='float64' )
alphas = np.array([1/nparticle for _ in range(nparticle)])
logoddss = [np.zeros((pixsize, pixsize)) for _ in range(len(mus))]
maps = [np.zeros((pixsize, pixsize)) for _ in range(len(mus))]
trajectories = [np.zeros((2,1400)) for _ in range(len(mus))]
tmap = np.zeros((pixsize, pixsize, 3))

#for t in range(int(nsteps*0.07),int(nsteps*0.95)): #ignore invalid data
#for t in range(int(nsteps*0.05),int(nsteps*0.95)): #ignore invalid data
for t in range(int(nsteps*0.04),int(nsteps*0.96)): #ignore invalid data
    z = lidar_ranges[:,t]
    indValid = np.logical_and((z < lidar_max),(z > lidar_min))
    z = z[indValid]
    angles = data.lidar_angle_min + float(data.lidar_angle_increment) * np.arange(1081)
    angles = angles[indValid]

    logodds, maps = mapping(mus, logoddss, maps, z)
    alphas, mus = update(mus, alphas, z, maps)
    mus = predict(mus, t)
    mus, alphas = resample(mus, alphas, n_thresh)
    for i in range(len(mus)):
        trajectories[i][:,t] = mus[i, :2]
    #if t%(int(nsteps/20)) == 0:
    if t%(int(nsteps/200)+1) == 0:
        print("time: ", t, " / ", nsteps)
        if output_data == 'map':
            plotting(trajectories,maps) # SLAM Trajectory GIF 
        elif output_data == 'tmap':
            texturemapping(tmap, mus, t) # Texture Map GIF

if output_data == 'map':
    ani = animation.ArtistAnimation(fig, ims, interval=30, blit=True, repeat_delay=1000)
    ani.save(data.giffilename)
elif output_data == 'tmap':
    ani = animation.ArtistAnimation(fig, ims_tmap, interval=30, blit=True, repeat_delay=1000)
    ani.save(data.tmap_giffilename)
plt.show()
plt.waitforbuttonpress()


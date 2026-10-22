# Reads imu.txt into stamps (N,), gyro (N,3) rad/s, accel (N,3) m/s^2, in the camera optical frame.

import numpy as np
   
def read_imu(path):

    imu = np.loadtxt(path)

    if imu.ndim !=2 or imu.shape[1]!=7:
        raise ValueError(f"{path}: Wrong number of columns")
    
    stamp = imu[:,0]
    gyro = imu[:,1:4]
    accel = imu[:,4:7]

    if not (np.all(np.diff(stamp)>0)):
        raise ValueError(f"{path}: IMU stamps are not strictly increasing")


    return stamp, gyro, accel

# Returns the IMU samples covering the time from two keyframe times
def imu_between(stamp, gyro, accel, t0, t1):

    # Guard clause: the IMU has to cover the whole interval, or we'd be missing motion
    if t0 < stamp[0] or t1 > stamp[-1] or t1 <= t0:
        raise ValueError(f"interval [{t0}, {t1}] is not inside the IMU data [{stamp[0]}, {stamp[-1]}]")

    # First sample: the last one with stamp <= t0 (insert after equal values, then step back one)
    i0 = np.searchsorted(stamp, t0, side="right") - 1

    # Last sample: the last one with stamp < t1 (strictly before)
    i1 = np.searchsorted(stamp, t1, side="left") - 1

    # When each used sample's reading starts counting: its own stamp...
    starts = stamp[i0:i1+1].copy()
    starts[0] = t0 

    # When each reading stops counting: the NEXT sample's stamp...
    ends = stamp[i0+1:i1+2].copy()
    ends[-1] = t1 

    dt = ends - starts

    gyro_seg = gyro[i0:i1+1,:]
    accel_seg = accel[i0:i1+1,:]

    return gyro_seg, accel_seg, dt


if __name__ == "__main__":
    stamps, gyro, accel = read_imu("data/run1/imu.txt")
    print(stamps.shape, gyro.shape, accel.shape)
    print(gyro[0], accel[0])

    # Load Key frame timestamps
    kf_stamps = np.loadtxt("data/run1/keyframes.txt")[:,2]
    sample_count =[]
    timing_error = []
    min_dt=[]


    for k in range (len(kf_stamps)-1):
        t0 = kf_stamps[k]
        t1 = kf_stamps[k+1]
        dt = imu_between(stamps,gyro,accel,t0,t1)[2]

        sample_count.append(len(dt))
        timing_error.append(abs(dt.sum()-(t1-t0)))
        min_dt.append(dt.min())

    print(np.median(sample_count),min(sample_count),max(sample_count))
    print(max(timing_error))
    print(min(min_dt))
    print(len(sample_count))

    # Toy test
    toy = np.array([0, 5, 10, 15, 20, 25, 30], dtype=float)
    g = np.zeros((7, 3))
    a = np.zeros((7, 3))
    gs, acs, dts = imu_between(toy, g, a, 7, 23)
    print(imu_between(toy, g, a, 7, 23)[2])    # expect [3. 5. 5. 3.]
    print(imu_between(toy, g, a, 10, 25)[2])   # expect [5. 5. 5.]
    print(gs.shape, acs.shape, dts.shape)      # expect (4, 3) (4, 3) (4,)



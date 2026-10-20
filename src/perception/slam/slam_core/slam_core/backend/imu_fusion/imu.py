import numpy as np
   
def read_imu(path):

    d = np.loadtxt(path)

    stamp = d[:,0]
    gyro = d[:,1:4]
    accel = d[:,4:7]

    if not (np.all(np.diff(stamp)>0)):
        raise ValueError(f"{path}: IMU stamps are not strictly increasing")


    return stamp, gyro, accel


if __name__ == "__main__":
    stamps, gyro, accel = read_imu("data/run1/imu.txt")
    print(stamps.shape, gyro.shape, accel.shape)
    print(gyro[0], accel[0])

from .imu import read_imu, imu_between
import numpy as np
import gtsam
from slam_core.types.pose import quat_to_rot, rotation_angle

def preintegrate(gyro_seg, accel_seg,dt, params, bias):
    pim = gtsam.PreintegratedImuMeasurements(params, bias)

    for n in range(len(dt)):
        pim.integrateMeasurement(accel_seg[n],gyro_seg[n], dt[n])
    
    return pim
    



if __name__ == "__main__":

    stamp, gyro, accel = read_imu("data/run1/imu.txt")
    keyframes = np.loadtxt("data/run1/keyframes.txt")
    params = gtsam.PreintegrationParams.MakeSharedU(9.81)
    bias = gtsam.imuBias.ConstantBias()
    preintegrate_error = []

    for k in range(len(keyframes)-1):
        gyro_seg, accel_seg, dt = imu_between(stamp, gyro, accel,keyframes[k,2],keyframes[k+1,2])
        pim = preintegrate(gyro_seg, accel_seg, dt,params,bias)

        R_imu = pim.deltaRij().matrix()
        R_vo = quat_to_rot(keyframes[k+1,6:10])
    
        err = np.degrees(rotation_angle(R_vo.T @ R_imu))

        preintegrate_error.append(err)

    print("preintegration max error",max(preintegrate_error))
    print("preintegration median error",np.median(preintegrate_error))




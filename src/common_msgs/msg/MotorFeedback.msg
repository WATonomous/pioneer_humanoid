int8 motor_id

# position: degrees in the MOTOR frame (apply arm_calibration.yaml's direction/zero_offset
# to get the command frame). velocity: servo mode -> ERPM, MIT mode -> the drive's velocity
# units (GL II: "r/s") scaled by mit_profiles.yaml.
float32 position
float32 velocity
float32 current

# N.m. GL II MIT: from the drive. AK: current * kt (mit_profiles.yaml). 0 otherwise.
float32 torque

int8 temperature

# Servo: FbkErrorCode (0 = no fault). GL II MIT: status nibble (0 Disable, 1 Enable, >= 8 fault).
int8 error_code

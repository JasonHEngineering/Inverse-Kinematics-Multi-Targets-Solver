## Inverse-Kinematics-Multi-Targets-Solver

### IK_Jacobian_multiple_target_CCD_Quat_V5.py

This is a python script that process a series of 3d targets (x, y, x) in a .csv file, and then proceed to solve for the joint angles base on the actual kinematics of my robotic arm.

The algorithm combines user choices of the start/end pose as well as the automatic switching between IK CCD (Inverse Kinematics Cyclic Coordinate Descent) to that of IK Jacobian Quaternions. Sample target points (.csv) are attached.


### Sample Target Coordinates (.csv)
100mm_half_circle_coordinates_V3.csv
line_target_2.csv
o_target_2.csv


### Use Cases
It is useful when there are cases of needing to solve multiple targets at one go where it is also fine to let the system work out the start and end pose on it's own.

Solution may not always be found, and it is entirely possible where solutions are found for all the targets (like above) or even none at all, at other time of the script run despite having the same targets.

Some care needs to be take when dealing at points near singularities - mainly tweaking parameters "step_size" as well as "MAX_STEP_RAD".


### Position Error Reduction - via IK Jacobian Quaternion
<img width="1559" height="853" alt="newplot" src="https://github.com/user-attachments/assets/fc9c63ed-d212-45ef-9c34-ea6573eda92c" />


### Quaternions Error Reduction - via IK Jacobian Quaternion
<img width="1559" height="853" alt="newplot(1)" src="https://github.com/user-attachments/assets/4729cecf-bb0b-4747-a554-0b4e047a0df3" />

### Joint Angles Updates - via IK Jacobian Quaternion
<img width="1559" height="853" alt="newplot(3)" src="https://github.com/user-attachments/assets/d42f3741-1d41-4fd1-8e4e-82ed1f06f7c5" />


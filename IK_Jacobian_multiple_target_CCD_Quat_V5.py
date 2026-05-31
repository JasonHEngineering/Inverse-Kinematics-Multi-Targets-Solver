
# -*- coding: utf-8 -*-
"""
V01 -   restart baseline with Proj Hayley V5, cleanup
        method utilizes random CCD tries on each of the targets in the full target list
        From the full table of tries of each CCD target, it will take the CCD solution where the angular RMSE was the lowest
        i.e. the angles of theta 1 to theta 5 where they are all closest to home position, ignore theta 6
        From this lowest angular RMSE, it sets the starting theta angles from which it is the basis to start the IK Jacobian Quat
        It is also the basis for the target pose for the IK Jacobian Quat
        after round 1 of IK Jacobian quat, there might be target points where solution was not reached
        it will then revisit the CCD table, filter out all those that was already solved with the Jacobian IK
        and then take the remaining unsolved targets (but CCD solved) for the next lowest angular RMSE to set the next theta baseline and target pose
        The newly solved solutions in second round is now appended
        The remaining unsolved ones may refer may still refer to CCD solutions
        
V02 -   Tryout a second method
        While still utilizing random CCD tries, this will try on the first and last targets only
        For each of the solved CCD in the first and last target, do a delta angular RMSE for each combination
        If there are 5x solved CCD in first target and 6x solved CCD in the last target, then there should be 5x6 = 30 permutations
        The frontrunner pair is deem as the one from the lowest RMSE
        The logic is that we want minimum amount of angular movements, that translate from first to the last pose
        Target orientation is that of the first target. Thus the last target have to adopt this orientation.
        Intermediate targets are presume to be near these angular values (minimun effort) 
        
V03 -   Tryout a third method
        Still focus CCD tries on the first and the last target
        For each of the solved CCD in the first and last target, do a delta angular RMSE for each combination
        If there are 5x solved CCD in first target and 6x solved CCD in the last target, then there should be 5x6 = 30 permutations
        The frontrunner pair is deem as the one from the lowest RMSE
        The logic is that we want minimum amount of angular movements, that translate from first to the last pose
        Orientation shall be SLERP interpolated from the first to the last accepted target <== main difference between V3 and V2
        Intermediate targets are presume to be near these angular values (minimun effort) 
        
V04 -   Tryout a fourth method
        Instead of minimum delta angluar RMS between start and end angles, which could be at awkard joint angles already
        So this will be like method 2, to get minimum angular change from zero position, however the start pose will transit to end pose via SLERP

V05 -   Cleanup

"""
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import random
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
from skspatial.objects import Plane, Point, Vector
import math
import plotly.io as pio
from math import (
    asin, pi, atan2, cos
)
from scipy.spatial.transform import Slerp, Rotation
from scipy.spatial.transform import Rotation as R
import time
from datetime import datetime

# file_path = 'line_target_2.csv'
file_path = 'o_target_2.csv'
# file_path = '100mm_half_circle_coordinates_V2.csv'

pose_method = "SLERP" # "SLERP" or "first_pose_locked_till_end"
IK_CCD_angular_method = "angular_RMSE_closest_to_each_other" #  "angular_RMSE_closest_to_each_other" or "angular_RMSE_closest_to_zero"

displacement_err = 0.3 # acceptable error for IK in mm, was 0.5
# experimental: open up orientation error in deg and let quaternion take its course
orientation_err = 360 # acceptable error for orientation in deg, was 45 deg
orientation_err_quat = 0.1 # acceptable error for orientation in quaternion, was 0.15

angular_RMSE =  math.inf
IK_CCD_tries = 10

hinge_length = 24 # for illustrating the hinge in 3d view
max_iter = 3000 # iterations allow for IK CCD
step_size=0.03 #step size for jacobian delta_theta. was 0.05

#real limits
lower_limit = [-100, -90, -90, -120, -100, -179]
upper_limit = [100, 90, 90, 120, 100, 179]

data = []
column_headers = ['Rows',
                  'Tgt. X', 'Tgt. Y', 'Tgt. Z',
                  'Act. X', 'Act. Y', 'Act. Z',
                  'Pos Error',
                  'Tgt. Roll', 'Tgt. Pitch', 'Tgt. Yaw',
                  'Act. Roll', 'Act. Pitch', 'Act. Yaw',
                  'Tgt. q1', 'Tgt. q2', 'Tgt. q3', 'Tgt. q4',
                  'Act. q1', 'Act. q2', 'Act. q3', 'Act. q4',
                  'Gimbal lock',
                  'IK Iterations',
                  'IK Timestamp', 'Solve Status',
                  'theta1', 'theta2', 'theta3', 'theta4', 'theta5', 'theta6',
                  'time_stay',
                  'Servo Status',
                  'Angular RMSE'
                  ]

interim_column_headers = ['Target',
                  'IK CCD Tries No.',
                  'Solve Status',
                  'IK Iterations',
                  'Tgt. X', 'Tgt. Y', 'Tgt. Z',
                  'Act. X', 'Act. Y', 'Act. Z',
                  'Pos Error',
                  'Act. Roll', 'Act. Pitch', 'Act. Yaw',
                  'Act. q1', 'Act. q2', 'Act. q3', 'Act. q4',
                  'theta1', 'theta2', 'theta3', 'theta4', 'theta5', 'theta6',
                  'Angular RMSE'
                  ]

start_time = time.time()

pio.renderers.default='browser'

figure_axis_limit = 600

global desired_orientation
global desired_orientation_quat
global target_quat_orientation
global target

error_list = []
current_orientation_list = []
current_orientation_quat_list = []
error_orientation_quat_to_euler_list = []
current_orientation_quat_to_euler_list = []


index_of_angles = [2, 8, 11, 18, 26, 30] #hinge for project Hayley v4
# data for project hayley V4 - V2 links
local_linkage_data = [
    [0,0,0,0,0], #0
    [1,0,0,54.5,0], #1 
    [1,0,0,0,0], #theta_1 - Axis1, anti-clockwise with positive theta #2
    [2,0,0,84,0], #3
    [2,0,-90,0,0], #4 
    [2,0,0,61,0], #5 - Z left, X front
    [2,0,180,0,0], #6 - Z right, X front
    [2,90,0,0,0], #7 - Z right, X Up
    [2,0,0,0,0], #theta_2 - Axis2, down with positive theta #8
    [3,0,-180,0,0], #9 - Z left, X Up
    [3,0,0,0,70],   #10    
    [3,0,0,0,0], #theta_3 - Axis3, down with positive theta #11
    [4,0,0,-61,0],     #12
    [4,0,0,0,52],     #13
    [4,90,0,0,0],     #14 - Z left, X front
    [4,0,0,0,57],   #15 
    [4,90,0,0,0],     #16 - Z left, X down
    [4,0,90,0,0],     #17 - Z front, X down
    [4,0,0,0,0], #theta_4 - Axis4, anti-clockwise from top view with positive theta  #18
    [5,90,0,0,0],   #19    - Z front, X left
    [5,0,0,0,47],   #20
    [5,0,0,44.5,0],   #21
    [5,-90,0,0,0],     #22 - Z front, X down
    [5,0,-90,0,0],     #23 - Z left, X down
    [5,-90,0,0,0],     #24 - Z left, X front
    [5,0,0,-47,0],    #25
    [5,0,0,0,0], #theta_5 - Axis5, down with positive theta #25
    [6,0,0,0,93],    #27
    [6,-90,0,0,0],     #28 - Z left, X up
    [6,0,-90,0,0],     #29 - Z front, X up    
    [6,0,0,0,0], #theta_6 - Axis6, anti-clockwise with positive theta #30
    [7,0,0,17.5,0], # dummy 17.5 mm extension #31
    [7,0,90,0,0], #32 - Z left, X up
    [7,90,0,0,0], #33 - Z left, X front
    [7,0,90,0,0], #34 - Z up, X front    
    ]




index_of_hinges = index_of_angles.copy()

list_of_thetas = []
starting_thetas = [0,0,0,0,0,0]

for i, content in enumerate(local_linkage_data):
    list_of_thetas.append(local_linkage_data[i][1])

for j, index in enumerate(index_of_angles):
    list_of_thetas[index] = starting_thetas[j]



angle_max = list_of_thetas.copy()
angle_min = list_of_thetas.copy()
list_of_blockers = list_of_thetas.copy()

for i, value in enumerate(index_of_angles):
    angle_max[value] = upper_limit[i]
    angle_min[value] = lower_limit[i]
    list_of_blockers[value] = 1


# special provision for the blockers to have theta to rotate the coodinate frame that is not due to hinge i.e theta due to mechanical link
for i, linkage in enumerate(local_linkage_data):
    if linkage[1] != 0:
        list_of_blockers[i] = 2
        list_of_thetas[i] = local_linkage_data[i][1]

def interpolate_two_points(points, num_interpolations):
    """
    Interpolate between two 3D points using NumPy's linspace.

    Parameters:
    - point1: Tuple of (x1, y1, z1)
    - point2: Tuple of (x2, y2, z2)
    - num_interpolations: Number of interpolations between the two points.

    Returns:
    - List of interpolated points as tuples [(x1, y1, z1), ..., (xn, yn, zn)]
    """
    x_values = np.linspace(points[0][0], points[1][0], num_interpolations)
    y_values = np.linspace(points[0][1], points[1][1], num_interpolations)
    z_values = np.linspace(points[0][2], points[1][2], num_interpolations)

    interpolated_points = list(zip(x_values, y_values, z_values))
    return interpolated_points


def DH_matrix(theta, alpha, delta, rho):


    transient_matrix = np.eye(4)
    # Handle 3d DH parameters, row-by-row, left-to-right
    theta_rad = theta/180*np.pi
    alpha_rad = alpha/180*np.pi
 
    transient_matrix[0,0]=np.cos(theta_rad)
    transient_matrix[0,1]=-np.sin(theta_rad)
    #transient_matrix[0,2]=0
    transient_matrix[0,3]=rho
 
    transient_matrix[1,0]=np.sin(theta_rad)*np.cos(alpha_rad)
    transient_matrix[1,1]=np.cos(theta_rad)*np.cos(alpha_rad)
    transient_matrix[1,2]=-np.sin(alpha_rad)
    transient_matrix[1,3]=-np.sin(alpha_rad) * delta
 
    transient_matrix[2,0]=np.sin(theta_rad)*np.sin(alpha_rad)
    transient_matrix[2,1]=np.cos(theta_rad)*np.sin(alpha_rad)
    transient_matrix[2,2]=np.cos(alpha_rad)
    transient_matrix[2,3]=np.cos(alpha_rad) * delta

    return transient_matrix


def input_linkage_angles(list_of_thetas):

    for i in range(len(list_of_thetas)):
        local_linkage_data[i][1] = list_of_thetas[i]

    array_matrix = []
    transformation_matrix = None

    for i, linkage in enumerate(local_linkage_data):

        # Rotations first
        transient_rotation = DH_matrix(linkage[1], linkage[2], 0, 0)

        if transformation_matrix is None:
            transformation_matrix = transient_rotation
        else:
            transformation_matrix = np.matmul(transformation_matrix, transient_rotation)

        # then the translations
        transient_translation = DH_matrix(0, 0, linkage[3], linkage[4])

        if transformation_matrix is None:
            transformation_matrix = transient_translation
        else:
            transformation_matrix = np.matmul(transformation_matrix, transient_translation)

        array_matrix.append(transformation_matrix)

    # if pose == True:
    #     transformation_matrix = np.matmul(transformation_matrix, orientation_matrix)
    #     array_matrix.append(transformation_matrix)

    return(array_matrix)


def sqrt_sum_aquare(input_list):
    sum_square = 0
    for value in input_list:
        sum_square += value*value
    return(math.sqrt(sum_square))

# Start - Rotation Matrix to Euler
def rotation_matrix_to_euler(orientation_matrix):

    R11 = orientation_matrix[0,0]
    R12 = orientation_matrix[0,1]
    R13 = orientation_matrix[0,2]

    R21 = orientation_matrix[1,0]
    R22 = orientation_matrix[1,1]
    R23 = orientation_matrix[1,2]

    R31 = orientation_matrix[2,0]
    R32 = orientation_matrix[2,1]
    R33 = orientation_matrix[2,2]

    # https://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf
    # https://stackoverflow.com/questions/15022630/how-to-calculate-the-angle-from-rotation-matrix

    if round(R31,4) != 1.0000 and round(R31,4) != -1.0000:
        #print(R31)
        pitch_1 = -1*asin(R31)
        pitch_2 = pi - pitch_1
        roll_1 = atan2( R32 / cos(pitch_1) , R33 /cos(pitch_1) )
        roll_2 = atan2( R32 / cos(pitch_2) , R33 /cos(pitch_2) )
        yaw_1 = atan2( R21 / cos(pitch_1) , R11 / cos(pitch_1) )
        yaw_2 = atan2( R21 / cos(pitch_2) , R11 / cos(pitch_2) )

         # IMPORTANT NOTE here, there is more than one solution but we choose the first for this case for simplicity !
         # You can insert your own domain logic here on how to handle both solutions appropriately (see the reference publication link for more info).
        pitch = pitch_1
        roll = roll_1
        yaw = yaw_1
    else:
         yaw = 0 # anything (we default this to zero)
         if R31 == -1:
            pitch = pi/2
            roll = yaw + atan2(R12,R13)
         else:
            pitch = -pi/2
            roll = -1*yaw + atan2(-1*R12,-1*R13)

    # convert from radians to degrees
    roll = roll*180/pi
    pitch = pitch*180/pi
    yaw = yaw*180/pi

    rxyz_deg = np.array([roll , pitch , yaw])

    return rxyz_deg
# End - Rotation Matrix to Euler

# Start - Rotation Matrix to Quaternion
def matrix_to_quaternion(rotation_matrix):
    q0 = np.sqrt(1 + rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2]) / 2
    q1 = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / (4 * q0)
    q2 = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / (4 * q0)
    q3 = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / (4 * q0)
    return np.array([q0, q1, q2, q3]) / np.linalg.norm([q0, q1, q2, q3])  # Normalize the quaternion
# End - Rotation Matrix to Quaternion

# Start - Quaternion delta
def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z])

def quaternion_inverse(q):
    w, x, y, z = q
    norm_squared = w**2 + x**2 + y**2 + z**2
    conjugate = np.array([w, -x, -y, -z])
    inverse = conjugate / norm_squared
    return inverse

def quaternion_difference(q1, q2): #q1 ==current_orientation_quaternion, q2 == perturbed_orientation_quaternion
    q1_inv = quaternion_inverse(q1)
    diff = quaternion_multiply(q2, q1_inv)
    return diff
# End - Quaternion delta


# Start - Euler to quaternion and back
def euler_to_quaternion(phi, theta, psi):
    qw = math.cos(phi/2) * math.cos(theta/2) * math.cos(psi/2) + math.sin(phi/2) * math.sin(theta/2) * math.sin(psi/2)
    qx = math.sin(phi/2) * math.cos(theta/2) * math.cos(psi/2) - math.cos(phi/2) * math.sin(theta/2) * math.sin(psi/2)
    qy = math.cos(phi/2) * math.sin(theta/2) * math.cos(psi/2) + math.sin(phi/2) * math.cos(theta/2) * math.sin(psi/2)
    qz = math.cos(phi/2) * math.cos(theta/2) * math.sin(psi/2) - math.sin(phi/2) * math.sin(theta/2) * math.cos(psi/2)
    return qw, qx, qy, qz

def quaternion_to_euler(q):
    theta_x = np.arctan2(2 * (q[0]*q[1] + q[2]*q[3]), 1 - 2 * (q[1]**2 + q[2]**2))
    theta_y = np.arcsin(2 * (q[0]*q[2] - q[3]*q[1]))
    theta_z = np.arctan2(2 * (q[0]*q[3] + q[1]*q[2]), 1 - 2 * (q[2]**2 + q[3]**2))
    return theta_x, theta_y, theta_z
# End - Euler to quaternion and back


def Inverse_Kinematics_Jacobian_Quat(count):

    solved = False

    err_end_to_target = math.inf
    minimum_error = math.inf
    epsilon = 1e-6 # a small value to perturb, assume this is in radians

    num_dimensions = 6  # first 3 are for X,Y,Z and the remaining 3 for quaternion imaginary component representation
    num_joints = 6 # 6 joints
    jacobian_quat_matrix = np.zeros([num_dimensions, num_joints])

    theta_angles = [None] * 6

    orientation_error_list = []
    error_orientation_quat_list = []
    err_end_to_target_list = []
    thetas_list = []

    for loop in range(max_iter):

        P = input_linkage_angles(list_of_thetas) # forward kinematics
        # P is an array of transformation matrix
        # adding on ... the array of matrix are for convenience of plotly traces later on
        # adding on ... IK itself does require to access the individual joint coordinates
        end_to_target = target - P[-1][:3, 3] # getting the last transformation [-1], to extract X, Y, Z
        err_end_to_target = sqrt_sum_aquare(end_to_target)
        err_end_to_target_list.append([loop, err_end_to_target])

        # also track error in euler form
        current_orientation = rotation_matrix_to_euler(P[-1][:3, :3])
        error_orientation = desired_orientation - current_orientation #RPY Euler, degrees
        orientation_error_list.append([loop, error_orientation[0], error_orientation[1], error_orientation[2]])

        # error in quat
        current_orientation_quat = matrix_to_quaternion(P[-1][:3, :3])
        error_orientation_quat = quaternion_difference(current_orientation_quat, target_quat_orientation)
        error_orientation_quat_list.append([loop, error_orientation_quat[0], error_orientation_quat[1], error_orientation_quat[2], error_orientation_quat[3]])
        error_of_rotation = error_orientation_quat[1:] #get only the imaginary parts

        # record the angles of the best minimal error so far; yes the error can increase in further iterations
        if err_end_to_target < minimum_error:
            minimum_error = err_end_to_target
            least_error_angles = list_of_thetas.copy()

        # print(loop, err_end_to_target)
        # print(loop, error_of_rotation)
        # print((np.array(error_of_rotation) < orientation_err_quat))
        # print((np.array(error_of_rotation) < orientation_err_quat).all())

        if (err_end_to_target < displacement_err) and (np.array(error_of_rotation) < orientation_err_quat).all():
            solved = True
            break
        else:

            for k, value in enumerate(index_of_angles): # work on the numerical Jacobian

                perturbed_theta = list_of_thetas.copy()
                perturbed_theta[value] = (perturbed_theta[value] + epsilon)

                perturbed_FK = input_linkage_angles(perturbed_theta) # forward kinematics, output is array of transformation matrices, already chain-multiplied from the individual link transformation
                perturbed_position_delta = perturbed_FK[-1][:3, 3] - P[-1][:3, 3] # get positional delta due to perturb (concept of partial differentiation)

                # Numerical differentiation to compute the Jacobian entry
                jacobian_quat_matrix[:3, k] = perturbed_position_delta / (epsilon)

                # Compute the perturbed end-effector pose
                perturbed_orientation_quat = matrix_to_quaternion(perturbed_FK[-1][:3, :3])

                # Numerical differentiation for quaternion
                perturbed_error_orientation_quat = quaternion_difference(current_orientation_quat, perturbed_orientation_quat)

                # Use the imaginary part of the quaternion as the axis of rotation
                axis_of_rotation = perturbed_error_orientation_quat[1:]

                # Compute the angular velocity
                rate_of_change_of_rotation = axis_of_rotation / epsilon

                # Store the angular velocity in the Jacobian
                jacobian_quat_matrix[3:, k] = rate_of_change_of_rotation

            #error vector
            end_to_target_array = np.transpose(np.append(np.array(end_to_target), error_of_rotation))
            delta_theta = np.linalg.pinv(jacobian_quat_matrix).dot(end_to_target_array)  # Pseudo-inverse used here
            delta_theta = delta_theta*step_size
            
            MAX_STEP_RAD = np.deg2rad(10.0)  # tune this — try 1–3 degrees
            max_joint_change = np.max(np.abs(delta_theta))
            if max_joint_change > MAX_STEP_RAD:
                delta_theta = delta_theta * (MAX_STEP_RAD / max_joint_change)
            

            # Update current joint angle values
            for k, value in enumerate(index_of_angles):
                list_of_thetas[value] = (list_of_thetas[value] + (delta_theta[k]))
                list_of_thetas[value] = (list_of_thetas[value] + 180) % 360 - 180

                # clamp angle
                # if list_of_thetas[value] > angle_max[value]: list_of_thetas[value] = angle_max[value]-(10*np.random.random(1)[0])
                # if list_of_thetas[value] < angle_min[value]: list_of_thetas[value] = angle_min[value]+(10*np.random.random(1)[0])

                if k < 5: #ignore axis 6
                    if list_of_thetas[value] > angle_max[value]: list_of_thetas[value] = angle_max[value]-1
                    if list_of_thetas[value] < angle_min[value]: list_of_thetas[value] = angle_min[value]+1

        for t, value in enumerate(index_of_angles):
            theta_angles[t] = list_of_thetas[value]
            
        thetas_list.append([loop, theta_angles[0], theta_angles[1], theta_angles[2], theta_angles[3], theta_angles[4], theta_angles[5]])

    if solved == False:
        for i in range(len(list_of_thetas)):
            list_of_thetas[i] = least_error_angles[i] #return least error
            err_end_to_target = minimum_error
            P = input_linkage_angles(list_of_thetas) # forward kinematics
   
    return P, list_of_thetas, err_end_to_target, solved, loop, err_end_to_target_list, error_orientation_quat_list, orientation_error_list, thetas_list



def Inverse_Kinematics_Jacobian_Euler(count):

    solved = False

    err_end_to_target = math.inf
    minimum_error = math.inf

    jacobian_matrix = np.zeros([6,6])

    z_vector = [None]*6
    end_effector_to_current_joint = [None]*6
    jacobian_array = [None]*6

    for loop in range(max_iter):

        P = input_linkage_angles(list_of_thetas) # forward kinematics
        # P is an array of transformation matrix
        # adding on ... the array of matrix are for convenience of plotly traces later on
        # adding on ... IK itself does require to access the individual joint coordinates
        end_to_target = target - P[-1][:3, 3] # getting the last transformation [-1], to extract X, Y, Z
        err_end_to_target = sqrt_sum_aquare(end_to_target)

        current_orientation = rotation_matrix_to_euler(P[-1][:3, :3])
        #print(current_orientation)
        #print(desired_orientation)
        error_orientation = desired_orientation - current_orientation #error in euler
        abs_error_orientation = [abs(ele) for ele in error_orientation]
        # print(abs_error_orientation)
        # print((np.array(abs_error_orientation) < orientation_err).all())
        error_list.append([loop, err_end_to_target])
        orientation_error_list.append([loop, error_orientation])

        # record the angles of the best minimal error so far; yes the error can increase in further iterations
        if err_end_to_target < minimum_error:
            minimum_error = err_end_to_target
            least_error_angles = list_of_thetas.copy()


        if (err_end_to_target < displacement_err) and (np.array(abs_error_orientation) < orientation_err).all():
            solved = True
            break
        else:
            for k, value in enumerate(index_of_angles):
                z_vector[k] = np.array(P[value][:3, 2])
                end_effector_to_current_joint[k] = np.transpose(np.array(P[-1][:3, 3] - P[value][:3, 3]))
                jacobian_array[k] = np.transpose(np.cross(z_vector[k],end_effector_to_current_joint[k]))

                for j in range(3):
                    jacobian_matrix[j,k] = jacobian_array[k][j]
                for j in range(3,6):
                    jacobian_matrix[j,k] = z_vector[k][j-3]

            end_to_target_array = np.transpose(np.append(np.array(end_to_target), error_orientation))
            delta_theta = np.linalg.pinv(jacobian_matrix).dot(end_to_target_array)  # Pseudo-inverse used here
            delta_theta = delta_theta*step_size # note delta_theta is in radians

            # Update current joint angle values
            for k, value in enumerate(index_of_angles):
                list_of_thetas[value] = (list_of_thetas[value] + (delta_theta[k]) * 180 / math.pi)
                list_of_thetas[value] = (list_of_thetas[value] + 180) % 360 - 180

                # clamp angle
                if list_of_thetas[value] > angle_max[value]: list_of_thetas[value] = angle_max[value]-(10*np.random.random(1)[0])
                if list_of_thetas[value] < angle_min[value]: list_of_thetas[value] = angle_min[value]+(10*np.random.random(1)[0])

    if solved == False:
        for i in range(len(list_of_thetas)):
            list_of_thetas[i] = least_error_angles[i] #return least error
            err_end_to_target = minimum_error
            P = input_linkage_angles(list_of_thetas) # forward kinematics

    return P, list_of_thetas, err_end_to_target, solved, loop

def Inverse_Kinematics_CCD():
    solved = False
    err_end_to_target = math.inf
    minimum_error = math.inf
    list_of_angles = [0,0,0,0,0,0]

    for loop in range(max_iter):
        for i in range(len(local_linkage_data)-1, -1, -1):

            if list_of_blockers[i] != 0:

                P = input_linkage_angles(list_of_thetas) # forward kinematics
                # P is an array of transformation matrix
                # adding on ... the array of matrix are for convenience of plotly traces later on
                # adding on ... IK itself does require to access the individual joint coordinates
                end_to_target = target - P[-1][:3, 3] # getting the last transformation [-1], to extract X, Y, Z
                err_end_to_target = sqrt_sum_aquare(end_to_target)
                error_list.append([loop, err_end_to_target])

                # record the angles of the best minimal error so far; yes the error can increase in further iterations
                if err_end_to_target < minimum_error:
                    minimum_error = err_end_to_target
                    least_error_angles = list_of_thetas.copy()

                if err_end_to_target < displacement_err:
                    solved = True
                else:

                    if list_of_blockers[i] != 2:

                        # Calculate distance between i-joint position to end effector position
                        # P[i] is position of current joint
                        # P[-1] is position of end effector

                        # reviewed and change code here to improve since normal vector is always Z-axis if theta is always used as rotation
                        # use the DH array matrix, because in the top-left 3x3 sub-matrix, it already contains the vectors
                        # for all 3 axis, the NORMAL axis are: top-row = X axis, middle = Y axis and last row = Z axis

                        # find normal of rotation plane, aka hinge axis (hinge is always normal to rotation plane)
                        normal_vector = list(P[i][2, :3])
                        plane = Plane(point=P[i][:3, 3], normal=normal_vector)

                        # find projection of tgt onto rotation plane
                        # https://scikit-spatial.readthedocs.io/en/stable/gallery/projection/plot_point_plane.html
                        target_point_projected = plane.project_point(target)
                        end_point_projected = plane.project_point(P[-1][:3, 3])

                        # find angle between projected tgt and cur_to_end
                        cur_to_end_projected = end_point_projected - P[i][:3, 3]
                        cur_to_target_projected = target_point_projected - P[i][:3, 3]

                        # end_target_mag = |a||b|
                        cur_to_end_projected_mag = sqrt_sum_aquare(cur_to_end_projected) # aka |a|
                        cur_to_target_projected_mag = sqrt_sum_aquare(cur_to_target_projected) # aka |b|
                        end_target_mag = cur_to_end_projected_mag * cur_to_target_projected_mag # aka |a||b|

                        # if the 2 vectors current-effector and current-target is already very close
                        if end_target_mag <= 0.0001:
                            cos_rot_ang = 1
                            sin_rot_ang = 0
                        else:
                            # dot product rule - https://en.wikipedia.org/wiki/Dot_product
                            # To solve for angle magnitude between 2 vectors
                            # dot product of two Euclidean vectors a and b
                            # a.b = |a||b|cos(lambda)
                            # cos_rot_ang = cos(lambda) = a.b / |a||b|
                            cos_rot_ang = (cur_to_end_projected[0] * cur_to_target_projected[0] + cur_to_end_projected[1] * cur_to_target_projected[1] + cur_to_end_projected[2] * cur_to_target_projected[2]) / end_target_mag

                            # cross product rule - https://en.wikipedia.org/wiki/Cross_product
                            # https://www.mathsisfun.com/algebra/vectors-cross-product.html
                            # cross product of two Euclidean vectors a and b
                            # a X b = |a||b|sin(lambda)
                            # sin_rot_ang = sin(lambda) = [a X b] / |a||b|
                            # To solve for direction of angle A->B or B->A
                            # for theta rotation (about Z axis) in right hand rule, keep using [0] and [1] for finding Z direction
                            # cross product of 3d vectors has i, j, k components
                            # after we do the projections onto the plane level, we will focus on the k component
                            sin_rot_ang = (cur_to_end_projected[0] * cur_to_target_projected[1] - cur_to_end_projected[1] * cur_to_target_projected[0]) / end_target_mag

                        rot_ang = math.acos(max(-1, min(1,cos_rot_ang)))

                        if sin_rot_ang < 0.0:
                            rot_ang = -rot_ang

                        # Update current joint angle values
                        list_of_thetas[i] = list_of_thetas[i] + (rot_ang * 180 / math.pi)
                        list_of_thetas[i] = (list_of_thetas[i] + 180) % 360 - 180

                        # clamp angle
                        if list_of_thetas[i] > angle_max[i]: list_of_thetas[i] = angle_max[i]
                        if list_of_thetas[i] < angle_min[i]: list_of_thetas[i] = angle_min[i]

                    elif list_of_blockers[i] == 2:
                        #list_of_thetas[i] = 90
                        # there was a bug here befoew where the blockers force it to only positive 90 deg
                        # now I have update to adopt whatever linkage data is needed e.g. -90 deg aka 270 deg
                        list_of_thetas[i] = local_linkage_data[i][1]

        if solved:
            break

    if solved == False:
        for i in range(len(list_of_thetas)):
            list_of_thetas[i] = least_error_angles[i] #return least error
            err_end_to_target = minimum_error
            P = input_linkage_angles(list_of_thetas) # forward kinematics

    for i, index_value in enumerate(index_of_angles):
        list_of_angles[i] = list_of_thetas[index_value]

    return P, list_of_angles, err_end_to_target, solved, loop




def quaternion_slerp(q_start, q_end, t): # t ∈ [0, 1]
    """Spherical Linear Interpolation between two quaternions."""
    # Ensure quaternions are normalized
    q0 = q_start / np.linalg.norm(q_start)
    q1 = q_end / np.linalg.norm(q_end)
    
    dot = np.dot(q0, q1)
    
    # If dot < 0, negate one quaternion to take the SHORT arc
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    
    # Clamp to avoid numerical issues with acos
    dot = np.clip(dot, -1.0, 1.0)
    
    # If quaternions are very close, fall back to LERP + normalize
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    
    theta_0 = np.arccos(dot)        # Angle between quaternions
    theta = theta_0 * t             # Angle at t
    
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    return s0 * q0 + s1 * q1


def quaternion_xyzw_to_wxyz(q):
    # Convert the quaternion from xyzw format to wxyz format
    converted_q = [q[3], q[0], q[1], q[2]]
    return converted_q

def is_gimbal_lock(quaternion):
    threshold = 0.001 # not clear yet as to what value constitutes a lock, TBD
    w, x, y, z = quaternion
    # Check if the rotation is close to a two-axis rotation
    return abs(x * x + y * y - 1) < threshold or abs(y * y + z * z - 1) < threshold or abs(z * z + x * x - 1) < threshold


if __name__ == '__main__':

    try:    
    
        # Read CSV file into a DataFrame
        csv_df = pd.read_csv(file_path)
        full_target_list = csv_df.values.tolist()
    
        target_list_length = len(full_target_list)-1
    
        now = datetime.now()
        dt_string = now.strftime("D"+"%d%m%Y"+"_T"+"%H%M%S")
        csv_name = r'IK_logger_'+dt_string+'_'+pose_method+'_'+IK_CCD_angular_method+'.csv'
    
        data = []
        interim_data = []
    
        least_burdensome_angles = list_of_thetas.copy()
    
        angular_RMSE = np.sqrt(sum(np.square(upper_limit)))
    
        delta_upper_limit = [360, 360, 360, 360, 360, 360]
        min_angular_RMSE = np.sqrt(sum(np.square(delta_upper_limit)))
        # get the first and last target from full_target_list
        first_and_last_target_list = [full_target_list[:1][0], full_target_list[-1:][0]]
    
        #for local_k in range(len(full_target_list)):
        for local_k in range(len(first_and_last_target_list)):
            
            #print(local_k)
    
            n=0
            #angular_RMSE = np.sqrt(sum(np.square(upper_limit)))
    
            while(n<IK_CCD_tries):
                target = first_and_last_target_list[local_k]
                # randomize theta
                for index_value in index_of_angles:
                    list_of_thetas[index_value] = random.randint(angle_min[index_value]+1,  angle_max[index_value]-1)
    
                array_matrix, list_of_angles, err_end_to_target, solve_status, iterations = Inverse_Kinematics_CCD()
    
                # angular_RMSE = np.sqrt(sum(np.square(upper_limit)))
                # new_angular_RMSE = angular_RMSE
    
                if (solve_status == True):
                    # get the angular RMSE values
    
                    array_of_angles = np.array(list_of_angles[:5])
                    new_angular_RMSE = np.sqrt(sum(np.square(array_of_angles)))
    
                    if (new_angular_RMSE < angular_RMSE):
                        angular_RMSE = new_angular_RMSE
                        least_burdensome_angles = list_of_thetas.copy()
    
                if (solve_status == False):
                    new_angular_RMSE = np.sqrt(sum(np.square(upper_limit)))                
    
                    # print("local_k:" + str(local_k))
                    # print("IK_CCD_tries:" + str(n))
                    # print("new_angular_RMSE:" + str(new_angular_RMSE))
                    # print("angular_RMSE:" + str(angular_RMSE))
    
                # record data in an inetrim csv for review
    
                calculated_orientation = np.round(rotation_matrix_to_euler(array_matrix[-1][:3, :3]),3)
                Roll = calculated_orientation[0]
                Pitch = calculated_orientation[1]
                Yaw = calculated_orientation[2]
    
                calculated_orientation_quat = list(np.round(matrix_to_quaternion(array_matrix[-1][:3, :3]),6))
                q1 = calculated_orientation_quat[0]
                q2 = calculated_orientation_quat[1]
                q3 = calculated_orientation_quat[2]
                q4 = calculated_orientation_quat[3]
    
                theta1 = round(list_of_angles[0],3)
                theta2 = round(list_of_angles[1],3)
                theta3 = round(list_of_angles[2],3)
                theta4 = round(list_of_angles[3],3)
                theta5 = round(list_of_angles[4],3)
                theta6 = round(list_of_angles[5],3)
    
                X = np.round(array_matrix[-1][0, 3],3)
                Y = np.round(array_matrix[-1][1, 3],3)
                Z = np.round(array_matrix[-1][2, 3],3)
                
                new_angular_RMSE = np.round(new_angular_RMSE,3)
     
                interim_row = [
                    local_k,
                    n,
                    solve_status,
                    iterations,
    
                    target[0],
                    target[1],
                    target[2],
                    X,
                    Y,
                    Z,
    
                    err_end_to_target,
    
                    Roll,
                    Pitch,
                    Yaw,
    
                    q1,
                    q2,
                    q3,
                    q4,
    
                    theta1,
                    theta2,
                    theta3,
                    theta4,
                    theta5,
                    theta6,
                    
                    new_angular_RMSE
    
                    ]
    
                interim_data.append(interim_row)
                n=n+1
    
        interim_df = pd.DataFrame(interim_data, columns = interim_column_headers)
        interim_df.to_csv("interim_csv.csv", index = False, header = True, mode='w+')
    
        # filter out the solved CCD, there must be at least 1 in each of the first and last target to proceed!
        filtered_interim_df = interim_df[interim_df["Solve Status"]==True]
        
        first_and_last_target_id = interim_df["Target"].unique()
        first_target_id = first_and_last_target_id[0]
        last_target_id = first_and_last_target_id[-1]
        
        first_target_df = filtered_interim_df[filtered_interim_df["Target"]==first_target_id]    
        last_target_df = filtered_interim_df[filtered_interim_df["Target"]==last_target_id]
    
        if (IK_CCD_angular_method == "angular_RMSE_closest_to_zero"):
            filtered_first_target_df = first_target_df.loc[[first_target_df["Angular RMSE"].idxmin()]]
            filtered_last_target_df = last_target_df.loc[[last_target_df["Angular RMSE"].idxmin()]]
            
            first_target_thetas = [
                               filtered_first_target_df["theta1"].iloc[0],
                               filtered_first_target_df["theta2"].iloc[0],
                               filtered_first_target_df["theta3"].iloc[0],
                               filtered_first_target_df["theta4"].iloc[0],
                               filtered_first_target_df["theta5"].iloc[0],
                               filtered_first_target_df["theta6"].iloc[0]
                               ]
    
            last_target_thetas = [
                               filtered_last_target_df["theta1"].iloc[0],
                               filtered_last_target_df["theta2"].iloc[0],
                               filtered_last_target_df["theta3"].iloc[0],
                               filtered_last_target_df["theta4"].iloc[0],
                               filtered_last_target_df["theta5"].iloc[0],
                               filtered_last_target_df["theta6"].iloc[0]
                               ]
            
        elif (IK_CCD_angular_method == "angular_RMSE_closest_to_each_other"): 
            
            for i in range(len(first_target_df)):
                
                for j in range(len(last_target_df)):    
                    
                    delta_theta1 = first_target_df.iloc[i]["theta1"] - last_target_df.iloc[j]["theta1"]
                    delta_theta2 = first_target_df.iloc[i]["theta2"] - last_target_df.iloc[j]["theta2"]
                    delta_theta3 = first_target_df.iloc[i]["theta3"] - last_target_df.iloc[j]["theta3"]
                    delta_theta4 = first_target_df.iloc[i]["theta4"] - last_target_df.iloc[j]["theta4"]
                    delta_theta5 = first_target_df.iloc[i]["theta5"] - last_target_df.iloc[j]["theta5"]
                    delta_theta6 = first_target_df.iloc[i]["theta6"] - last_target_df.iloc[j]["theta6"]
        
                    delta_theta_array = np.array([delta_theta1,
                                                 delta_theta2,
                                                 delta_theta3,
                                                 delta_theta4,
                                                 delta_theta5,
                                                 delta_theta6])
                    delta_angular_RMSE = np.sqrt(sum(np.square(delta_theta_array)))
        
                    if (delta_angular_RMSE < min_angular_RMSE):
                        min_angular_RMSE = delta_angular_RMSE
        
                        first_target_thetas = [
                                           first_target_df.iloc[i]["theta1"],
                                           first_target_df.iloc[i]["theta2"],
                                           first_target_df.iloc[i]["theta3"],
                                           first_target_df.iloc[i]["theta4"],
                                           first_target_df.iloc[i]["theta5"],
                                           first_target_df.iloc[i]["theta6"]
                                           ]
    
                        last_target_thetas = [
                                           last_target_df.iloc[j]["theta1"],
                                           last_target_df.iloc[j]["theta2"],
                                           last_target_df.iloc[j]["theta3"],
                                           last_target_df.iloc[j]["theta4"],
                                           last_target_df.iloc[j]["theta5"],
                                           last_target_df.iloc[j]["theta6"]
                                           ]
    
        for k, index in enumerate(index_of_angles):
            list_of_thetas[index] = first_target_thetas[k]
            
        P = input_linkage_angles(list_of_thetas) # forward kinematics
        desired_orientation_start = rotation_matrix_to_euler(P[-1][:3, :3]) # target orientation in euler
        start_quat_orientation = matrix_to_quaternion(P[-1][:3, :3]) # target orientation in quat, of the first solved target
        
        if (pose_method == "first_pose_locked_till_end"): # or "SLERP"        
            end_quat_orientation = start_quat_orientation
    
        elif (pose_method == "SLERP"):
            for l, index in enumerate(index_of_angles):
                list_of_thetas[index] = last_target_thetas[l]
                
            P = input_linkage_angles(list_of_thetas) # forward kinematics
            desired_orientation_end = rotation_matrix_to_euler(P[-1][:3, :3]) # target orientation in euler  
            end_quat_orientation = matrix_to_quaternion(P[-1][:3, :3]) # target orientation in quat, of the first solved target
    
            
        for local_k in range(len(full_target_list)):
    
            # get the interpolated quaternion orientation
            t = (local_k + 1) / (len(full_target_list))
            target_quat_orientation = quaternion_slerp(start_quat_orientation, end_quat_orientation, t)
            
            desired_orientation = np.degrees(quaternion_to_euler(target_quat_orientation))
    
            # restart list_of_thetas to the starting point
            for k, index in enumerate(index_of_angles):
                list_of_thetas[index] = first_target_thetas[k]
                        
            # create and save a fig as html
            fig_1 = px.line()
            fig_2 = px.line()
            fig_3 = px.line()
            fig_4 = px.line() #for tracking theta angles
    
            target = full_target_list[local_k]
            array_matrix, list_of_angles, err_end_to_target, solve_status, iterations, err_end_to_target_list, error_orientation_quat_list, orientation_error_list, thetas_list  = Inverse_Kinematics_Jacobian_Quat(local_k)
    
            print("target: ", target)
            print(f"Solution: {solve_status}")
    
            # if solve_status == False:
            #     false_flag = false_flag+1
    
            theta_angles = [None] * 6
            Rows = local_k
            X = np.round(array_matrix[-1][0, 3],3)
            Y = np.round(array_matrix[-1][1, 3],3)
            Z = np.round(array_matrix[-1][2, 3],3)
            Pos_Error = err_end_to_target
            calculated_orientation = np.round(rotation_matrix_to_euler(array_matrix[-1][:3, :3]),3)
            actual_Roll = calculated_orientation[0]
            actual_Pitch = calculated_orientation[1]
            actual_Yaw = calculated_orientation[2]
            calculated_orientation_quat = list(np.round(matrix_to_quaternion(array_matrix[-1][:3, :3]),6))
            actual_q1 = calculated_orientation_quat[0]
            actual_q2 = calculated_orientation_quat[1]
            actual_q3 = calculated_orientation_quat[2]
            actual_q4 = calculated_orientation_quat[3]
            gimbal_check = is_gimbal_lock(calculated_orientation_quat)
            #gimbal_check = gimbal_check[0]
            IK_Iterations = iterations
            IK_time = time.time() - start_time
            solve_status = str(solve_status)
            for j, index in enumerate(index_of_angles):
                theta_angles[j] = list_of_angles[index]
            theta1 = theta_angles[0]
            theta2 = theta_angles[1]
            theta3 = theta_angles[2]
            theta4 = theta_angles[3]
            theta5 = theta_angles[4]
            theta6 = theta_angles[5]
            
            servo_status = 2
            
            new_angular_RMSE = np.sqrt(sum(np.square(theta_angles[:5])))                
    
            time_stay = 0.03
    
            # if local_k == 0 or (local_k-1 > 1):
            #     time_stay = 5
            # else:
            #     time_stay = 0.03
    
            row = [
                Rows,
                target[0],
                target[1],
                target[2],
                X,
                Y,
                Z,
                Pos_Error,
    
                desired_orientation[0],
                desired_orientation[1],
                desired_orientation[2],
    
                actual_Roll,
                actual_Pitch,
                actual_Yaw,
    
                target_quat_orientation[0],
                target_quat_orientation[1],
                target_quat_orientation[2],
                target_quat_orientation[3],
    
                actual_q1,
                actual_q2,
                actual_q3,
                actual_q4,
    
                gimbal_check,
                IK_Iterations,
                IK_time,
                solve_status,
                theta1,
                theta2,
                theta3,
                theta4,
                theta5,
                theta6,
                time_stay,
                servo_status,
                new_angular_RMSE
                ]
    
            data.append(row)
            df = pd.DataFrame(data, columns = column_headers)
            df.to_csv(csv_name, index = False, header = True, mode='w+')
        
            custom_text = "target no. " + str(local_k)
    
            err_end_to_target_df = pd.DataFrame(err_end_to_target_list, columns =['loop', 'pos_error'])
            error_orientation_quat_df = pd.DataFrame(error_orientation_quat_list, columns =['loop', 'q0_error', 'q1_error', 'q2_error', 'q3_error'])
            orientation_error_df = pd.DataFrame(orientation_error_list, columns =['loop', 'roll_error', 'pitch_error', 'yaw_error'])
            thetas_df = pd.DataFrame(thetas_list, columns =['loop', 'theta1', 'theta2', 'theta3', 'theta4', 'theta5', 'theta6'])
    
    
            fig_1.add_scatter(x=err_end_to_target_df["loop"], y=err_end_to_target_df["pos_error"]#, text=target, textposition="top right",
                            #mode="markers+text", marker_size=18,
                              # marker_symbol= "triangle-up",
                              # marker_color= "black",
                              # showlegend=False
                            )
    
            fig_1.update_layout(
                xaxis_title="Iteration", yaxis_title="Positional Error (mm)",
                title=dict(text="<b>" + custom_text + " " + str(target) +  "</b>", font=dict(size=14), yref='paper'),
                # legend=dict(
                #     x=0.05,
                #     y=0.95,
                #     traceorder="normal",
                #     font=dict(
                #         family="sans-serif",
                #         size=12,
                #         color="black"
                #     ),
                # )
                )
    
            for i, columns in enumerate(error_orientation_quat_df.columns):
                if i>1:
                    fig_2.add_scatter(x=error_orientation_quat_df["loop"], y=error_orientation_quat_df[columns])#, text=target, textposition="top right")
    
            fig_2.update_layout(
                xaxis_title="Iteration", yaxis_title="Quaternions Error",
                title=dict(text="<b>" + custom_text + " " + str(target) + "</b>", font=dict(size=14), yref='paper'),
                # legend=dict(
                #     x=0.05,
                #     y=0.95,
                #     traceorder="normal",
                #     font=dict(
                #         family="sans-serif",
                #         size=12,
                #         color="black"
                #     ),
                # )
                )
    
    
            for i, columns in enumerate(orientation_error_df.columns):
                if i>0:
                    fig_3.add_scatter(x=orientation_error_df["loop"], y=orientation_error_df[columns])#, text=target, textposition="top right")
    
            fig_3.update_layout(
                xaxis_title="Iteration", yaxis_title="Euler Error (°)",
                title=dict(text="<b>" + custom_text + " " + str(target) + "</b>", font=dict(size=14), yref='paper'),
                # legend=dict(
                #     x=0.05,
                #     y=0.95,
                #     traceorder="normal",
                #     font=dict(
                #         family="sans-serif",
                #         size=12,
                #         color="black"
                #     ),
                # )
                )
    
    
    
            for i, columns in enumerate(thetas_df.columns):
                if i>0:
                    fig_4.add_scatter(x=thetas_df["loop"], y=thetas_df[columns])#, text=target, textposition="top right")
    
            fig_4.update_layout(
                xaxis_title="Iteration", yaxis_title="Thetas (°)",
                title=dict(text="<b>" + custom_text + " " + str(target) + "</b>", font=dict(size=14), yref='paper'),
                # legend=dict(
                #     x=0.05,
                #     y=0.95,
                #     traceorder="normal",
                #     font=dict(
                #         family="sans-serif",
                #         size=12,
                #         color="black"
                #     ),
                # )
                )
    
            with open('IK_logger_'+dt_string+'error_chart.html', 'a') as f:
                f.write(fig_1.to_html(full_html=False, include_plotlyjs='cdn'))
                f.write(fig_2.to_html(full_html=False, include_plotlyjs='cdn'))
                f.write(fig_3.to_html(full_html=False, include_plotlyjs='cdn'))
                f.write(fig_4.to_html(full_html=False, include_plotlyjs='cdn'))
            print("--- %s seconds ---" % IK_time)

    except:
        print("Error - sanity check coordinates and also increase number of IK CCD tries!")
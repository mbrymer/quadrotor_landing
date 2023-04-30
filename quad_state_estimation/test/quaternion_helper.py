#!/usr/bin/env python

#
# AER 1810 Quadrotor Landing Project
# Quaternion Helper Functions
#

from __future__ import division, print_function, absolute_import

import numpy as np

small_angle_tol = 1E-10

def quaternion_exp(pure_quat):
    "Quaternion exponential map from pure to unit quaternion"
    unit_quat = np.zeros(4)
    norm = np.linalg.norm(pure_quat)

    # Real part is fine
    unit_quat[3] = np.math.cos(norm / 2)

    if norm < small_angle_tol:
        # Small rotation, use Taylor series approximation for vector part of quaternion to avoid zero division
        unit_quat[0:3] = pure_quat.flatten() / 2 * (1 - (norm ** 2) / 24)
    else:
        # Rotation is large enough, use true rotation axis
        unit_quat[0:3] = pure_quat.flatten() / norm * np.math.sin(norm / 2)
        
    return unit_quat

def quaternion_log(unit_quat):
    "Quaternion logarithmic map from unit to pure quaternion"
    pure_quat = np.zeros(3)

    norm = np.linalg.norm(unit_quat[0:3])

    if norm < small_angle_tol:
        # Approximate map with Taylor series for atan
        pure_quat = 2 / unit_quat[3] * (1 - (norm / unit_quat[3]) ** 2 / 3)*unit_quat[0:3]
    else:
        # Large angles, use full version of atan
        phi = 2 * np.math.atan2(norm, unit_quat[3])
        u = unit_quat[0:3] / norm
        pure_quat = phi * u
    
    return pure_quat

def quaternion_norm(unit_quat):
    "Normalize unit length quaternion and clip to single cover range"
    q_w_lim = -0.75

    unit_quat_norm = unit_quat / np.linalg.norm(unit_quat)

    if unit_quat_norm[3] < q_w_lim:
        return -unit_quat_norm
    else:
        return unit_quat_norm

def identity_quaternion():
    "Identity quaternion"
    q = np.zeros(4)
    q[3] = 1
    return q

def skew_symm(vector):
    "Return the 3x3 skew symmetric matrix of a vector"
    vector_flatten = vector.flatten()
    skew = np.array([[0,-vector_flatten[2], vector_flatten[1]],
                    [vector_flatten[2], 0, -vector_flatten[0]],
                    [-vector_flatten[1], vector_flatten[0], 0]])
    return skew

def vex_symm(skew):
    "Return the vex operator of a skew symmetric matrix"
    vex = np.zeros((3,1))
    vex[0,0] = (skew[2,1] - skew[1,2]) / 2
    vex[1,0] = (skew[0,2] - skew[2,0]) / 2
    vex[2,0] = (skew[1,0] - skew[0,1]) / 2
    return vex

def exponential_map(vector):
    "Return the rotation matrix corresponding to a Lie algebra vector"
    norm = np.linalg.norm(vector)

    if norm < small_angle_tol:
        # First order approximation
        return np.eye(3) + skew_symm(vector)
    else:
        # Full exponential map
        return (np.math.cos(norm) * np.eye(3) +
        (1 - np.math.cos(norm)) * np.dot(vector.reshape((3,1)), vector.reshape((1,3))) +
        np.math.sin(norm) * skew_symm(vector))

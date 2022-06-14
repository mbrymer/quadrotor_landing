#!/usr/bin/env python

#
# AER 1810 Quadrotor Landing Project
# Quaternion Helper Functions
#

from __future__ import division, print_function, absolute_import

import numpy as np

def quaternion_exp(pure_quat):
    "Quaternion exponential map from pure to unit quaternion"
    norm_tol = 1E-10
    unit_quat = np.zeros(4)
    norm = np.linalg.norm(pure_quat)

    # Real part is fine
    unit_quat[3] = np.math.cos(norm/2)

    if norm<norm_tol:
        # Small rotation, use Taylor series approximation for vector part of quaternion to avoid zero division
        unit_quat[0:3] = pure_quat/2*(1-(norm**2)/24)
    else:
        # Rotation is large enough, use true rotation axis
        unit_quat[0:3] = pure_quat/norm*np.math.sin(norm/2)
        
    return unit_quat

def quaternion_log(unit_quat):
    "Quaternion logarithmic map from unit to pure quaternion"
    norm_tol = 1E-10
    pure_quat = np.zeros(3)

    norm = np.linalg.norm(unit_quat[0:3])

    if norm < norm_tol:
        # Approximate map with Taylor series for atan
        pure_quat = 2/unit_quat[3]*(1-(norm/unit_quat[3])**2/3)*unit_quat[0:3]
    else:
        # Large angles, use full version of atan
        phi = 2*np.math.atan2(norm,unit_quat[3])
        u = unit_quat[0:3]/norm
        pure_quat = phi*u
    
    return pure_quat

def skew_symm(vector):
    "Return the 3x3 skew symmetric matrix of a vector"
    skew = np.array([[0,-vector[2],vector[1]],
                    [vector[2],0,-vector[0]],
                    [-vector[1],vector[0],0]])
    return skew
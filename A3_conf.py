# -*- coding: utf-8 -*-
import random
import numpy as np

N = 200        # horizon size
dt = 2e-2        # time step

w_u = 1e-10    # weight for control input
w_x = dt
w_v = dt    # weight for velocity cost
nq = 1                              # number of joint position
nv = 1                              # number of joint velocity
nu = 1                              # number of control

DATA_FOLDER = 'data/'     # your data folder name
DATA_FILE_NAME = 'warm_start' # your data file name
save_warm_start = 0
use_warm_start = 0
if use_warm_start:
    INITIAL_GUESS_FILE = DATA_FILE_NAME
else:
    INITIAL_GUESS_FILE = None

lowerPositionLimit = -np.pi      # min joint position
upperPositionLimit = np.pi      # max joint position
upperVelocityLimit = 10             # min joint velocity
lowerVelocityLimit = -10            # min joint velocity
lowerControlBound    = -9.8   # lower bound joint torque
upperControlBound    = 9.8    # upper bound joint torque

x_des_final = np.array([0,0])        # final desired joint position and veloci
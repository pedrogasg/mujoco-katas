import numpy as np
import mujoco as mj
from control import Control

class DbPen(Control):
    def __init__(self, model, data) -> None:
        data.qpos[0] = -0.1

    def control(self, model, data):
        mj.mj_energyPos(model, data)

        # Evaluate velocity-dependent energy (kinetic).
        mj.mj_energyVel(model, data)

        # Convert sparse inertia matrix M into full (i.e. dense) matrix.
        # M is filled with the data from data.qM
        M = np.zeros((2, 2))
        mj.mj_fullM(model, M, data.qM)
        # Defines PD gain and reference angles
        Kp = 100 * np.eye(2)
        Kd = 10 * np.eye(2)
        qref = np.array([[-1.6], [1.6]])

        # f compensates Coriolis and gravitational forces
        f = data.qfrc_bias[:, np.newaxis]

        # τ = M * ddqref
        ddqref = Kp @ (qref - data.qpos[:2][:, np.newaxis]) + \
            Kd @ (0 - data.qvel[:2][:, np.newaxis])
        tau = M @ ddqref

        data.qfrc_applied = (tau + f)[:, 0]
            
        

    
import numpy as np
import mujoco as mj
from control import Control
from numpy.linalg import inv
from scipy.linalg import solve_continuous_are

class Acrobot(Control):
    def __init__(self, model, data) -> None:
        
        A, B = self.linearization(model, data)
        Q = np.diag([10, 10, 10, 10])
        R = np.diag([0.1])

        # Naive lqr
        #P = solve_continuous_are(A, B, Q, R)
        #self.K = -inv(B.T @ P @ B + R) @ B.T @ P @ A

        # Numerical stable lqr 
        X = np.matrix(solve_continuous_are(A, B, Q, R))
        self.K = -np.matrix(inv(R)*(B.T*X))


    def control(self, model, data):
        """
        This function implements a LQR controller for balancing.
        """
        state = np.array([
            [data.qpos[0]],
            [data.qvel[0]],
            [data.qpos[1]],
            [data.qvel[1]],
        ])
        data.ctrl[0] = (self.K @ state)[0, 0]

        # Apply noise to shoulder
        noise = mj.mju_standardNormal(0.0)
        data.xfrc_applied[2,0] = 2*noise
        #data.qfrc_applied[0] = noise
        #data.qfrc_applied[1] = noise
            
    def linearization(self, model ,data, pert=0.001):
        f0 = self.get_dx(model, data, np.zeros(5))

        Jacobians = []
        for i in range(5):
            inputs_i = np.zeros(5)
            inputs_i[i] = pert
            jac = (self.get_dx(model, data, inputs_i) - f0) / pert
            Jacobians.append(jac[:, np.newaxis])

        A = np.concatenate(Jacobians[:4], axis=1)
        B = Jacobians[-1]

        return A, B  

    def get_dx(self, model, data, inputs):
        """
        The state is [q1, dq1, q2, dq2]
        The inputs are [q1, dq1, q2, dq2, u]

        The function outputs [dq1, ddq1, dq2, ddq2]
        """
        # Apply inputs
        data.qpos[0] = inputs[0]
        data.qvel[0] = inputs[1]
        data.qpos[1] = inputs[2]
        data.qvel[1] = inputs[3]
        data.ctrl[0] = inputs[4]

        mj.mj_forward(model, data)

        # Record outputs
        dq1 = data.qvel[0]
        dq2 = data.qvel[1]

        # Convert sparse inertia matrix M into full (i.e. dense) matrix.
        # M is filled with the data from data.qM
        M = np.zeros((2, 2))
        mj.mj_fullM(model, M, data.qM)

        # Calculate f = ctrl - qfrc_bias
        f = np.array([
            [0 - data.qfrc_bias[0]],
            [data.ctrl[0] - data.qfrc_bias[1]]
        ])

        # Calculate qacc
        ddq = inv(M) @ f

        outputs = np.array([dq1, ddq[0, 0], dq2, ddq[1, 0]])

        return outputs
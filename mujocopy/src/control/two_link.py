import math
import numpy as np
import mujoco as mj
from itertools import cycle
from control import Control

class TwoLink(Control):

    def __init__(self, model, data) -> None:

        #self.theta1 = np.pi/3
        #self.theta2 = -np.pi/2
        #data.qpos[0] = self.theta1
        #data.qpos[1] = self.theta2
        self.theta1 = -0.5
        self.theta2 = 1.0
        data.qpos[0] = self.theta1
        data.qpos[1] = self.theta2
        mj.mj_forward(model,data)
        self.n = 10000
        position_Q = data.sensordata[:3]
        self.r = 0.5
        self.center = np.array([position_Q[0]-self.r, position_Q[1]])
        self.phi = np.linspace(0,2*np.pi,self.n)
        self.philess = cycle(np.linspace(0,2*np.pi,600))
        self.positions = self.next_position()

        super().__init__(model, data)

    def next_position(self):
        while True:
            phi_big = next(self.philess)
            x = 1 * np.cos(phi_big)
            y = 1 * np.sin(phi_big)
            for phi in self.phi:
                yield x  + self.r * np.cos(phi), y + self.r * np.sin(phi)


    def control(self, model, data):

        end_eff_pos = data.sensordata[:3]

        # Compute end-effector Jacobian
        jacp = np.zeros((3, 2))
        mj.mj_jac(model, data, jacp, None, end_eff_pos, 2)

        # Δq = Jinv * Δx
        J = jacp[[0, 1], :]
        #dx = np.array([
        #    [self.theta1+ self.r * np.cos(data.time) - data.sensordata[0]],
        #    [self.theta2+ self.r * np.sin(data.time) - data.sensordata[1]]
        #])
        dx = np.array([
            [(self.theta1+ self.r * np.sqrt(2) * np.cos(data.time)/(np.sin(data.time)**2 + 1)) - data.sensordata[0]],
            [(self.theta2+ self.r * np.sqrt(2) * np.cos(data.time) * np.sin(data.time)/ (np.sin(data.time)**2 + 1)) - data.sensordata[1]]
        ])
        Jinv = np.linalg.inv(J)
        dq = Jinv.dot(dx)

        # Target position is q + Δq
        data.ctrl[0] = data.qpos[0] + dq[0, 0]
        data.ctrl[2] = data.qpos[1] + dq[1, 0]

        return super().control(model, data)

    def control2(self, model, data):

        position_Q = data.sensordata[:3]
        jacp = np.zeros((3,2))
        mj.mj_jac(model,data,jacp,None,position_Q,2)
        
        J = jacp[[0,1],:]
        Jinv = np.linalg.inv(J)
        
        x, y = next(self.positions)
        dX = np.array([x-position_Q[0],y- position_Q[1]])
        dq = Jinv.dot(dX)
        #self.theta1 += dq[0]
        #self.theta2 += dq[1]
        position_Q = data.sensordata[:3]
        data.ctrl[0] = position_Q[0] + dq[0]
        data.ctrl[2] = position_Q[1] + dq[1]
        #data.qpos[0] = self.theta1
        #data.qpos[1] = self.theta2

        return super().control(model, data)
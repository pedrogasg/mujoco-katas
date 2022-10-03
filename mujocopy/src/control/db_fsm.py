import numpy as np
import mujoco as mj
from control import Control

class DbFsm(Control):

    FSM_HOLD = 0
    FSM_SWING1 = 1
    FSM_SWING2 = 2
    FSM_STOP = 3

    def __init__(self, model, data) -> None:

        data.qpos[0] = -1.0

        self.fsm_state = DbFsm.FSM_HOLD

        self.t_hold = 0.5
        t_swing1 = 1.0
        t_swing2 = 1.0

        # Define setpoints
        self.q_init = np.array([[-1.0], [0.0]])
        self.q_mid = np.array([[0.5], [-2.0]])
        self.q_end = np.array([[1.0], [0.0]])

        # Define setpoint times
        self.t_init = self.t_hold
        self.t_mid = self.t_hold + t_swing1
        self.t_end = self.t_hold + t_swing1 + t_swing2


        self.a_swing1 = self.generate_trajectory(
        self.t_init, self.t_mid, self.q_init, self.q_mid)

        self.a_swing2 = self.generate_trajectory(
        self.t_mid, self.t_end, self.q_mid, self.q_end)

    def generate_trajectory(self, t0, tf, q0, qf):
        """
        Generates a trajectory
        q(t) = a0 + a1t + a2t^2 + a3t^3
        which satisfies the boundary condition
        q(t0) = q0, q(tf) = qf, dq(t0) = 0, dq(tf) = 0
        """
        tf_t0_3 = (tf - t0)**3
        a0 = qf*(t0**2)*(3*tf-t0) + q0*(tf**2)*(tf-3*t0)
        a0 = a0 / tf_t0_3

        a1 = 6 * t0 * tf * (q0 - qf)
        a1 = a1 / tf_t0_3

        a2 = 3 * (t0 + tf) * (qf - q0)
        a2 = a2 / tf_t0_3

        a3 = 2 * (q0 - qf)
        a3 = a3 / tf_t0_3

        return a0, a1, a2, a3

    def set_state(self, time):
        # Check for state change
        if self.fsm_state == DbFsm.FSM_HOLD and time >= self.t_hold:
            self.fsm_state = DbFsm.FSM_SWING1
        elif self.fsm_state == DbFsm.FSM_SWING1 and time >= self.t_mid:
            self.fsm_state = DbFsm.FSM_SWING2
        elif self.fsm_state == DbFsm.FSM_SWING2 and time >= self.t_end:
            self.fsm_state = DbFsm.FSM_STOP

    def get_posvel(self, time):
        
        # Get reference joint position & velocity
        if self.fsm_state == DbFsm.FSM_HOLD:

            return (self.q_init,
                np.zeros((2, 1)))

        elif self.fsm_state == DbFsm.FSM_SWING1:

            return (self.a_swing1[0] + self.a_swing1[1]*time + \
                self.a_swing1[2]*(time**2) + self.a_swing1[3]*(time**3),
                self.a_swing1[1] + 2 * self.a_swing1[2] * \
                time + 3 * self.a_swing1[3]*(time**2))

        elif self.fsm_state == DbFsm.FSM_SWING2:

            return(self.a_swing2[0] + self.a_swing2[1]*time + \
                self.a_swing2[2]*(time**2) + self.a_swing2[3]*(time**3),
                self.a_swing2[1] + 2 * self.a_swing2[2] * \
                time + 3 * self.a_swing2[3]*(time**2))

        elif self.fsm_state == DbFsm.FSM_STOP:
            return (self.q_end,
                np.zeros((2, 1)))

        else:
            return (np.zeros((2, 1)),
                np.zeros((2, 1)))

    def control(self, model, data):
        time = data.time

        self.set_state(time)

        q_ref, dq_ref = self.get_posvel(time)

        # Define PD gains
        kp = 500
        kv = 50

        # Compute PD control
        torque = kp * (q_ref[:, 0] - data.qpos) + \
            kv * (dq_ref[:, 0] - data.qvel)

        for i in range(0,6):
            data.ctrl[i]=0

        data.ctrl[0] = torque[0]
        data.ctrl[3] = torque[1]
            
        

    
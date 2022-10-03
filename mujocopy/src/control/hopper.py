import numpy as np
import mujoco as mj
from control import Control

class Hopper(Control):
    FSM_AIR1 = 0
    FSM_STANCE1 = 1
    FSM_STANCE2 = 2
    FSM_AIR2 = 3
    def __init__(self, model, data) -> None:
        
        self.fsm = self.FSM_AIR1
        self.step_no = 0
        # pservo-hip
        
        self.set_position_servo(model, 0, 100)

        # vservo-hip
        self.set_velocity_servo(model, 1, 10)

        # pservo-knee
        self.set_position_servo(model, 2, 1000)

        # vservo-knee
        self.set_velocity_servo(model, 3, 0)

    def set_state(self, z_foot, vz_torso):
                # Lands on the ground
        if self.fsm == Hopper.FSM_AIR1 and z_foot < 0.05:
            self.fsm = Hopper.FSM_STANCE1

        # Moving upward
        if self.fsm == Hopper.FSM_STANCE1 and vz_torso > 0.0:
            self.fsm = Hopper.FSM_STANCE2

        # Take off
        if self.fsm == Hopper.FSM_STANCE2 and z_foot > 0.05:
            self.fsm = Hopper.FSM_AIR2

        # Moving downward
        if self.fsm == Hopper.FSM_AIR2 and vz_torso < 0.0:
            self.fsm = Hopper.FSM_AIR1
            self.step_no += 1

    def get_posvel(self):

        if self.fsm == Hopper.FSM_STANCE1:
            return 0, 1000, 0

        if self.fsm == Hopper.FSM_STANCE2:
            return -0.2, 1000, 0

        
        return 0, 100,  10

    def control(self, model, data):

        body_no = 3
        z_foot = data.xpos[body_no, 2]
        vz_torso = data.qvel[1]

        self.set_state(z_foot, vz_torso)
        ctrl, kp, kv = self.get_posvel()
        self.set_position_servo(model, 2, kp)
        self.set_velocity_servo(model, 3, kv)
        data.ctrl[0] = ctrl


    def control2(self, model, data):

        body_no = 3
        z_foot = data.xpos[body_no, 2]
        vz_torso = data.qvel[1]

        self.set_state(z_foot, vz_torso)

        if self.fsm == Hopper.FSM_AIR1:
            self.set_position_servo(model, 2, 100)
            self.set_velocity_servo(model, 3, 10)

        if self.fsm == Hopper.FSM_STANCE1:
            self.set_position_servo(model, 2, 1000)
            self.set_velocity_servo(model, 3, 0)

        if self.fsm == Hopper.FSM_STANCE2:
            self.set_position_servo(model, 2, 1000)
            self.set_velocity_servo(model, 3, 0)
            data.ctrl[0] = -0.2

        if self.fsm == Hopper.FSM_AIR2:
            self.set_position_servo(model, 2, 100)
            self.set_velocity_servo(model, 3, 10)
            data.ctrl[0] = 0.0

    def set_position_servo(self, model, actuator_no, kp):
        model.actuator_gainprm[actuator_no, 0] = kp
        model.actuator_biasprm[actuator_no, 1] = -kp

    def set_velocity_servo(self, model, actuator_no, kv):
        model.actuator_gainprm[actuator_no, 0] = kv
        model.actuator_biasprm[actuator_no, 2] = -kv
        

    
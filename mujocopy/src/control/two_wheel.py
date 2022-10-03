from control import Control

class TwoWheel(Control):
    def __init__(self, model, data) -> None:
        pass

    def control(self, model, data):
        data.ctrl[0] = 0.4
        data.ctrl[1] = 0.2
        

    
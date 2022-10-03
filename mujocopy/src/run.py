
import os
import numpy as np
import mujoco as mj
from viewer import Viewer
from control import TwoLink, TwoWheel, DbPen, DbFsm, Acrobot, Hopper

from absl import app


def main(argv):
    controls = {
        "two_link.xml": TwoLink,
        "two_wheel.xml": TwoWheel,
        "db_pen.xml": DbPen,
        "db_fsm.xml": DbFsm,
        "acrobot.xml": Acrobot,
        "hopper.xml": Hopper
    }
    if len(argv) < 2:
        raise Exception("Required positional argument missing. Example Usage: python xml-explore.py cheetah.xml")

    xml_path = argv[1]

    sim = os.path.basename(xml_path)


    # MuJoCo data structures
    model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
    data = mj.MjData(model)                # MuJoCo data

    controller = controls[sim](model, data)

    viewer = Viewer(model, data)

    #initialize the controller

    #set the controller


    mj.set_mjcb_control(controller.control)

    viewer.run()

    


if __name__ == '__main__':
    app.run(main)
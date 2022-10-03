from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import io as resources

SUITE = containers.TaggedTasks()


@SUITE.add()
def empty(xml):
    xml_string = common.read_model(xml)
    physics = mujoco.Physics.from_xml_string(xml_string, common.ASSETS)
    task = EmptyTask()
    env = control.Environment(physics, task)
    return env


class EmptyTask(base.Task):

    def initialize_episode(self, physics):
        
        super().initialize_episode(physics)

    def before_step(self, action, physics):  
        #physics.named.data.ctrl["motor_front_left"] = .73550
        #physics.named.data.ctrl["motor_back_left"] = .73550
        #physics.named.data.ctrl["motor_back_right"] = .73550
        #physics.named.data.ctrl["motor_front_right"] = .73550
        pass

    def get_observation(self, physics):

        return dict()

    def get_reward(self, physics):
        return 0.0



from absl import app

from dm_control import viewer


def main(argv):
    if len(argv) < 2:
        raise Exception("Required positional argument missing. Example Usage: python xml-explore.py cheetah.xml")

    xml_file = argv[1]


    def loader():
        xml_string = resources.GetResource(xml_file)
        physics = mujoco.Physics.from_xml_string(xml_string)
        
        task = EmptyTask()
        env = control.Environment(physics, task)
        return env

    viewer.launch(loader)


if __name__ == '__main__':
    app.run(main)
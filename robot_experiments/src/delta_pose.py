import sys
import copy
import rospy
import tf
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import math
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
import collections
import numpy as np
import time

pose_vec = collections.namedtuple('pose_vec', ['x', 'y', 'z', 'rx', 'ry', 'rz', 'rw'], verbose=False)

# def q_mult(q1, q2):
#     w1, x1, y1, z1 = q1.w, q1.x, q1.y, q1.z 
#     w2, x2, y2, z2 = q2.w, q2.x, q2.y, q2.z

#     w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
#     x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
#     y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
#     z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
#     return w, x, y, z

def quaternion2list(q):
    return [q.x, q.y, q.z, q.w]

def list2quaternion(l):
    assert len(l) == 4
    return geometry_msgs.msg.Quaternion(*l)

class PR2RobotController(object):
    """docstring for PR2RobotController"""
    def __init__(self, group_name):
        super(PR2RobotController, self).__init__()
        self.group_name = group_name
        
        ## First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)

        if rospy.client._init_node_args is None:
            rospy.init_node('delta_pose_mover', anonymous=True)

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander(self.group_name)

        self.reference_frame = '/base_link'
        self.group.set_pose_reference_frame(self.reference_frame)
        self.group.set_goal_tolerance(0.001)

        self.group.set_planning_time(1)
        # self.group.set_max_velocity_scaling_factor(0.5)
        self.group.set_max_acceleration_scaling_factor(0.5)

    def reset_pose(self):
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation.w = 1.0
        pose_goal.position.x = 0.5
        pose_goal.position.y = -0.6
        pose_goal.position.z = 0.9
        self.group.set_pose_target(pose_goal)
        plan = self.group.go(wait=True)
        self.group.stop()
        self.group.clear_pose_targets()
        print('Reset to initial condition')

    def apply_delta_pose(self, orig, delta):
        new = geometry_msgs.msg.Pose()
        
        new.position.x = orig.position.x + delta.position.x
        new.position.y = orig.position.y + delta.position.y
        new.position.z = orig.position.z + delta.position.z

        # Todo: add orientation offset
        # w, x, y, z = q_mult(orig.orientation, delta.orientation)
        # new.orientation.x = x
        # new.orientation.y = y
        # new.orientation.z = z
        # new.orientation.w = w

        # http://wiki.ros.org/tf2/Tutorials/Quaternions
        new.orientation = list2quaternion(tf.transformations.quaternion_multiply(quaternion2list(delta.orientation), quaternion2list(orig.orientation)))

        return new

    # Tf useful - https://answers.ros.org/question/69754/quaternion-transformations-in-python/?answer=69799#post-id-69799
    def _move_delta_geom_pose(self, d):
        # Calc new pose

        c = self.group.get_current_pose().pose
        
        if self.reference_frame == "/base_link":
            c.position.z -= 0.051
        elif self.reference_frame != "/odom_combined":
            print("WARNING: PLEASE CHECK YOUR REFERENCE FRAME!")

        new = self.apply_delta_pose(c, d)

        # Test:
        quaternion = (c.orientation.x, c.orientation.y, c.orientation.z, c.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        quaternion2 = (new.orientation.x, new.orientation.y, new.orientation.z, new.orientation.w)
        euler2 = tf.transformations.euler_from_quaternion(quaternion2)
        # print('Euler before', euler, ' Euler after: ', euler2)
        ## End test

        # Move to new pose
        self.group.set_pose_target(new)
        print('Going to pose')
        plan = self.group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        # self.group.stop()
        # self.group.clear_pose_targets()
        print('Stopped')

    def move_delta(self, d):
        delta = geometry_msgs.msg.Pose()

        for k in ['x', 'y', 'z']:
            setattr(delta.orientation, k, getattr(d, 'r' + k))
            setattr(delta.position, k, getattr(d, k))
        delta.orientation.w = d.rw

        self._move_delta_geom_pose(delta)

    def move_delta_t_rpy(self, t, rpy):
        assert len(t) == 3
        assert len(rpy) == 3

        qt = tf.transformations.quaternion_from_euler(rpy[0], rpy[1], rpy[2])
        delta = pose_vec(t[0], t[1], t[2], qt[0], qt[1], qt[2], qt[3])

        self.move_delta(delta)

    def move_delta_t(self, t):
        self.move_delta_t_rpy(t, [0, 0, 0])


def main3():
    pr2 = PR2RobotController('right_arm')
    pr2.reset_pose()

    for _ in range(10):
        delta_t = np.random.uniform(low=-0.02, high=0.02, size=3)
        delta_rpy = np.random.uniform(low=-math.pi/6., high=math.pi/6., size=3)
        print(delta_t)
        print(delta_rpy)
        # time.sleep(2)
        # pr2.move_delta_t(delta_t)
        pr2.move_delta_t_rpy(delta_t, delta_rpy)


def main2():
    pr2 = PR2RobotController('right_arm')
    pr2.reset_pose()


    roll = 0
    pitch = 0
    yaw = math.pi/4
    qt = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
    print(qt)
    # #type(pose) = geometry_msgs.msg.Pose
    # pose.orientation.x = qt[0]
    # pose.orientation.y = qt[1]
    # pose.orientation.z = qt[2]
    # pose.orientation.w = qt[3]


    delta = pose_vec(-0.00, 0, 0, qt[0], qt[1], qt[2], qt[3])

    for _ in range(2):
        pr2.move_delta(delta)
    # pr2.move_delta(delta)
    # pr2.move_delta(delta)


def main():
    ## First initialize `moveit_commander`_ and a `rospy`_ node:
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('delta_gripper_mover', anonymous=True)

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group_name = "right_arm"
    group = moveit_commander.MoveGroupCommander(group_name)


    planning_frame = group.get_planning_frame()
    print "============ Reference frame: %s" % planning_frame
    # We can also print the name of the end-effector link for this group:
    eef_link = group.get_end_effector_link()
    print "============ End effector: %s" % eef_link

    # We can get a list of all the groups in the robot:
    group_names = robot.get_group_names()
    print "============ Robot Groups:", robot.get_group_names()

    # Sometimes for debugging it is useful to print the entire state of the
    # robot:
    print "============ Printing robot state"
    print robot.get_current_state()
    print('Current pose:')
    print group.get_current_pose()
    print ""

    group.set_goal_tolerance(0.001)

    #group.set_pose_reference_frame('/r_wrist_roll_link')

    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.orientation.w = 1.0
    pose_goal.position.x = 0.4
    pose_goal.position.y = -0.2
    pose_goal.position.z = 0.5
    group.set_pose_target(pose_goal)

    ## Now, we call the planner to compute the plan and execute it.
    print('Going to pose')
    plan = group.go(wait=True)
    # Calling `stop()` ensures that there is no residual movement
    group.stop()
    print('Stopped')
    # It is always good to clear your targets after planning with poses.
    # Note: there is no equivalent function for clear_joint_value_targets()
    group.clear_pose_targets()
    print('Done.')
    print('Pose reference frame: ', group.get_pose_reference_frame())
    print('Tolerance: ', group.get_goal_joint_tolerance())
    print('Planning time: ', group.get_planning_time())
    print group.get_current_pose()

if __name__ == '__main__':
  main3()

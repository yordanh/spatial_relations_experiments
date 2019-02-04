import sys, time, collections
import copy, math
import rospy
import tf
import actionlib
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
import numpy as np

from pr2_picknplace_msgs.msg import PickPlaceAction, PickPlaceGoal
from pr2_head.srv import Query

pose_vec = collections.namedtuple('pose_vec', ['x', 'y', 'z', 'rx', 'ry', 'rz', 'rw'], verbose=False)

def quaternion2list(q):
    return [q.x, q.y, q.z, q.w]

def list2quaternion(l):
    assert len(l) == 4
    return geometry_msgs.msg.Quaternion(*l)

class PR2RobotController(object):
    """docstring for PR2RobotController"""
    def __init__(self, group_name, add_table=True):
        super(PR2RobotController, self).__init__()
        self.group_name = group_name
        
        ## First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)

        if rospy.client._init_node_args is None:
            rospy.init_node('delta_pose_mover', anonymous=True)

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander(self.group_name)


        # if group_name == "right_arm":
        #     plan_status = self.init_passive_arm("left_arm")
        #     print("PASSIVE {0} initiated:\t {1}".format("left_arm", plan_status))
        # elif group_name == "left_arm":
        #     plan_status = self.init_passive_arm("right_arm")
        #     print("PASSIVE {0} initiated:\t {1}".format("right_arm", plan_status))
        # else:
        #     print("WRONG GROUP NAME FOR MOVEIT")


        self.reference_frame = '/base_link'
        self.group.set_pose_reference_frame(self.reference_frame)
        self.group.set_goal_tolerance(0.001)

        self.group.set_planning_time(1)
        # self.group.set_max_velocity_scaling_factor(0.5)
        self.group.set_max_acceleration_scaling_factor(0.5)

        if (add_table):
            self.add_collision_table()
        else:
            self.remove_collision_table()


        self.left_arm_client = actionlib.SimpleActionClient(
            '/pr2_picknplace_left/pr2_picknplace', PickPlaceAction)
        self.left_arm_client.wait_for_server(rospy.Duration(30.0))
        rospy.loginfo("Waiting for left_arm action server")
        self.object_last_pose = [0.5, 0, 0.7]
        self.neutral_position_left = [0.2, 0.5, 0.9]
        self.neutral_position_right = [0.4, -0.6, 0.9]

    def init_passive_arm(self, passive_group_name):
        passive_group = moveit_commander.MoveGroupCommander(passive_group_name)
        reference_frame = '/base_link'
        passive_group.set_pose_reference_frame(reference_frame)
        passive_group.set_goal_tolerance(0.001)

        passive_group.set_planning_time(0.5)
        passive_group.set_max_acceleration_scaling_factor(0.75)

        pose_goal = geometry_msgs.msg.Pose()

        if passive_group_name == "left_arm":
            pose_goal.position.x = 0.5
            pose_goal.position.y = 0.03
            pose_goal.position.z = 0.8

            roll = 0
            # pitch = - math.pi / 2
            pitch = 0
            yaw = 0

        elif passive_group_name == "right_arm":
            pose_goal.position.x = 0.5
            pose_goal.position.y = -0.03
            pose_goal.position.z = 0.9

            roll = 0
            pitch = math.pi / 2
            yaw = 0

        qt = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        pose_goal.orientation.x = qt[0]
        pose_goal.orientation.y = qt[1]
        pose_goal.orientation.z = qt[2]
        pose_goal.orientation.w = qt[3]

        passive_group.set_pose_target(pose_goal)
        plan_success = passive_group.go(wait=True)

        return plan_success

    def add_collision_table(self):
        time.sleep(2)
        p = geometry_msgs.msg.PoseStamped()
        p.header.frame_id = self.robot.get_planning_frame()
        p.pose.position.x = 0.5
        p.pose.position.y = 0.
        p.pose.position.z = 0.725 + 0.01
        p.pose.orientation.w = 1
        self.scene.add_box("table", p, (0.75, 1.5, 0.025))

    def remove_collision_table(self):
        self.scene.remove_world_object("table")

    def reset_pose(self):

        speak = rospy.ServiceProxy('/pr2_head/say', Query)
        resp1 = speak("Reset")

        # pose_goal = geometry_msgs.msg.Pose()
        # roll = 0
        # pitch = 0
        # # yaw = math.pi/2
        # yaw = 0
        # qt = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        # pose_goal.orientation.x = qt[0]
        # pose_goal.orientation.y = qt[1]
        # pose_goal.orientation.z = qt[2]
        # pose_goal.orientation.w = qt[3]
        # pose_goal.position.x = self.neutral_position_right[0]
        # pose_goal.position.y = self.neutral_position_right[1]
        # pose_goal.position.z = self.neutral_position_right[2]

        # self.group.set_pose_target(pose_goal)
        # plan_success = self.group.go(wait=True)

        # time.sleep(1)

        # self.reset_object()

        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation.w = 1.0

        roll = 0
        pitch = 0
        yaw = math.pi/2
        qt = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        plan_success = False

        while not plan_success:
            # x_center = 0.55
            # y_center = 0
            x = np.random.uniform(low=0.3, high=0.7)
            y = np.random.uniform(low=-0.25, high=0.25)

            # wrist offset
            y -= 0.1

            print("X, Y", x, y)

            # 5cm buffer
            while((x > 0.65 and y < -0.1)):
                x = np.random.uniform(low=0.3, high=0.7)
                y = np.random.uniform(low=-0.25, high=0.25)

            # print(x, y)

            pose_goal.position.x = x
            pose_goal.position.y = y
            pose_goal.position.z = 0.8

            pose_goal.orientation.x = qt[0]
            pose_goal.orientation.y = qt[1]
            pose_goal.orientation.z = qt[2]
            pose_goal.orientation.w = qt[3]

            self.group.set_pose_target(pose_goal)
            plan_success = self.group.go(wait=True)

        self.group.stop()
        self.group.clear_pose_targets()
        # print("Reset to initial position {0}".format(pose_goal.position))


    def reset_object(self):
        print("RESET OBJECT PICK")
        goal = self.get_pickplace_goal(self.object_last_pose, request="pick")
        self.left_arm_client.send_goal_and_wait(goal, rospy.Duration(30))
        result = self.left_arm_client.get_result()
        print(result)

        time.sleep(2)

        result = False
        while not result:
            self.object_last_pose[0] = np.random.uniform(low=0.4, high=0.9)
            self.object_last_pose[1] = np.random.uniform(low=-0.3, high=0.3)

            print(self.object_last_pose)

            print("RESET OBJECT MOVETO")
            goal = self.get_pickplace_goal(self.object_last_pose, request="moveto")
            self.left_arm_client.send_goal_and_wait(goal, rospy.Duration(30))
            result = self.left_arm_client.get_result()
            print(result)
            time.sleep(2)

        print("RESET OBJECT PLACE")
        goal = self.get_pickplace_goal(self.object_last_pose, request="place")
        self.left_arm_client.send_goal_and_wait(goal, rospy.Duration(30))
        result = self.left_arm_client.get_result()
        print(result)
        time.sleep(2)

        print("RESET OBJECT MOVETO NEUTRAL")
        goal = self.get_pickplace_goal(self.neutral_position_left, request="moveto")
        self.left_arm_client.send_goal_and_wait(goal, rospy.Duration(30))
        result = self.left_arm_client.get_result()
        print(result)
        time.sleep(2)


    def get_pickplace_goal(self, position, request):
        pose = geometry_msgs.msg.Pose()
        pose.position.x = position[0]
        pose.position.y = position[1]
        pose.position.z = position[2]

        pose.orientation.x = 0
        pose.orientation.y = 0
        pose.orientation.z = 0
        pose.orientation.w = 1

        if request == "pick":
            request = 0
        elif request == "moveto":
            request = 2
        elif request == "place":
            request = 5

        result = PickPlaceGoal()
        result.goal.request = request
        result.goal.header.frame_id = "base_link"
        result.goal.object_pose = pose
        return result




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
        # print('Going to pose')
        success = self.group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        # self.group.stop()
        # self.group.clear_pose_targets()
        # print('Stopped')
        return success

    def move_delta(self, d):
        delta = geometry_msgs.msg.Pose()

        for k in ['x', 'y', 'z']:
            setattr(delta.orientation, k, getattr(d, 'r' + k))
            setattr(delta.position, k, getattr(d, k))
        delta.orientation.w = d.rw

        return self._move_delta_geom_pose(delta)

    def move_delta_t_rpy(self, t, rpy):
        assert len(t) == 3
        assert len(rpy) == 3

        qt = tf.transformations.quaternion_from_euler(rpy[0], rpy[1], rpy[2])
        delta = pose_vec(t[0], t[1], t[2], qt[0], qt[1], qt[2], qt[3])

        return self.move_delta(delta)

    def move_delta_t(self, t):
        return self.move_delta_t_rpy(t, [0, 0, 0])


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
        success = pr2.move_delta_t_rpy(delta_t, delta_rpy)
    if (not success):
        print('Couldn\'t move to that pose')


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
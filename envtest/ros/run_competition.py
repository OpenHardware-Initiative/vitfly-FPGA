#!/usr/bin/python3
import argparse

import rospy
from dodgeros_msgs.msg import Command
from dodgeros_msgs.msg import QuadState
from cv_bridge import CvBridge
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Empty
from std_msgs.msg import String

from envsim_msgs.msg import ObstacleArray

# from rl_example import load_rl_policy
from user_code_FPGA import compute_command_vision_based, compute_command_state_based
from utils import AgileCommandMode, AgileQuadState

import time
import numpy as np
import pandas as pd
import os, sys
from os.path import join as opj
from copy import deepcopy
import cv2
import torch

sys.path.append(opj(os.path.dirname(os.path.abspath(__file__)), '../../models'))
from model import *

class AgilePilotNode:
    def __init__(self, vision_based=False, model_type=None, model_path=None, desVel=None, keyboard=False):
        print("[RUN_COMPETITION] Initializing agile_pilot_node...")
        rospy.init_node("agile_pilot_node", anonymous=False)

        self.vision_based = vision_based
        self.rl_policy = None
        self.publish_commands = False
        self.cv_bridge = CvBridge()
        self.state = None
        self.keyboard = keyboard

        quad_name = "kingfisher"

        self.init = 0
        self.col = None
        self.t1 = 0 #Time flag
        self.timestamp = 0 #Time stamp initial
        self.last_valid_img = None #Image that will be logged
        data_log_format = {'timestamp':[],
                           'desired_vel':[],
                           'quat_1':[],
                           'quat_2':[],
                           'quat_3':[],
                           'quat_4':[],
                           'pos_x':[],
                           'pos_y':[],
                           'pos_z':[],
                           'vel_x':[],
                           'vel_y':[],
                           'vel_z':[],
                           'velcmd_x':[],
                           'velcmd_y':[],
                           'velcmd_z':[],
                           'ct_cmd':[],
                           'br_cmd_x':[],
                           'br_cmd_y':[],
                           'br_cmd_z':[],
                           'is_collide': [],
        } 
        self.data_log = pd.DataFrame(data_log_format) # store in the data frame
        self.count = 0 # counter for the csv
        
        # @NOTE: Dont log too fast, I have not tested that
        self.time_interval = .03 #Time interval for logging

        self.data_collection_xrange = [2, 60]

        # make the folder for the epoch
        self.folder = f"train_set/{int(time.time()*100)}" 
        os.mkdir(self.folder)

        self.desiredVel = desVel #self.readVel("velocity.txt") #np.random.uniform(low=2.0, high=3.0)
        print()
        print(f"[RUN_COMPETITION] Desired velocity = {self.desiredVel}")
        print()

        # load trained model here (copied over from user_code.py)
        if model_path is not None:
            print(f"[RUN_COMPETITION] Model loading from {model_path} ...")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if model_type == 'LSTMNet':
                self.model = LSTMNet().to(self.device).float()
            elif model_type == 'UNetLSTM':
                self.model = UNetConvLSTMNet().to(self.device).float()
            elif model_type == 'ConvNet':
                self.model = ConvNet().to(self.device).float()                
            elif model_type == 'ViT':
                self.model = ViT().to(self.device).float()
            elif model_type == 'ViTLSTM':
                self.model = LSTMNetVIT().to(self.device).float()                
            else:
                print(f'[RUN_COMPETITION] Invalid model_type {model_type}. Exiting.')
                exit()

            # Give full path if possible since the bash script runs from outside the folder
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()

            # Initialize hidden state
            self.model_hidden_state = None

            print(f"[RUN_COMPETITION] Model loaded")
            time.sleep(2)

        self.start_time = 0
        self.logged_time_flag = 0
        self.depth_im_threshold = 0.09

        self.curr_cmd = None

        # Logic subscribers
        self.start_sub = rospy.Subscriber(
            "/" + quad_name + "/start_navigation",
            Empty,
            self.start_callback,
            queue_size=1,
            tcp_nodelay=True,
        )

        # Observation subscribers
        self.odom_sub = rospy.Subscriber(
            "/" + quad_name + "/dodgeros_pilot/state",
            QuadState,
            self.state_callback,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.img_sub = rospy.Subscriber(
            "/" + quad_name + "/dodgeros_pilot/unity/depth",
            Image,
            self.img_callback,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.obstacle_sub = rospy.Subscriber(
            "/" + quad_name + "/dodgeros_pilot/groundtruth/obstacles",
            ObstacleArray,
            self.obstacle_callback,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.cmd_sub = rospy.Subscriber(
            "/" + quad_name + "/dodgeros_pilot/command",
            Command,
            self.cmd_callback,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.keyboard_sub = rospy.Subscriber(
            "/keyboard_input",
            String,
            self.keyboard_callback,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.rgb_img_sub = rospy.Subscriber(
            "/" + quad_name + "/dodgeros_pilot/unity/image",
            Image,
            self.rgb_callback,
            queue_size=1,
            tcp_nodelay=True,
        )


        # Command publishers
        self.cmd_pub = rospy.Publisher(
            "/" + quad_name + "/dodgeros_pilot/feedthrough_command",
            Command,
            queue_size=1,
        )
        self.linvel_pub = rospy.Publisher(
            "/" + quad_name + "/dodgeros_pilot/velocity_command",
            TwistStamped,
            queue_size=1,
        )
        self.debug_img1_pub = rospy.Publisher(
            "/debug_img1",
            Image,
            queue_size=1,
        )
        self.debug_img2_pub = rospy.Publisher(
            "/debug_img2",
            Image,
            queue_size=1,
        )
        print("[RUN_COMPETITION] Initialization completed!")

        self.ctr = 0

        self.keyboard_input = ''
        self.got_keypress = 0.0
        self.rgb_img = None

    def rgb_callback(self, img):
        self.rgb_img = self.cv_bridge.imgmsg_to_cv2(img, desired_encoding="passthrough")

    def cmd_callback(self, msg):
        self.curr_cmd = msg

    def keyboard_callback(self, msg):
        self.got_keypress = rospy.Time().now().to_sec()
        self.keyboard_input = msg.data

    def readVel(self,file):
        with open(file,"r") as f:
            x = f.readlines()
            for i in range(len(x)):
                if i == 0:
                    return float(x[i].split("\n")[0])

    def img_callback(self, img_data):
        self.ctr += 1
        self.prevImg = deepcopy(self.last_valid_img)
        img = self.cv_bridge.imgmsg_to_cv2(img_data, desired_encoding="passthrough")
        img = np.clip(img/self.depth_im_threshold, 0, 1)
                
        if self.prevImg is None:
            self.prevImg = img

        self.last_valid_img = deepcopy(img) if img.min() > 0.0 else self.last_valid_img
        
        
        
        
        

        if not self.vision_based:
            return
        
        if self.state is None:
            return
        
        print('[RUN_COMPETITION] calling compute_command_vision_based')
        start_compute_time = time.time()

        command, (debug_img1, debug_img2), self.model_hidden_state = compute_command_vision_based(self.state, img, self.prevImg,self.desiredVel, self.model, self.model_hidden_state)

        # publish debug images
        self.debug_img1_pub.publish(self.cv_bridge.cv2_to_imgmsg(debug_img1, encoding="passthrough"))
        self.debug_img2_pub.publish(self.cv_bridge.cv2_to_imgmsg(debug_img2, encoding="passthrough"))

        if self.ctr % 30 == 0:
            print(f'[RUN_COMPETITION] compute_command_vision_based took {time.time() - start_compute_time} seconds')

        self.publish_command(command)
        print(f'[RUN_COMPETITION] output: {command.velocity}')
        print(f'[RUN_COMPETITION] output FULL COMMAND: {command}')

        if self.state.pos[0] < 0.1:
            self.start_time = command.t

        if self.state.pos[0] >= 60 and self.logged_time_flag == 0:
            file = "timeTaken.dat"
            with open(file, "a") as file:
                file.write(str(float(command.t - self.start_time))+"\n")
            self.logged_time_flag = 1
        
        #if we exceed the time interval then save the data
        if (self.state.t - self.t1 > self.time_interval or self.t1==0) and self.state.pos[0] < 63:
            #reset the time flag
            self.t1 = self.state.t

            # Get the current time stamp - instant
            timestamp = round(
                self.state.t, 3
            )  # If you need more hz, you might need to modify this round

            # Save the image by the name of that instant
            cv2.imwrite(f"{self.folder}/{str(timestamp)}.png", (self.last_valid_img*255).astype(np.uint8))

            # Get the collision flag
            if self.col is None:
                self.col = 0
            # Append the data frame
            # @TODO: This needs to be managed better if the number of datapoints exceeds 10,000
            self.data_log.loc[len(self.data_log)] = [
                timestamp,
                self.desiredVel,
                self.state.att[0],
                self.state.att[1],
                self.state.att[2],
                self.state.att[3],
                self.state.pos[0],
                self.state.pos[1],
                self.state.pos[2],
                self.state.vel[0],
                self.state.vel[1],
                self.state.vel[2],
                command.velocity[0],
                command.velocity[1],
                command.velocity[2],
                self.curr_cmd.collective_thrust,
                self.curr_cmd.bodyrates.x,
                self.curr_cmd.bodyrates.y,
                self.curr_cmd.bodyrates.z,
                self.col,
            ]

            # Counter flag for saving the data frame
            self.count += 1

        # Save once every 10 instances - writing every instance can be expensive
        if self.count % 5 == 0:
            self.data_log.to_csv(self.folder + "/data.csv")

    def state_callback(self, state_data):
        self.state = AgileQuadState(state_data)

    def obstacle_callback(self, obs_data):
        if self.state is None:
            return
        self.col = self.if_collide(obs_data.obstacles[0])
        if self.vision_based:
            return
        if self.rgb_img is None:
            print("no rgb image yet")
            return

        # try:
        #     self.desiredVel = self.readVel("velocity.txt") #Changed some thing
        # except:
        #     pass
        # usable keypress?
        if rospy.Time().now().to_sec() - self.got_keypress > 0.1:
            self.keyboard_input = ''

        command = compute_command_state_based(
            state=self.state,
            obstacles=obs_data,
            desiredVel=self.desiredVel,
            rl_policy=self.rl_policy,
            keyboard=self.keyboard,
            keyboard_input=self.keyboard_input,
        )
        self.publish_command(command)

        if self.state.pos[0] < 0.1:
            self.start_time = command.t
        if self.state.pos[0] >= 60 and self.logged_time_flag == 0:
            file = "timeTaken.dat"
            with open(file, "a") as file:
                file.write(str(float(command.t - self.start_time))+"\n")
            self.logged_time_flag = 1
        
        # if we exceed the time interval then save the data
        if (self.state.t - self.t1 > self.time_interval or self.t1 == 0 or self.col) and (self.state.pos[2] > 2.95 or self.init == 1):
            
            self.init = 1

            if self.state.pos[0] > self.data_collection_xrange[0] and self.state.pos[0] < self.data_collection_xrange[1]:

                # reset the time flag
                self.t1 = self.state.t

                # Get the current time stamp
                timestamp = round(self.state.t, 3)  # If you need more hz, you might need to modify this round

                # Save the image by the name of that instant
                # np.save(self.folder + f"/im_{timestamp}", self.last_valid_img)
                cv2.imwrite(f"{self.folder}/{str(timestamp)}.png", (self.last_valid_img*255).astype(np.uint8))
                cv2.imwrite(f"{self.folder}/{str(timestamp)}_rgb.png", (self.rgb_img*255).astype(np.uint8))

                # Get the collision flag
                col = self.if_collide(obs_data.obstacles[0])
                # Append the data frame
                # @TODO: This needs to be managed better if the number of datapoints exceeds 10,000
                self.data_log.loc[len(self.data_log)] = [
                    timestamp,
                    self.desiredVel,
                    self.state.att[0],
                    self.state.att[1],
                    self.state.att[2],
                    self.state.att[3],
                    self.state.pos[0],
                    self.state.pos[1],
                    self.state.pos[2],
                    self.state.vel[0],
                    self.state.vel[1],
                    self.state.vel[2],
                    command.velocity[0],
                    command.velocity[1],
                    command.velocity[2],
                    self.curr_cmd.collective_thrust,
                    self.curr_cmd.bodyrates.x,
                    self.curr_cmd.bodyrates.y,
                    self.curr_cmd.bodyrates.z,
                    self.col,
                ]

                # Counter flag for saving the data frame
                self.count += 1

        # Save once every 10 instances - writing every instance can be expensive
        if self.count % 2 == 0 and self.count != 0 or abs(self.state.pos[0] - 20) < 1:
            self.data_log.to_csv(self.folder + "/data.csv")

    def if_collide(self, obs):
        """
        Borrowed and modified from evaluation_node
        """

        dist = np.linalg.norm(
            np.array([obs.position.x, obs.position.y, obs.position.z])
        )
        margin = dist - obs.scale
        # Ground hit condition
        if margin < 0 or self.state.pos[2] <= 0.01:
            hit_obstacle = True
        else:
            hit_obstacle = False

        return hit_obstacle

    def publish_command(self, command):
        if command.mode == AgileCommandMode.SRT:
            assert len(command.rotor_thrusts) == 4
            cmd_msg = Command()
            cmd_msg.t = command.t
            cmd_msg.header.stamp = rospy.Time(command.t)
            cmd_msg.is_single_rotor_thrust = True
            cmd_msg.thrusts = command.rotor_thrusts
            if self.publish_commands:
                self.cmd_pub.publish(cmd_msg)
                return
        elif command.mode == AgileCommandMode.CTBR:
            assert len(command.bodyrates) == 3
            cmd_msg = Command()
            cmd_msg.t = command.t
            cmd_msg.header.stamp = rospy.Time(command.t)
            cmd_msg.is_single_rotor_thrust = False
            cmd_msg.collective_thrust = command.collective_thrust
            cmd_msg.bodyrates.x = command.bodyrates[0]
            cmd_msg.bodyrates.y = command.bodyrates[1]
            cmd_msg.bodyrates.z = command.bodyrates[2]
            if self.publish_commands:
                self.cmd_pub.publish(cmd_msg)
                return
        elif command.mode == AgileCommandMode.LINVEL:
            vel_msg = TwistStamped()
            vel_msg.header.stamp = rospy.Time(command.t)
            vel_msg.twist.linear.x = command.velocity[0]
            vel_msg.twist.linear.y = command.velocity[1]
            vel_msg.twist.linear.z = command.velocity[2]
            vel_msg.twist.angular.x = 0.0
            vel_msg.twist.angular.y = 0.0
            vel_msg.twist.angular.z = command.yawrate
            if self.publish_commands:
                self.linvel_pub.publish(vel_msg)
                return
        else:
            assert False, "Unknown command mode specified"

    def start_callback(self, data):
        print("[RUN_COMPETITION] Start publishing commands!")
        self.publish_commands = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agile Pilot.")
    parser.add_argument("--vision_based", help="Fly vision-based", required=False, dest="vision_based", action="store_true")
    parser.add_argument('--model_type', type=str, default='ViTLSTM', help='string matching model name in lstmArch.py')
    parser.add_argument('--model_path', type=str, default=None, help='absolute path to model checkpoint')
    parser.add_argument('--des_vel', type=float, default=None, help='desired velocity for quadrotor')
    parser.add_argument("--keyboard", help="Fly state-based mode but take velocity commands from keyboard WASD", required=False, dest="keyboard", action="store_true")

    args = parser.parse_args()
    agile_pilot_node = AgilePilotNode(vision_based=args.vision_based, model_type=args.model_type, model_path=args.model_path, desVel=args.des_vel, keyboard=args.keyboard)
    rospy.spin()

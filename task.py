import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
    
class TaskTakeOff():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        self.init_position = init_pose[:3] if init_pose is not None else np.array([0., 0., 0.])
        self.orig_distances = abs(self.target_pos - self.init_position)
        #print("target_pos = {} init_position = {} orig_distances = {}".format(self.target_pos, self.init_position, self.orig_distances ))

    def get_reward_orig(self):
        """Uses current pose of sim to return reward."""
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward
    
    def get_reward_basic(self):
        """Uses current pose of sim to return reward."""
        #indices# X=0, Y=1, Z=2, PSI=3, THETA=4, PHI=5
        X = self.sim.pose[0]
        Y = self.sim.pose[1]
        Z = self.sim.pose[2]
        PSI   = self.sim.pose[3]
        THETA = self.sim.pose[4]
        PHI   = self.sim.pose[5] 
        #set rewards
        #positive reward for positive Z, penalties for X & Y motion
        #Euler angles and velocities are neglected as their effects 
        #will reflect upon the (x,y,z) coordinates anyway and will then be 
        #followed through.
        reward =  0.0 + 1*(Z) - 0.3*(abs(X)+abs(Y))
        #End episode if max height is breached 
        done = False
        if self.sim.pose[2] >= self.target_pos[2]: # agent has crossed the target height
            #raise TypeError
            reward += 50.0  # bonus reward
            done = True
        #normalize reward to avoid overflows in other computations such as gradients
        #Since we will end an episode once it reaches a height of 10, and give it a bonus reward of 50
        #the max positice reward is 60:
        #  as (0,0,10) -> reward calculated above of 10 + 50 as bonus
        #to normalize we will consider the min value to be -60 
        Z = (reward + 60) / (120) # Z = (value - min)/(max - min), here (max - min)=(60-(-60)) = 120 always
        reward = Z
        return reward, done
    
    def get_reward_complex(self):
        """Uses current pose of sim to return reward."""
        #indices of self.sim.pose --> X=0, Y=1, Z=2, PSI=3, THETA=4, PHI=5
        X = self.sim.pose[0]
        Y = self.sim.pose[1]
        Z = self.sim.pose[2]
        PSI   = self.sim.pose[3]
        THETA = self.sim.pose[4]
        PHI   = self.sim.pose[5] 
        body_velocity = self.sim.find_body_velocity()
        #print("body_velocity.type={}".format(type(body_velocity))
        #set rewards
        euler_angle_sum = abs(PSI)+abs(THETA)+abs(PHI)
        reward = 100.0
        reward -= 0.3 * (abs(X)+abs(Y)) #penalty for movement along x and y planes
        reward -= 0.5 * euler_angle_sum #penalty for euler angle movement
        return round(reward,2)
    
    def get_reward_complex2(self):
        """Uses current pose of sim to return reward."""
        #indices of self.sim.pose --> X=0, Y=1, Z=2, PSI=3, THETA=4, PHI=5
        X = self.sim.pose[0]
        Y = self.sim.pose[1]
        Z = self.sim.pose[2]
        PSI   = self.sim.pose[3]
        THETA = self.sim.pose[4]
        PHI   = self.sim.pose[5] 
        body_velocity = self.sim.find_body_velocity()
        #print("body_velocity.type={}".format(type(body_velocity))
        #set rewards 
        euler_angle_sum = abs(PSI)+abs(THETA)+abs(PHI)
        #positive reward for positive Z movement
        #penalty for movement along x and y planes
        reward =  0.0 + 1*(Z) - 0.3*(abs(X)+abs(Y))
        reward -= 0.5 * euler_angle_sum #penalty for euler angle movement
        return round(reward,2)
    
    get_reward = get_reward_basic  #switch to select the particular get_reward needed.

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            #check if episode is done due to desired height being reached
            delta_reward, done_height  = self.get_reward() 
            reward += delta_reward
            if done_height:
                done = done_height
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
import numpy as np
from physics_sim import PhysicsSim

class Task():
    """
    Task (environment) that defines the goal and provides feedback to the agent.
    
    Note: 
    toDo: create a subclass TakeOff_task for this project to get the proper reward including penalty handling
    """
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """ Initialize a Task object.
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

        
    def get_TakeOff_reward(self):
        """ 
        for TakeOff_task: 
        the device shall start from the ground and reach a given target height (z value).
        
        Function creates the reward and 
        cheques if target height position z is reached;
        regarding penalties:
        - in general the take-off process should happen only a specific amount of time,
          this is not taken into account, so, no penalty created for that
        - additionally changing the positions often because of drift issue
          should lead to penalty as well, but is not taken into account either
        """
        # the device agent has reached the target height, means take-off is finished,
        # and bonus reward is given to the existing value only during that process;
        # because velocity is important for having a stable flight, it is included
        # self.sim.v is used because self.sim.init_velocities can be none
        # because we are interested in the z position, the psi velocity is relevant
        if self.sim.pose[2] > self.target_pos[2]:
            reward = -20.0 * 1/3 * self.sim.v[2]
        else:
            reward = 30.0 * 2/3 * self.sim.v[2]
            
        # penalty situation: agent runs out of time for take-off process
        # toDo: situation could be handled with specific TakeOff_task subclass
        #if timestamp > self.max_duration: 
            #reward = -20.0 
            #done = True
            
        # penalty situation: drift behaviour let the device be flying unstable           
        
        return reward

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        
        if self.name == "take-off":
            reward = self.get_TakeOff_reward()
        
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

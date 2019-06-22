import numpy as np
from collections import defaultdict
from agents.Model import Model
from agents.OUNoise import OUNoise
from agents.ReplayBuffer import ReplayBuffer

class DDPG_Agent:
    """ RL agent using actor-critic method with DDPG concept (Deep Deterministic Policy Gradient) """
   
    def __init__(self, task, alpha=0.0001, beta=0.001, gamma=0.9, tau=0.125 , mu=0.2,
                 theta=0.15, sigma=0.2, max_size=100000):
        """ Initialize DDPG agent.

        Params
        ======
        - task: the task that shall be learned (for the quadcopter project: take-off or landing)
                implemented is the take-off task which uses only the PhysicsSim pose information:
                from the quadcopter position delivered by such physical basic class only the target height
                (means the z value) is relevant; if its value is reached the episode termination condition
                is valid
        - alpha: actor learning rate
        - beta: critic learning rate
        - gamma: discount value, for termination function: final state with gamma(s)=0
        - tau: used for smooth shifting from the local prediction models to the target models (soft update of targets)
        - mu, theta, sigma: for creation of Ornstein-Uhlenbeck noise instance, part of exploration policy creation
        - max_size: memory buffer size
        """
        
        #
        # set hyperparameters
        #

        self.task = task      
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        self.gamma = gamma   # algorithm discount value
        self.tau = tau       # algorithm soft update of targets
        self.actor_learning_rate = alpha
        self.critic_learning_rate = beta
        
        # 
        # networks creation
        #
        
        # networks of actor and critic components
        self.network_name = "DQNetwork"
        self.target_name = "TargetNetwork"
        self.actor = Model(self.task, "actor", self.network_name).get_model()
        self.actor_target = Model(self.task, "actor", self.target_name).get_model()
        self.critic = Model(self.task, "critic", self.network_name).get_model()
        self.critic_target = Model(self.task, "critic", self.target_name).get_model()
                
        #
        # training resp. learning part
        #
        
        # set memory
        self.buffer_size = max_size
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # set noise
        # Ornstein-Uhlenbeck noise process (as mentioned in the DDPG paper)
        # for construction of exploration policy, 
        # by adding noise sampled from a noice process N to the actor policy
        # in the paper supplementary as param values theta=0.15 and sigma=0.2 are used, shall be default values
        self.exploration_mu = mu
        self.exploration_theta = theta
        self.exploration_sigma = sigma
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)  
        
    def get_memory(self):
        return self.memory
    
    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        return state
    
    
    def step(self, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state

        
    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor.model.predict(state)[0]
        return list(action + self.noise.get_noise_sample())  # add some noise for exploration

        
    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor.train_func([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic.model, self.critic_target.model)
        self.soft_update(self.actor.model, self.actor_target.model)   

        
    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
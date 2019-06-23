from agents.DDPG_agent import DDPG_Agent

class Agent:
    """ Factory class of agent types. """
    
    def __init__(self, task=None, type_name=None, params=None):
        """ Initialise the factory agent. 

        Params
        ======
        - task: the task that shall be learned (for the quadcopter project: takeoff or landing),
                delivers e.g. number of actions and states available to the agent
        """
        self.agent = None
        self.task = task
        self.memory = None
        self.type_name = type_name
        
        if self.type_name in ["DDPG"]:
            if params is None:
                self.agent = DDPG_Agent(task=self.task)
            else:
                self.agent = DDPG_Agent(task=self.task,
                                        alpha=params['alpha'], beta=params['beta'],
                                        gamma=params['gamma'], tau=params['tau'],
                                        mu=params['mu'], theta=params['theta'], sigma=params['sigma'],
                                        max_size=params['max_size'])
                
            self.memory = self.agent.get_memory()
        else:
            print("Wrong Agent type - {} -, does not exist, therefore no agent instance building possible.".format(type_name))
        
     
    def get_agent(self):
        # Can return None, if agent type does not exist yet.
        return self.agent
    
    def get_memory(self):
        # Can return None; for this Quadcopter project status it is DDPG ReplayBuffer instance 
        return self.memory   

    def get_params(self):
        # Can return None; for this Quadcopter project status it is DDPG ReplayBuffer instance 
        return self.params
    
        
    

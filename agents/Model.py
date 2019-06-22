from agents.Actor import Actor
from agents.Critic import Critic

class Model:
    """ Model factory class for actor-critic local network and target models. """
   
    def __init__(self, task=None, type_name=None, name=None):
        '''
        Factory class for actor-critic models
        '''
        self.model = None
        self.name = name
        self.task = task
        self.type_name = type_name
        
        if type_name in ["actor"]:
            self.model = Actor(state_size=self.task.state_size, action_size=self.task.action_size,
                               action_low=task.action_low, action_high=task.action_high, name=name)
        elif type_name in ["critic"]: 
            self.model = Critic(state_size=self.task.state_size, action_size=self.task.action_size, name=name)
        else:
            print("Wrong Model type - {} -, does not exist, therefore no model building possible.".format(type_name))

            
    def get_model(self):
        # Can return None, if model type does not exist yet.
        return self.model
    
           
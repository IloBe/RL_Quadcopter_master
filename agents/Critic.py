from keras import backend as K
from keras import optimizers, layers, models

class Critic:
    """
    Critic class initialises parameters and
    builds models with Keras (see documentation on https://keras.io/);
    the models input state-action pairs are mapped to output Q-values.
    """
    
    def __init__(self, state_size, action_size, beta=0.001, name=None):
        """Initialize parameters and model"""
        self.model_name = name
        self.critic_learning_rate = beta
        self.state_size = state_size    # integer - state space
        self.action_size = action_size  # integer - action space
        
        self.model = _build_model()
       
    
    def _build_model():
        """ 
        Creates critic network for mapping state-action pairs to Q-values. 
        
        Because as input state-action pairs are necessary,
        compared to the Actor class here the other Model creation process is used.
        """
        
        # layer foundation inputs 
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')
        
        # Hidden layers for states
        net_states = layers.Dense(units=32)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation(activation='relu')(net_states)
        net_states = layers.Dropout(rate=0.3)(net_states)
        net_states = layers.Dense(units=64, activation='relu')(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation(activation='relu')(net_states)
        net_states = layers.Dropout(rate=0.4)(net_states)
        net_states = layers.Dense(units=128, activation='relu')(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation(activation='relu')(net_states)
        net_states = layers.Dropout(rate=0.5)(net_states)
            
        # Hidden layers for actions
        net_actions = layers.Dense(units=32)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Activation(activation='relu')(net_actions)
        net_actions = layers.Dropout(rate=0.3)(net_actions)
        net_actions = layers.Dense(units=64)(net_actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Activation(activation='relu')(net_actions)
        net_actions = layers.Dropout(rate=0.4)(net_actions)
        net_actions = layers.Dense(units=128)(net_actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Activation(activation='relu')(net_actions)
        net_actions = layers.Dropout(rate=0.4)(net_actions)
            
        # Combine state and action values
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation(activation='relu')(net)
            
        # Final output layer for Q-values
        Q_vals = layers.Dense(units=1, name='q_vals')(net)
            
        # Create model
        model = models.Model(inputs=[states, actions], outputs=Q_vals)
          
        # Learning configuration: compile with Adam optimiser and mean squared error as loss function
        model.compile(optimizer=optimizers.Adam(lr=self.critic_learning_rate), loss='mse')
            
        # Compute action gradients (derivative of Q values with respect to actions)
        action_grads = K.gradients(Q_vals, actions)
            
        # With Keras backend concept create function to get action gradients to be used by actor model
        self.get_action_gradients = K.function(inputs=[*self.model.input, K.learning_phase()],
                                               outputs=action_grads)
        
        return model
    
    
    def get_model():
        return self.model
    
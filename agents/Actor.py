from keras import backend as K
from keras import optimizers
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Dropout, BatchNormalization


class Actor:
    """
    Actor class initialises parameters and
    builds models with Keras (see documentation on https://keras.io/);
    the models input states are mapped to output actions,
    they are used to update the policy function π(a|s,θ)
    
    for general CNN see:
    https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
    
    specific model information:
    states are given as input for the input layer
    in the hidden layers, 
        for stabilisation BatchNormalization() instances and
        to avoid overfitting Dropout() instances are added
    the output layer delivers one node for each action of the action space
    and
    because of using a sigmoid activation function, its output range [0, 1] is scaled
    
    the project implementation is a regression issue, therefore 
        the learning process is configured with Adam as optimiser
        and mean backend loss function
    """
    
    def __init__(self, state_size, action_size, action_low, action_high, alpha=0.0001, name=None):
        self.model_name = name
        self.actor_learning_rate = alpha
        self.state_size = state_size     # integer - state space
        self.action_size = action_size   # integer - action space
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low       
        self.model = _build_model()
        
    def _scale_output():
        return Lambda(lambda x: (x * self.action_range) + self.action_low, name='output_actions')
        
    def _build_model():
        """ Create actor network that maps states to actions. """
        
        # layer foundation 
        model = Sequential()

        # input layer for 'input_states', starting with one node for each state
        # and output arrays of shape (*, 16)
        model.add(Dense(16, input_dim = self.state_size, activation = "relu")) 
        
        # hidden layers
        # BatchNormalization included to improve performance and stability, and it has to be included
        # after each fully-connected layer, but before the activation function and dropout
        # Dropout layers are added to avoid overfitting
        
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation(activation = "relu"))                                
        model.add(Dropout(0.3))
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation(activation = "relu"))                                
        model.add(Dropout(0.4))
        model.add(Dense(128))
        model.add(BatchNormalization())
        model.add(Activation(activation = "relu"))                                
        model.add(Dropout(0.5))
        
        # Output Layer with number of actions, one node for each action of the action space;
        # because of the sigmoid activation function, the output range is [0, 1]
        model.add(Dense(self.action_space, activation="sigmoid"))
        # Scale [0, 1] output for each action dimension to proper range for each action dimension.
        # This produces a deterministic action for any given state vector.
        # A noise will be added later to this action to produce some exploratory behaviour.
        output_actions = model.add(function = _scale_output())
                  
        # print model information
        model.summary()
        
        # create the model: before training, the learning process has to be configured;
        # the target variable is continuous, for that most widely used regressive loss function is Mean Square Error
        # as optimisation function for network weight modification Adam is selected
        # 
        # in general, learning process configuration would be done by compile(), like
        # model.compile(loss = "mean_squared_error", optimizer = opt_adam)
        # but now 
        # to realise the policy gradient another approach is implemented:
        # for training use the Keras backend concept: keras.backend.function(inputs, outputs, updates=None)
        # with a loss function of Q value gradients 
        # note: Keras Backend delivers a huge range of functions, like mean, which we can define for our purpose 
        actor_weights = self.model.trainable_weights
        opt_adam = optimizers.Adam(lr = self.actor_learning_rate)
        # Loss function using Q-value gradients
        action_grads = layers.Input(shape=(self.action_size,))
        loss_func = K.mean(-action_grads * output_actions)
        updates_opt = optimizer.get_updates(params=actor_weights, loss=loss_func)
        self.train_func = K.function(inputs=[model.input, action_grads, K.learning_phase()],
                                     outputs=[], updates=updates_opt)
        
        return model
 

    def get_model():
        return self.model
 
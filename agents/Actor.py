from keras import backend as K
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Lambda, Dense, BatchNormalization, Activation, Dropout, Input

class Actor:
    """
    Actor (policy) class initialises parameters and
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
        self.build_model()       

           
    def build_model(self):
        """ Creates actor (policy) network that maps states to actions. """
        
        # layer foundation 
        self.model = Sequential()

        # input layer for 'input_states', starting with one node for each state
        # and output arrays of shape (*, 16)
        self.model.add(Dense(16, input_dim = self.state_size, activation = "relu")) 
        
        # hidden layers
        # BatchNormalization included to improve performance and stability, and it has to be included
        # after each fully-connected layer, but before the activation function and dropout
        # Dropout layers are added to avoid overfitting
        
        self.model.add(Dense(32))
        self.model.add(BatchNormalization())
        self.model.add(Activation(activation = "relu"))                                
        self.model.add(Dropout(0.3))
        self.model.add(Dense(64))
        self.model.add(BatchNormalization())
        self.model.add(Activation(activation = "relu"))                                
        self.model.add(Dropout(0.4))
        self.model.add(Dense(128))
        self.model.add(BatchNormalization())
        self.model.add(Activation(activation = "relu"))                                
        self.model.add(Dropout(0.5))
        
        # Output Layer with number of actions, one node for each action of the action space;
        # because of the sigmoid activation function, the output range is [0, 1]
        self.model.add(Dense(self.action_size, activation="sigmoid"))
        # Scale [0, 1] output for each action dimension to proper range for each action dimension.
        # This produces a deterministic action for any given state vector.
        # A noise will be added later to this action to produce some exploratory behaviour.
        self.model.add(Lambda(lambda x: (x * self.action_range) + self.action_low, name='output_actions'))
        
        # print model information
        print("\n--- Actor ---")
        print("How much layers has the model?  {}".format(len(self.model.layers))) 
        
        output_actions = self.model.layers[len(self.model.layers)-1].output
        # to show the output_actions: import tensorflow as tf
        #output_actions = output_actions.eval(session=tf.Session())
        #print("output_actions:\n{}".format(output_actions))

        print("--- Build model summary of {}: ---".format(self.model_name))
        self.model.summary()
        
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
        opt_adam = Adam(lr = self.actor_learning_rate)
        # Loss function using Q-value gradients
        action_grads = Input(shape=(self.action_size,))
        
        #print("--- Actor model method:  K.mean(-action_grads * output_actions) ---")
        #print("action_grads:\n{}".format(action_grads))
        #print("\noutput_actions:\n{}\n".format(output_actions))
        
        loss_func = K.mean(-action_grads * output_actions)
        updates_opt = opt_adam.get_updates(params=actor_weights, loss=loss_func)
        self.train_func = K.function(inputs=[self.model.input, action_grads, K.learning_phase()],
                                     outputs=[], updates=updates_opt)
  

    def get_model(self):     
        return self.model
 

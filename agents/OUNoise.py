# see: https://calculushowto.com/what-is-the-ornstein-uhlenbeck-process/

import numpy as np
import copy

class OUNoise:
    '''
    Calculation of Ornstein-Uhlenbeck process
    OU Process = dxt = θ(μ – xt)dt + σ dWt

    Where:

        xt = the particle’s current position
        θ = a mean reversion constant
        μ = the mean particle position
        σ = a constant volatility
        dWt = a Wiener process (Brownian motion)
    '''

    def __init__(self, size = 1, mu = 0, theta = 0.15, sigma = 0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def get_noise_sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        
        return self.state

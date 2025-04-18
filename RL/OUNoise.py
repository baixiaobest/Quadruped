import numpy as np
import matplotlib.pyplot as plt

class OUNoise:
    def __init__(self, theta=0.15, mu=0.0, sigma=0.2, dt=1e-2, x_initial=None):
        """
        Ornstein-Uhlenbeck process implementation
        
        Parameters:
            theta (float): Rate of mean reversion
            mu (float/array): Long-term mean
            sigma (float/array): Volatility/noise scale
            dt (float): Time step size
            x_initial (array): Initial value (defaults to mu)
        """
        self.theta = theta
        self.mu = np.array(mu, dtype=np.float64)
        self.sigma = sigma
        self.dt = dt
        self.x_prev = np.copy(self.mu) if x_initial is None else np.array(x_initial, dtype=np.float64)
        
    def __call__(self):
        """Generate next noise value"""
        return self.sample()
    
    def sample(self):
        """Generate next noise sample"""
        dx = self.theta * (self.mu - self.x_prev) * self.dt
        dx += self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev += dx
        return np.copy(self.x_prev)
    
    def set_noise(self, noise):
        self.x_prev = noise
    
    def reset(self):
        """Reset the process state to mean"""
        self.x_prev = np.copy(self.mu)


if __name__=="__main__":
    # Generate and plot 1D OU noise
    def plot_1d_ou_noise():
        ou_noise = OUNoise(theta=0.15, mu=0.0, sigma=0.2, dt=0.002)
        
        # Generate 1000 samples
        n_samples = 10000
        samples = []
        for _ in range(n_samples):
            noise = ou_noise.sample()
            noise = np.clip(noise, -0.2, 0.2)
            ou_noise.set_noise(noise)
            samples.append(noise)
        
        # Convert to numpy array
        samples = np.array(samples)
        
        # Create time axis
        time = np.arange(0, n_samples * ou_noise.dt, ou_noise.dt)
        
        plt.figure(figsize=(10, 6))
        plt.plot(time, samples, label='OU Noise')
        plt.xlabel('Time (s)')
        plt.ylabel('Noise Value')
        plt.title('1D Ornstein-Uhlenbeck Noise')
        plt.grid(True)
        plt.legend()
        plt.show()

    # Generate and plot 2D OU noise
    def plot_2d_ou_noise():
        ou_noise = OUNoise(theta=0.1, mu=[1.0, -1.0], sigma=0.2, dt=0.01, x_initial=[0, 0])
        
        # Generate 1000 samples
        n_samples = 10000
        samples = []
        for _ in range(n_samples):
            samples.append(ou_noise.sample())
        
        # Convert to numpy array (shape: [1000, 2])
        samples = np.array(samples)
        
        # Create time axis
        time = np.arange(0, n_samples * ou_noise.dt, ou_noise.dt)
        
        plt.figure(figsize=(10, 6))
        plt.plot(time, samples[:, 0], label='Dimension 1')
        plt.plot(time, samples[:, 1], label='Dimension 2')
        plt.xlabel('Time (s)')
        plt.ylabel('Noise Value')
        plt.title('2D Ornstein-Uhlenbeck Noise')
        plt.grid(True)
        plt.legend()
        plt.show()

    # Run the visualizations
    plot_1d_ou_noise()
    plot_2d_ou_noise()
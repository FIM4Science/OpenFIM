import torch

class SDE:
    def __init__(self, drift, diffusion, dt=0.01):
        """
        Initialize the SDE with given drift and diffusion functions.
        
        Parameters:
        drift (callable): Drift function f(x, t).
        diffusion (callable): Diffusion function g(x, t).
        dt (float): Time step size.
        """
        self.drift = drift
        self.diffusion = diffusion
        self.dt = dt

    def euler_maruyama_step(self, x):
        """
        Perform one step of the Euler-Maruyama method.
        
        Parameters:
        x (torch.Tensor): Current state.
        
        Returns:
        torch.Tensor: Updated state.
        """
        dW = torch.sqrt(torch.tensor(self.dt)) * torch.randn_like(x)
        x_new = x + self.drift(x) * self.dt + self.diffusion(x) * dW
        return x_new

    def simulate(self, x_0, n_steps, num_paths=1):
        """
        Simulate the SDE using the Euler-Maruyama method.
        
        Parameters:
        x0 (float): Initial condition.
        n_steps (int): Number of time steps.
        num_paths (int): Number of paths to simulate.
        
        Returns:
        torch.Tensor: Simulated trajectories.
        """
        x = x_0.repeat((num_paths,1))
        trajectory = [x[:,None,:].clone()]
        
        for _ in range(n_steps):
            x = self.euler_maruyama_step(x)
            trajectory.append(x[:,None,:].clone())
        
        return torch.concat(trajectory,dim=1)

class TwoDimensionalSynDrift:
    
    def __init__(self):
        pass
    
    def __call__(self, X):
        x = X[:,0]
        y = X[:,1]
        dxdt = x*(1.-x**2 - y**2)
        dydt = y*(1.-x**2 - y**2)
        return torch.stack([dxdt, dydt], dim=1)
    
    def derivative(self,X):
        x = X[:,0]
        y = X[:,1]
        A = (1.-x**2 - y**2)
        fxx = A -2.*x**2
        fxy = -2.*x*y
        fyx = -2.*x*y
        fyy = A - 2.*y**2

        J = torch.zeros((X.size(0),X.size(1),X.size(1)))
        J[:,0,0] = fxx
        J[:,0,1] = fxy
        J[:,1,0] = fyx
        J[:,1,1] = fyy

        return J

class Lorenz63Drift:

    def __init__(self, sigma=1.0, rho=1.0,beta=1.):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
    
    def __call__(self, X):
        x = X[0]
        y = X[1]
        z = X[2]
        dxdt = self.sigma * (y - x)
        dydt = x * (self.rho - z) - y
        dzdt = x * y - self.beta * z
        return torch.Tensor([dxdt, dydt, dzdt])
    
# Define the drift and diffusion functions for the double well potential SDE
class DoubleWellDrift:
    def __init__(self, a=1.0, b=1.0):
        self.a = a
        self.b = b
    
    def __call__(self, x):
        x = -(self.a * x**3 - self.b * x)
        return x

class LinearDrift1D:
    def __init__(self, a=1.0,b=0.2,x0=0.2):
        self.a = a
        self.b = b
        self.x0 = x0
    
    def __call__(self, x):
        x = - self.a * (x + self.b) + self.x0
        return x
    
class ConstantDiffusion:
    def __init__(self, sigma=0.5):
        self.sigma = sigma
    
    def __call__(self, x):
        return torch.sqrt(self.sigma * torch.ones_like(x))

class MaxDiffusion:
    def __init__(self, sigma=1.):
        self.sigma = sigma
    
    def __call__(self, x):
        D = self.sigma*torch.clamp(4. - 1.25*x**2,0.)
        return torch.sqrt(D)
    
if __name__=="__main__":
    num_paths = 2
    t_nmc = 10
    x_nmc = 9
    x_0 = torch.rand((1,)).unsqueeze(-1)

    drift = DoubleWellDrift()
    diffusion = ConstantDiffusion(sigma=1.)
    two_dimensional_sde = SDE(drift,diffusion)
    paths = two_dimensional_sde.simulate(x_0, 100, num_paths=num_paths)
    print(paths.shape)
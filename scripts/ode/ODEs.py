import numpy as np

def VDP_ode(t, y):
    """Van der Pol ODE: dx1/dt = x2, dx2/dt = -x1 + 0.5*x2*(1 - x1^2).
    
    Args:
        t: Time (not used, but required by scipy.integrate.solve_ivp)
        y: State vector [x1, x2] of shape (2,)
    
    Returns:
        np.ndarray: Derivative vector [dx1/dt, dx2/dt] of shape (2,)
    """
    x1, x2 = y[0], y[1]
    dx1dt = x2
    dx2dt = -x1 + 0.5 * x2 * (1 - x1**2)
    return np.array([dx1dt, dx2dt])


def FHN_ode(t, y):
    """FitzHugh-Nagumo ODE: dx1/dt = 3*(x1 - x1^3/3 + x2), dx2/dt = 0.2 - 3*x1 - 0.2*x2.
    
    Args:
        t: Time (not used, but required by scipy.integrate.solve_ivp)
        y: State vector [x1, x2] of shape (2,)
    
    Returns:
        np.ndarray: Derivative vector [dx1/dt, dx2/dt] of shape (2,)
    """
    x1, x2 = y[0], y[1]
    dx1dt = 3 * (x1 - (x1**3) / 3 + x2)
    dx2dt = (0.2 - 3 * x1 - 0.2 * x2) / 3
    return np.array([dx1dt, dx2dt])

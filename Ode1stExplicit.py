from abc import ABC, abstractmethod   # Abstract base class
import numpy as np
from scipy.integrate import solve_ivp

class Ode1stExplicit(ABC):
    """
    An abstract class that solves the 1st order ode system:

        dy_dt = f(y,p,t)
    
    The user could also define the output variable:

        z = I1(y_t1, p, t1) + int_t(I2(y,p,t))
    
    The final states y_t1 and the value of z are computed 
    by a forward integration; their sensitivity w.r.t parameters
    p is given by a backward integration of adjoint system.

    When using this class, the user is only required to:

        1. Give the dimension of state y
        2. Provide the value and Jacobian required
    
    We'll compute the value and sensitivity for you automatically.
    """
    def __init__ (self, dim_y, dim_p, method="RK45"):
        self.dim_y = dim_y
        self.dim_p = dim_p
        self.method = method

    ## Method provided by user ################################################
    @abstractmethod
    def f(self, t, y):
        """
        Right-hand side of the system
        """
        pass
    
    @abstractmethod
    def fj(self, t, y):
        """
        Jacobian of f
        """
        pass

    ## Forward Integration ####################################################
    def computeValue(self, t0, t1, y0):
        """
        Forward integration to compute the value of states
        """
        self.y = y0

        sol = solve_ivp(
            fun = self.f,
            t_span = (t0, t1),
            y0 = self.y,
            method = self.method,
            vectorized=False,
            jac = self.fj,
            dense_output = True,
            )

        self.y = sol.y[:,-1]   # Get the last column
        return sol

    ## Property ###############################################################
    @property
    def dim_y(self):
        return self._dim_y
    
    @dim_y.setter
    def dim_y(self, value):
        assert (value > 0)
        self._dim_y = value
        self._y = np.zeros(value)  # State
        self._l = np.zeros(value)  # Adjoint variable

    @property
    def dim_p(self):
        return self._dim_p
    
    @dim_p.setter
    def dim_p(self, value):
        self._dim_p = value
        self._p = np.zeros(value)  # Parameter
    
    @property
    def y(self):
        return self._y
    
    @y.setter
    def y(self, value):
        new_y = np.array(value)
        if new_y.shape == self._y.shape:
            self._y = new_y
        else:
            raise RuntimeError("Dimension of y doesn't match. Cannot set y to new value.")

    @property
    def p(self):
        return self._p
    
    @p.setter
    def p(self, value):
        new_p = np.array(value)
        if new_p.shape == self._p.shape:
            self._p = new_p
        else:
            raise RuntimeError("Dimension of p doesn't match. Cannot set p to new value.")

    

    
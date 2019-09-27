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
    def __init__ (self, dim_y, dim_p, dim_z, method="RK45"):
        self.dim_y = dim_y
        self.dim_p = dim_p
        self.dim_z = dim_z
        self.dzdp = np.zeros((dim_z, dim_p))
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

    @abstractmethod
    def fp(self, t, y):
        """
        Partial derivative of f w.r.t parameters p.
        fp[i][j] - derivative of f_i w.r.t p_j
        """
        pass

    @abstractmethod
    def I1(self, t, y):
        """
        Value of I1. Row vector with d = dim_z
        """
        pass

    @abstractmethod
    def dI1dp(self, t, y):
        """
        Derivative of I1 w.r.t parameters p. 
        dI1dp[i][j] - derivative of I1_i w.r.t p_j
        """
        pass

    @ abstractmethod
    def dI1dy(self, t, y):
        """
        Derivative of I1 w.r.t states y.
        dI1dy[i][j] - derivative of I1_i w.r.t y_j
        """
        pass

    @abstractmethod
    def I2(self, t, y):
        """
        Value of I2. Row vector with d = dim_z
        """
        pass

    @abstractmethod
    def dI2dp(self, t, y):
        """
        Derivative of I2 w.r.t parameters p. 
        dI2dp[i][j] - derivative of I2_i w.r.t p_j
        """
        pass

    @ abstractmethod
    def dI2dy(self, t, y):
        """
        Derivative of I1 w.r.t states y.
        dI2dy[i][j] - derivative of I2_i w.r.t y_j
        """
        pass

    ## Method used by Integrator ##############################################
    def rhs(self, t, y_aux):
        """
        The right hand side of the entire 1st system. y_aux = [y, z]
        """
        y = y_aux[:self.dim_y]
        y_aux_dot = np.zeros(self.dim_y+self.dim_z)
        y_aux_dot[:self.dim_y] = self.f(t, y)
        y_aux_dot[self.dim_y:] = self.I2(t,y)
        return y_aux_dot

    def rhs_adjoint(self, t, y_aux):
        """
        RHS of adjoint system. y_aux = [y, z, lambda, dz_i/dp_j]
        """
        dim_l = self.dim_z * self.dim_y
        dim_zp = self.dim_z * self.dim_p
        y_aux_dot = np.zeros(self.dim_y + self.dim_z + dim_l + dim_zp)
        
        # Getting value from auxilary states
        y = y_aux[0:self.dim_y]
        z = y_aux[self.dim_y:self.dim_y+self.dim_z]
        # l = y_aux[self.dim_y+self.dim_z:self.dim_y+self.dim_z+dim_l]
        
        # Compute y_dot
        y_aux_dot[0:self.dim_y] = self.f(t,y)
        # Compute z_dot
        y_aux_dot[self.dim_y:self.dim_y+self.dim_z] = self.I2(t, y)
        # Compute lambda_dot
        l_index = self.dim_y+self.dim_z
        dh_dx_t = -self.fj(t, y).T
        dI2dy = self.dI2dy(t, y)
        for i in range(self.dim_z):
            l_i = y_aux[l_index:l_index+self.dim_y].reshape(self.dim_y, 1)
            l_i_dot = dI2dy[i,:] + dh_dx_t.dot(l_i)
            y_aux_dot[l_index:l_index+self.dim_y] = l_i_dot
            l_index += self.dim_y
        # Compute dz_i/dp_j_dot
        l_index = self.dim_y+self.dim_z
        zp_index = self.dim_y + self.dim_z + dim_l
        dI2dp = self.dI2dp(t, y)
        dhdp = -self.fp(t, y)
        for i in range(self.dim_z):
            l_i = y_aux[l_index:l_index+self.dim_y].reshape(1, self.dim_y)
            dzdp_i = dI2dp[i].reshape(self.dim_p, 1) + l_i.dot(dhdp)
            y_aux_dot[zp_index:zp_index+self.dim_p] = dzdp_i
            l_index += self.dim_y
            zp_index += self.dim_p
        
        return y_aux_dot

    ## Forward Integration ####################################################
    def computeValue(self, t0, t1, y0):
        """
        Forward integration to compute the value of states
        """
        assert len(y0) == self.dim_y
        y_aux = np.zeros(self.dim_y+self.dim_z)
        y_aux[:self.dim_y] = y0

        sol = solve_ivp(
            fun = self.rhs,
            t_span = (t0, t1),
            y0 = y_aux,
            method = self.method,
            vectorized = False,
            dense_output = True,
            )

        if not sol.success:
            raise RuntimeError("Integration Fails.")

        y_aux = sol.y[:,-1]   # Get the last column
        self.y = y_aux[:self.dim_y]
        self.z = y_aux[self.dim_y:] + self.I1(t1, self.y)
        return sol

    ## Backward Integration ###################################################
    def computeSensitivity(self, t0, t1, y0):
        """
        Backward integration to compute the value of sensitivity
        """
        self.computeValue(t0, t1, y0)  # Compute self.y = y_1


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

    @property
    def dim_z(self):
        return self._dim_z
    
    @dim_z.setter
    def dim_z(self, value):
        self._dim_z = value
        self._z = np.zeros(value)  # Variable of interest

    @property
    def z(self):
        return self._z
    
    @z.setter
    def z(self, value):
        new_z = np.array(value)
        if new_z.shape == self._z.shape:
            self._z = new_z
        else:
            raise RuntimeError("Dimension of z doesn't match. Cannot set z to new value.")

    # Adjoint variable
    @property
    def l(self):
        return self._l
    
    @l.setter
    def l(self, value):
        new_l = np.array(value)
        if new_l.shape == self._l.shape:
            self._l = new_l
        else:
            raise RuntimeError("Dimension of l doesn't match. Cannot set z to new value.")

    @property
    def dzdp(self):
        return self._dzdp
    
    @dzdp.setter
    def dzdp(self, value):
        new_dzdp = np.array(value)
        if new_dzdp.shape == (self.dim_z, self.dim_p):
            self._dzdp = new_dzdp
        else:
            raise RuntimeError("Dimension of l doesn't match. Cannot set z to new value.")
    
"""
In this example, a kinematic bicycle model is defined and
solved by odeAdjoint. The value and sensitivity are computed
and used by scipy.optimize to find the optimal control
input (steering angle and acceleration). The goal is to
let the vehicle follow the target trajectory and achieve
the target speed.

This can be considered as one step in Model Predictive Control(MPC).
"""

from odeAdjoint import Ode1stExplicit
import numpy as np
import matplotlib.pyplot as plt

class Bicycle(Ode1stExplicit):
    """
    Kinematic Bicycle Model
    """
    def __init__ (self, lf=4, k=3, method="RK45"):
        dim_y = 4
        dim_z = 1
        dim_p = 2*(k+1)
        super().__init__(dim_y, dim_p, dim_z, method=method)
        self.lf = lf
        self.k = k
        self.p = np.zeros(dim_p)   # Not sure if 3rd order poly is good enough
        self.traj = np.poly1d([-5.5555555556e-5, 5e-3, 0, 0])

    def plotTraj(self, x0=0, x1=60):
        """
        Plot the target trajectory for testing purpose
        """
        x = np.linspace(x0, x1, 101)
        y = self.traj(x)
        # dydx = self.traj.deriv(1)(x)
        plt.plot(x,y)
        # plt.plot(x,dydx)
        plt.show()
        
    def f(self, t, y):
        """
        RHS of kinematic bicycle model. 4 equations.
        """
        steer_poly = self.p[:self.k+1]
        acc_poly = self.p[self.k+1:]

        # RHS - y = [x, y, phi, v]
        rhs = np.zeros(4)
        rhs[0] = y[3]*np.cos(y[2])
        rhs[1] = y[3]*np.sin(y[2])
        rhs[2] = y[3]/self.lf*np.polyval(steer_poly, t)
        rhs[3] = np.polyval(acc_poly, t)
        return rhs
    
    def fj(self, t, y):
        """
        Jacobian of RHS
        """
        jac = np.zeros((4,4))
        steer_poly = self.p[:self.k+1]
        jac[0,2] = -y[3]*np.sin(y[2])
        jac[0,3] = np.cos(y[2])
        jac[1,2] = y[3]*np.cos(y[2])
        jac[1,3] = np.sin(y[2])
        jac[2,3] = np.polyval(steer_poly, t)/self.lf
        return jac

    def fp(self, t, y):
        """
        Partial derivative of f w.r.t parameters p.
        fp[i][j] - derivative of f_i w.r.t p_j
        """
        jac = np.zeros((4, self.dim_p))
        
        # Derivative of polynomial
        for i in range(self.k+1):
            t_n = t**i
            jac[2, self.k+1-i] = y[3]/self.lf*t_n
            jac[3, -i] = t_n

        return jac

    def I1(self, t, y):
        """
        Four variables of interest. Only the first three components are non-zero.
        See comments in __main__.
        """
        i1 = np.zeros(4)
        i1[0] = y[3] - 22.22
        i1[1] = 27.78 - y[3]
        i1[2] = (y[0]-60)**2 + (y[1]-6)**2
        return i1

    def dI1dp(self, t, y):
        """
        I1 is not related to parameters p. See comments in __main__.
        """
        return np.zeros((4,self.dim_p))

    def dI1dy(self, t, y):
        """
        Derivative of I1 w.r.t states y.
        dI1dy[i][j] - derivative of I1_i w.r.t y_j
        """
        jac = np.zeros((4,self.dim_y))
        jac[0][3] = 1
        jac[1][3] = -1
        jac[2][0] = 2*(y[0]-60)
        jac[2][1] = 2*(y[1]-6)
        return jac

    def I2(self, t, y):
        """
        Value of I2. Row vector with d = dim_z
        """
        pass

    def dI2dp(self, t, y):
        """
        Derivative of I2 w.r.t parameters p. 
        dI2dp[i][j] - derivative of I2_i w.r.t p_j
        """
        pass

    def dI2dy(self, t, y):
        """
        Derivative of I1 w.r.t states y.
        dI2dy[i][j] - derivative of I2_i w.r.t y_j
        """
        pass

    ## Helper functions #######################################################

if __name__ == "__main__":
    car = Bicycle()
    car.plotTraj()
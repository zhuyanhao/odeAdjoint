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
from scipy.optimize import minimize

class Bicycle(Ode1stExplicit):
    """
    Kinematic Bicycle Model
    """
    def __init__ (self, lf=4, k=3, method="RK45"):
        dim_y = 4
        dim_z = 4
        dim_p = 2*(k+1)
        super().__init__(dim_y, dim_p, dim_z, method=method)
        self.lf = lf
        self.k = k
        self.p = np.zeros(dim_p)   # Not sure if 3rd order poly is good enough
        self.traj = np.poly1d([-5.5555555556e-5, 5e-3, 0, 0])
        self.traj_d = self.traj.deriv(m=1)
        self.cache = dict()
        
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
        i2 = np.zeros(4)
        i2[3] = (y[1] - self.traj(y[0])) ** 2
        return i2

    def dI2dp(self, t, y):
        """
        Derivative of I2 w.r.t parameters p. 
        dI2dp[i][j] - derivative of I2_i w.r.t p_j
        """
        return np.zeros((4, self.dim_p))

    def dI2dy(self, t, y):
        """
        Derivative of I1 w.r.t states y.
        dI2dy[i][j] - derivative of I2_i w.r.t y_j
        """
        jac = np.zeros((4, 4))
        mul = 2*(y[1]-self.traj(y[0]))
        jac[3,0] = -mul*self.traj_d(y[0])
        jac[3,1] = mul
        return jac

    ## Helper functions #######################################################
    def plotTraj(self, y0=np.array([0.0,0.0,0.0,15.0]), t0=0, t1=4):
        """
        Plot the actual trajectory on top of target
        """
        x_target = np.linspace(0, 60, 101)
        y_target = self.traj(x_target)
        plt.plot(x_target, y_target, label="Target")
        # sol = self.computeSensitivity(t0, t1, y0)
        sol = self.computeValue(t0, t1, y0)
        x_actual = sol.y[0,:]
        y_actual = sol.y[1,:]
        plt.plot(x_actual, y_actual, label="Actual")
        plt.legend()
        plt.title("Trajectory of Car")
        plt.show()

    def targetTraj(self):
        x_target = np.linspace(0, 60, 101)
        y_target = self.traj(x_target)
        return x_target, y_target

    def runSimulation(self, p):
        """
        Run simulation if p is a new design
        """
        if tuple(p) in self.cache.keys():
            return
        
        self.p = np.copy(p)
        self.computeSensitivity(0, 4, np.array([0.0,0.0,0.0,15.0]))
        cached_sol = dict(
            y = np.copy(self.y),
            z = np.copy(self.z),
            dzdp = np.copy(self.dzdp)
        )
        self.cache[tuple(p)] = cached_sol
    
    def obj(self, p):
        """
        obj = z[2] + 10*z[3]
        """
        self.runSimulation(p)
        sol = self.cache[tuple(p)]
        return sol['z'][2] + 10*sol['z'][3]

    def obj_j(self, p):
        """
        Jacobian of obj
        """
        self.runSimulation(p)
        sol = self.cache[tuple(p)]
        return sol['dzdp'][2,:] + 10*sol['dzdp'][3,:]

    def con1(self, p):
        """
        Constraint 1 - lower bound of velocity
        """
        self.runSimulation(p)
        sol = self.cache[tuple(p)]
        return sol['z'][0]

    def con1_j(self, p):
        """
        Jacobian of con1
        """
        self.runSimulation(p)
        sol = self.cache[tuple(p)]
        return sol['dzdp'][0, :]

    def con2(self, p):
        """
        Constraint 2 - upper bound of velocity
        """
        self.runSimulation(p)
        sol = self.cache[tuple(p)]
        return sol['z'][1]

    def con2_j(self, p):
        """
        Jacobian of con2
        """
        self.runSimulation(p)
        sol = self.cache[tuple(p)]
        return sol['dzdp'][1, :]

    def callback(self, p):
        """
        Update p
        """
        self.p = np.copy(p)

if __name__ == "__main__":
    car = Bicycle(k=3)
    p0 = np.zeros(8)
    pb = ((-0.01,0.01), (-0.05,0.05), (-0.1,0.1), (-1,1), (-0.01,0.01), (-0.05,0.05), (-0.1,0.1), (-1,1))
    car.p = p0

    # Store the initial design
    sol = car.computeValue(t0=0, t1=4, y0=np.array([0.0,0.0,0.0,15.0]))
    x_init = sol.y[0,:]
    y_init = sol.y[1,:]

    # Get the target design
    x_target, y_target = car.targetTraj()

    # Optimization
    minimize(
        fun = car.obj,
        x0 = p0,
        bounds = pb,
        method = 'SLSQP',
        jac = car.obj_j,
        tol = 1e-3,
        callback = car.callback,
        options = dict(maxiter=20, disp=True)
    )

    # Store the optimal design
    sol = car.computeValue(t0=0, t1=4, y0=np.array([0.0,0.0,0.0,15.0]))
    x_opt = sol.y[0,:]
    y_opt = sol.y[1,:]

    # Plot - Car Trajectory
    fig = plt.figure(figsize=(12,6))
    ax1 = plt.subplot(1,3,1)
    ax1.plot(x_init, y_init, label="Initial")
    ax1.plot(x_target, y_target, label="Target")
    ax1.plot(x_opt, y_opt, label="Optimal")
    ax1.set_title("Car Trajectory")
    ax1.set_xlabel("X(m)")
    ax1.set_ylabel("Y(m)")
    ax1.legend()
    
    # Plot - Steer input
    ps0 = p0[:4]
    ps1 = car.p[:4]
    t = np.linspace(0,4,101)
    s0 = np.polyval(ps0, t)
    s1 = np.polyval(ps1, t)
    ax2 = plt.subplot(1,3,2)
    ax2.plot(t, s0, label="Initial")
    ax2.plot(t, s1, label="Optimal")
    ax2.set_title("Steering Input")
    ax2.set_xlabel("Time(s)")
    ax2.set_ylabel("Steering Angle(rad)")
    ax2.legend()
    
    # Plot - Acceleration
    pa0 = p0[4:]
    pa1 = car.p[4:]
    t = np.linspace(0,4,101)
    a0 = np.polyval(pa0, t)
    a1 = np.polyval(pa1, t)
    ax3 = plt.subplot(1,3,3)
    ax3.plot(t, a0, label="Initial")
    ax3.plot(t, a1, label="Optimal")
    ax3.set_title("Acceleration Input")
    ax3.set_xlabel("Time(s)")
    ax3.set_ylabel("Acceleration(m/s^2)")
    ax3.legend()

    plt.tight_layout()
    plt.show()

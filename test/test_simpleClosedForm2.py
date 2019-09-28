from odeAdjoint import *
import numpy as np
import pytest

from odeAdjoint import *
import numpy as np
import pytest

class SimpleClosedForm2(Ode1stExplicit):
    """
    Mimic a second order system
    
    y0_dot = p0*p1
    y1_dot = y0

    Initial condition = [0, 0]
    
    Solution: y = [p0*p1*t, 0.5*p0*p1*t**2]

    Variable of interest:
    z0 = (y1 - 5)**2
    z1 = (p0 + p1)**2
    z2 = int_t(y0-0)**2
    """
    def f(self, t, y):
        ans = np.zeros((2,1))
        ans[0,0] = self.p[0]*self.p[1]
        ans[1,0] = y[0]
        return ans

    def fj(self, t, y):
        ans = np.zeros((2,2))
        ans[1,0] = 1
        return ans
    
    def fp(self, t, y):
        ans = np.zeros((2,2))
        ans[0,0] = self.p[1]
        ans[0,1] = self.p[0]
        return ans

    def I1(self,t,y):
        ans = np.zeros(3)
        ans[0] = (y[1]-5)**2
        ans[1] = (self.p[0]+self.p[1])**2
        return ans
    
    def I2(self,t,y):
        ans = np.zeros(3)
        ans[2] = y[0]**2
        return ans

    def dI1dp(self, t, y):
        """
        Derivative of I1 w.r.t parameters p. 
        dI1dp[i][j] - derivative of I1_i w.r.t p_j
        """
        ans = np.zeros((3,2))
        ans[1,0] = ans[1,1] = 2*(self.p[0]+self.p[1])
        return ans

    def dI1dy(self, t, y):
        """
        Derivative of I1 w.r.t states y.
        dI1dy[i][j] - derivative of I1_i w.r.t y_j
        """
        ans = np.zeros((3,2))
        ans[0,1] = 2*(y[1]-5)
        return ans

    def dI2dp(self, t, y):
        """
        Derivative of I2 w.r.t parameters p. 
        dI2dp[i][j] - derivative of I2_i w.r.t p_j
        """
        return np.zeros((3,2))

    def dI2dy(self, t, y):
        """
        Derivative of I1 w.r.t states y.
        dI2dy[i][j] - derivative of I2_i w.r.t y_j
        """
        ans = np.zeros((3,2))
        ans[2,0] = 2*y[0]
        return ans

def test_RK45():
    test = SimpleClosedForm2(dim_p=2, dim_y=2, dim_z=3, method="RK45")
    test.p = np.array([2,3])
    y0 = np.zeros(2)
    test.computeSensitivity(t0=0, t1=2, y0=y0)
    # Check value
    assert np.isclose(test.y[0], 12, 1e-2, 1e-2)
    assert np.isclose(test.y[1], 12, 1e-2, 1e-2)
    assert np.isclose(test.z[0], 49, 1e-2, 1e-2)
    assert np.isclose(test.z[1], 25, 1e-2, 1e-2)
    assert np.isclose(test.z[2], 96, 1e-2, 1e-2)
    # Check sensitivity
    dzdp = test.dzdp
    assert np.isclose(dzdp[0,0], 28*3, 1e-2, 1e-2)
    assert np.isclose(dzdp[0,1], 28*2, 1e-2, 1e-2)
    assert np.isclose(dzdp[1,0], 10, 1e-2, 1e-2)
    assert np.isclose(dzdp[1,1], 10, 1e-2, 1e-2)
    assert np.isclose(dzdp[2,0], 16/3*2*9, 1e-2, 1e-2)
    assert np.isclose(dzdp[2,1], 16/3*4*3, 1e-2, 1e-2)

def test_BDF():
    test = SimpleClosedForm2(dim_p=2, dim_y=2, dim_z=3, method="BDF")
    test.p = np.array([2,3])
    y0 = np.zeros(2)
    test.computeSensitivity(t0=0, t1=2, y0=y0)
    # Check value
    assert np.isclose(test.y[0], 12, 1e-2, 1e-2)
    assert np.isclose(test.y[1], 12, 1e-2, 1e-2)
    assert np.isclose(test.z[0], 49, 1e-2, 1e-2)
    assert np.isclose(test.z[1], 25, 1e-2, 1e-2)
    assert np.isclose(test.z[2], 96, 1e-2, 1e-2)
    # Check sensitivity
    dzdp = test.dzdp
    assert np.isclose(dzdp[0,0], 28*3, 1e-2, 1e-2)
    assert np.isclose(dzdp[0,1], 28*2, 1e-2, 1e-2)
    assert np.isclose(dzdp[1,0], 10, 1e-2, 1e-2)
    assert np.isclose(dzdp[1,1], 10, 1e-2, 1e-2)
    assert np.isclose(dzdp[2,0], 16/3*2*9, 1e-2, 1e-2)
    assert np.isclose(dzdp[2,1], 16/3*4*3, 1e-2, 1e-2)
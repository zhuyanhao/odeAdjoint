from odeAdjoint import *
import numpy as np
import pytest

class SimpleClosedForm(Ode1stExplicit):
    """
    dy_dt = p[0]y
    z[0] = y1 + p[0]
    z[1] = p[1] + int_t(y)
    """
    def f(self, t, y):
        return np.array([self.p[0]*y[0]])
    
    def fj(self, t, y):
        return np.array([[self.p[0]]])   # Shape (1,1)
    
    def fp(self, t, y):
        ans = np.zeros((1,2))
        ans[0][0] = y[0]
        ans[0][1] = 0
        return ans

    def I1(self,t,y):
        ans = np.zeros(2)
        ans[0] = y[0] + self.p[0]
        ans[1] = self.p[1]
        return ans
    
    def I2(self,t,y):
        ans = np.zeros(2)
        ans[1] = y[0]
        return ans

    def dI1dp(self, t, y):
        """
        Derivative of I1 w.r.t parameters p. 
        dI1dp[i][j] - derivative of I1_i w.r.t p_j
        """
        return np.eye(2)

    def dI1dy(self, t, y):
        """
        Derivative of I1 w.r.t states y.
        dI1dy[i][j] - derivative of I1_i w.r.t y_j
        """
        ans = np.zeros((2,1))
        ans[0][0] = 1
        return ans

    def dI2dp(self, t, y):
        """
        Derivative of I2 w.r.t parameters p. 
        dI2dp[i][j] - derivative of I2_i w.r.t p_j
        """
        return np.zeros((2,2))

    def dI2dy(self, t, y):
        """
        Derivative of I1 w.r.t states y.
        dI2dy[i][j] - derivative of I2_i w.r.t y_j
        """
        ans = np.zeros((2,1))
        ans[1][0] = 1
        return ans

def test_RK45():
    test = SimpleClosedForm(1, 2, 2, method="RK45")
    test.p = [2, 3]
    test.computeSensitivity(0, 1, [1])
    # Check value
    assert np.isclose(test.y[0], np.e**2, 1e-2, 1e-2)
    assert np.isclose(test.z[0], np.e**2+2, 1e-2, 1e-2)
    assert np.isclose(test.z[1], 0.5*np.e**2-0.5+3, 1e-2, 1e-2)
    # Check sensitivity
    dzdp = test.dzdp
    assert np.isclose(dzdp[0,0], np.e**2+1, 1e-2, 1e-2)
    assert np.isclose(dzdp[0,1], 0, 1e-2, 1e-2)
    assert np.isclose(dzdp[1,0], (np.e**2+1)/4, 1e-2, 1e-2)
    assert np.isclose(dzdp[1,1], 1, 1e-2, 1e-2)

def test_BDF():
    test = SimpleClosedForm(1, 2, 2, method="BDF")
    test.p = [2, 3]
    test.computeSensitivity(0, 1, [1])
    # Check value
    assert np.isclose(test.y[0], np.e**2, 1e-2, 1e-2)
    assert np.isclose(test.z[0], np.e**2+2, 1e-2, 1e-2)
    assert np.isclose(test.z[1], 0.5*np.e**2-0.5+3, 1e-2, 1e-2)
    # Check sensitivity
    dzdp = test.dzdp
    assert np.isclose(dzdp[0,0], np.e**2+1, 1e-2, 1e-2)
    assert np.isclose(dzdp[0,1], 0, 1e-2, 1e-2)
    assert np.isclose(dzdp[1,0], (np.e**2+1)/4, 1e-2, 1e-2)
    assert np.isclose(dzdp[1,1], 1, 1e-2, 1e-2)
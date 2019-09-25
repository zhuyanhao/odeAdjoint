from odeAdjoint import *
import numpy as np

class SimpleClosedForm(Ode1stExplicit):
    """
    dx_dt = p[0]x + p[1]
    """
    def f(self, t, y):
        return self.p[0]*y[0]
    
    def fj(self, t, y):
        return [[self.p[0]]]   # Shape (1,1)

def test_RK45():
    test = SimpleClosedForm(1, 1, method="RK45")
    test.p = [2]
    test.computeValue(0, 1, [1])
    assert np.isclose(test.y[0], np.e**2, 1e-2, 1e-2)

def test_BDF():
    test = SimpleClosedForm(1, 1, method="BDF")
    test.p = [2]
    test.computeValue(0, 1, [1])
    assert np.isclose(test.y[0], np.e**2, 1e-2, 1e-2)
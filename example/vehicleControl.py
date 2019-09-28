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

class Bicycle(Ode1stExplicit):
    """
    Kinematic Bicycle Model
    """
    
import numpy as np
from numpy.linalg import norm

from gym_art.quadrotor_multi.quad_utils import normalize

GRAV = 9.81
def cross(a, b):
    return np.array([a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]])

def clamp_norm(x, maxnorm):
    #n = np.linalg.norm(x)
   # n = np.sqrt(np.cumsum(np.square(x)))[2]
    n = (x[0]**2 + x[1]**2 + x[2]**2)**0.5
    return x if n <= maxnorm else (maxnorm / n) * x

def quadrotor_jacobian(dynamics):
    torque = dynamics.thrust_max * dynamics.prop_crossproducts.T
    torque[2,:] = dynamics.torque_max * dynamics.prop_ccw
    thrust = dynamics.thrust_max * np.ones((1,4))
    dw = (1.0 / dynamics.inertia)[:,None] * torque
    dv = thrust / dynamics.mass
    J = np.vstack([dv, dw])
    J_cond = np.linalg.cond(J)
    # assert J_cond < 100.0
    if J_cond > 50:
        print("WARN: Jacobian conditioning is high: ", J_cond)
    return J

# this is an "oracle" policy to drive the quadrotor towards a goal
# using the controller from Mellinger et al. 2011
class MellingerController(object):
    # @profile
    def __init__(self, dynamics):
        jacobian = quadrotor_jacobian(dynamics)
        self.Jinv = np.linalg.inv(jacobian)
        ## Jacobian inverse for our quadrotor
        # Jinv = np.array([[0.0509684, 0.0043685, -0.0043685, 0.02038736],
        #                 [0.0509684, -0.0043685, -0.0043685, -0.02038736],
        #                 [0.0509684, -0.0043685,  0.0043685,  0.02038736],
        #                 [0.0509684,  0.0043685,  0.0043685, -0.02038736]])
        self.action = None

        self.kp_p, self.kd_p = 4.5, 3.5
        self.kp_a, self.kd_a = 200.0, 50.0

        self.rot_des = np.eye(3)

        self.step_func = self.step

    # modifies the dynamics in place.
    # @profile
    def step(self, dynamics, goal, dt, action=None, observation=None):
        to_goal = goal - dynamics.pos
        # goal_dist = np.sqrt(np.cumsum(np.square(to_goal)))[2]
        goal_dist = (to_goal[0] ** 2 + to_goal[1] ** 2 + to_goal[2] ** 2) ** 0.5
        ##goal_dist = norm(to_goal)
        e_p = -clamp_norm(to_goal, 4.0)
        e_v = dynamics.vel
        # print('Mellinger: ', e_p, e_v, type(e_p), type(e_v))
        acc_des = -self.kp_p * e_p - self.kd_p * e_v + np.array([0, 0, GRAV])

        # I don't need to control yaw
        # if goal_dist > 2.0 * dynamics.arm:
        #     # point towards goal
        #     xc_des = to_xyhat(to_goal)
        # else:
        #     # keep current
        #     xc_des = to_xyhat(dynamics.rot[:,0])

        xc_des = self.rot_des[:, 0]
        # xc_des = np.array([1.0, 0.0, 0.0])

        # rotation towards the ideal thrust direction
        # see Mellinger and Kumar 2011
        zb_des, _ = normalize(acc_des)
        yb_des, _ = normalize(cross(zb_des, xc_des))
        xb_des = cross(yb_des, zb_des)
        R_des = np.column_stack((xb_des, yb_des, zb_des))
        R = dynamics.rot

        def vee(R):
            return np.array([R[2, 1], R[0, 2], R[1, 0]])

        e_R = 0.5 * vee(np.matmul(R_des.T, R) - np.matmul(R.T, R_des))
        e_R[2] *= 0.2  # slow down yaw dynamics
        e_w = dynamics.omega

        dw_des = -self.kp_a * e_R - self.kd_a * e_w
        # we want this acceleration, but we can only accelerate in one direction!
        thrust_mag = np.dot(acc_des, R[:, 2])

        des = np.append(thrust_mag, dw_des)

        # print('Jinv:', self.Jinv)
        thrusts = np.matmul(self.Jinv, des)
        thrusts[thrusts < 0] = 0
        thrusts[thrusts > 1] = 1

        dynamics.step(thrusts, dt)
        self.action = thrusts.copy()
        return self.action
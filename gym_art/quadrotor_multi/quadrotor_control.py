from numpy.linalg import norm
from gymnasium import spaces
from gym_art.quadrotor_multi.quad_utils import *
import qpsolvers
import math
from scipy.sparse import csc_matrix

GRAV = 9.81


class NominalSBC:
    class State:
        def __init__(self, position, velocity):
            self.position = position
            self.velocity = velocity

    class ObjectDescription:
        def __init__(
                self, state, radius, maximum_linf_acceleration_lower_bound,
                is_infinite_height_cylinder=False):
            self.state = state
            self.radius = radius
            self.maximum_linf_acceleration_lower_bound = maximum_linf_acceleration_lower_bound
            self.is_infinite_height_cylinder = is_infinite_height_cylinder

    def __init__(
            self, maximum_linf_acceleration, aggressiveness, radius, room_box):
        self.maximum_linf_acceleration = maximum_linf_acceleration
        self.aggressiveness = aggressiveness
        self.radius = radius
        self.room_box = room_box

    def _compute_maximum_distance_to_boundary(self, v, G, h):
        """
            Computes the maximum distance of v to the constraint boundary
            defined by Gx <= h.

            If a constraint is satisfied by v, the distance of v to the
            constraint boundary is negative. Constraints take value zero at the
            boundary. Distance to violated constraints is positive.

            If the return value is positive, the return value is the distance
            to the most violated constraint.

            If it is non-positive, all constraints are satisfied,
            and the return value is the (negated) distance to the
            closest constraint.

            In any case, we want this value to be minimized.
        """
        G_norms = np.linalg.norm(G, axis=1).reshape((-1, 1))
        G_unit = G / G_norms
        h = h.reshape((-1, 1))
        h_norm = h / G_norms
        v = v.reshape((-1, 1))
        distances = np.dot(G_unit, v) - h_norm
        return np.max(distances)

    def plan(self, self_state, object_descriptions, desired_acceleration):
        P = 2.0 * np.eye(3)
        q = -2.0 * desired_acceleration
        G = np.ndarray((0, 3), np.float64)
        h = np.array([])
        A = np.ndarray((0, 3), np.float64)
        b = np.array([])
        lb = np.array([-self.maximum_linf_acceleration]*3)
        ub = np.array([self.maximum_linf_acceleration]*3)

        # Add robot and obstacle collision avoidance constraints
        for object_description in object_descriptions:
            dim = 2 if object_description.is_infinite_height_cylinder else 3

            relative_position = self_state.position[:
                                                    dim] - object_description.state.position
            relative_position_norm = np.linalg.norm(relative_position)
            if relative_position_norm < 1e-10:
                return None, None
            safety_distance = self.radius + object_description.radius

            if relative_position_norm < safety_distance:
                return None, None

            relative_velocity = self_state.velocity[:
                                                    dim] - object_description.state.velocity
            relative_position_dot_relative_velocity = np.dot(
                relative_position, relative_velocity)

            hij = math.sqrt(2.0 * (self.maximum_linf_acceleration + object_description.maximum_linf_acceleration_lower_bound) * (
                relative_position_norm - safety_distance)) + (relative_position_dot_relative_velocity / relative_position_norm)

            bij = -relative_position_dot_relative_velocity * np.dot(
                relative_position, self_state.velocity[:dim]) / (
                relative_position_norm * relative_position_norm) + np.dot(
                relative_velocity, self_state.velocity[:dim]) + (
                self.maximum_linf_acceleration /
                (self.maximum_linf_acceleration + object_description.
                 maximum_linf_acceleration_lower_bound)) * (
                self.aggressiveness * hij * hij * hij * relative_position_norm +
                (math.sqrt(
                    self.maximum_linf_acceleration + object_description.
                    maximum_linf_acceleration_lower_bound) *
                 relative_position_dot_relative_velocity) /
                (math.sqrt(2.0 * (relative_position_norm - safety_distance))))

            if dim == 2:
                relative_position = np.append(relative_position, 0.0)

            G = np.append(G, [-1.0 * relative_position], axis=0)
            h = np.append(h, bij)

        # Add room_box constraints
        for d in range(3):
            relative_positions = [
                self_state.position[d] - self.room_box[0, d],
                self_state.position[d] - self.room_box[1, d]]

            for relative_position in relative_positions:
                relative_position_abs = abs(relative_position)
                if relative_position_abs < 1e-10:
                    return None, None

                if relative_position_abs < self.radius:
                    return None, None

                relative_position_times_velocity = relative_position * \
                    self_state.velocity[d]

                hij = math.sqrt(
                    2.0 * (self.maximum_linf_acceleration) *
                    (relative_position_abs - self.radius)) + (
                    relative_position_times_velocity /
                    relative_position_abs)

                bij = -relative_position_times_velocity * \
                    relative_position_times_velocity / \
                    (relative_position * relative_position) + \
                    self_state.velocity[d] * self_state.velocity[d] + \
                    (self.aggressiveness *
                     hij * hij * hij * relative_position_abs +
                     (math.sqrt(self.maximum_linf_acceleration) *
                      relative_position_times_velocity) /
                     (math.sqrt(2.0 * (relative_position_abs - self.radius))))

                coefficients = np.zeros((1, 3))
                coefficients[0, d] = -1.0 * relative_position

                G = np.append(G, coefficients, axis=0)
                h = np.append(h, bij)

        maximum_distance = self._compute_maximum_distance_to_boundary(
            desired_acceleration, G, h)

        P = csc_matrix(P)
        G = csc_matrix(G)
        A = csc_matrix(A)

        # This is the bottleneck
        # TODO: find a better QP solver
        solution = qpsolvers.solve_qp(P, q, G, h, A, b, lb, ub, solver="osqp")

        return solution, maximum_distance


class RawControl(object):
    def __init__(self, dynamics, zero_action_middle=True):
        self.zero_action_middle = zero_action_middle
        self.action = None
        self.step_func = self.step
        self.high = np.ones(4)
        self.low = -np.ones(4)
        self.bias = 1.0
        self.scale = 0.5

    def action_space(self, dynamics):
        if not self.zero_action_middle:
            # Range of actions 0 .. 1
            self.low = np.zeros(4)
            self.bias = 0.0
            self.scale = 1.0
        else:
            # Range of actions -1 .. 1
            self.low = -np.ones(4)
            self.bias = 1.0
            self.scale = 0.5
        self.high = np.ones(4)
        return spaces.Box(self.low, self.high, dtype=np.float32)

    # modifies the dynamics in place.
    # @profile
    def step(self, dynamics, action, dt):
        action = np.clip(action, a_min=self.low, a_max=self.high)
        action = self.scale * (action + self.bias)
        dynamics.step(action, dt)
        self.action = action.copy()

    # @profile
    def step_tf(self, dynamics, action, goal, dt, observation=None):
        # print('bias/scale: ', self.scale, self.bias)
        action = np.clip(action, a_min=self.low, a_max=self.high)
        action = self.scale * (action + self.bias)
        dynamics.step(action, dt)
        self.action = action.copy()


class VerticalControl(object):
    def __init__(self, dynamics, zero_action_middle=True, dim_mode="3D"):
        self.zero_action_middle = zero_action_middle

        self.dim_mode = dim_mode
        if self.dim_mode == '1D':
            self.step = self.step1D
        elif self.dim_mode == '3D':
            self.step = self.step3D
        else:
            raise ValueError(
                'QuadEnv: Unknown dimensionality mode %s' % self.dim_mode)
        self.step_func = self.step

    def action_space(self, dynamics):
        if not self.zero_action_middle:
            # Range of actions 0 .. 1
            self.low = np.zeros(1)
            self.bias = 0
            self.scale = 1.0
        else:
            # Range of actions -1 .. 1
            self.low = -np.ones(1)
            self.bias = 1.0
            self.scale = 0.5
        self.high = np.ones(1)
        return spaces.Box(self.low, self.high, dtype=np.float32)

    # modifies the dynamics in place.
    # @profile
    def step3D(self, dynamics, action, goal, dt, observation=None):
        # print('action: ', action)
        action = self.scale * (action + self.bias)
        action = np.clip(action, a_min=self.low, a_max=self.high)
        dynamics.step(np.array([action[0]] * 4), dt)

    # modifies the dynamics in place.
    # @profile
    def step1D(self, dynamics, action, goal, dt, observation=None):
        # print('action: ', action)
        action = self.scale * (action + self.bias)
        action = np.clip(action, a_min=self.low, a_max=self.high)
        dynamics.step(np.array([action[0]]), dt)


class VertPlaneControl(object):
    def __init__(self, dynamics, zero_action_middle=True, dim_mode="3D"):
        self.zero_action_middle = zero_action_middle

        self.dim_mode = dim_mode
        if self.dim_mode == '2D':
            self.step = self.step2D
        elif self.dim_mode == '3D':
            self.step = self.step3D
        else:
            raise ValueError(
                'QuadEnv: Unknown dimensionality mode %s' % self.dim_mode)
        self.step_func = self.step

    def action_space(self, dynamics):
        if not self.zero_action_middle:
            # Range of actions 0 .. 1
            self.low = np.zeros(2)
            self.bias = 0
            self.scale = 1.0
        else:
            # Range of actions -1 .. 1
            self.low = -np.ones(2)
            self.bias = 1.0
            self.scale = 0.5
        self.high = np.ones(2)
        return spaces.Box(self.low, self.high, dtype=np.float32)

    # modifies the dynamics in place.
    # @profile
    def step3D(self, dynamics, action, goal, dt, observation=None):
        # print('action: ', action)
        action = self.scale * (action + self.bias)
        action = np.clip(action, a_min=self.low, a_max=self.high)
        dynamics.step(
            np.array([action[0], action[0], action[1], action[1]]), dt)

    # modifies the dynamics in place.
    # @profile
    def step2D(self, dynamics, action, goal, dt, observation=None):
        # print('action: ', action)
        action = self.scale * (action + self.bias)
        action = np.clip(action, a_min=self.low, a_max=self.high)
        dynamics.step(np.array(action), dt)


# jacobian of (acceleration magnitude, angular acceleration)
#       w.r.t (normalized motor thrusts) in range [0, 1]
def quadrotor_jacobian(dynamics):
    torque = dynamics.thrust_max * dynamics.prop_crossproducts.T
    torque[2, :] = dynamics.torque_max * dynamics.prop_ccw
    thrust = dynamics.thrust_max * np.ones((1, 4))
    dw = (1.0 / dynamics.inertia)[:, None] * torque
    dv = thrust / dynamics.mass
    J = np.vstack([dv, dw])
    J_cond = np.linalg.cond(J)
    # assert J_cond < 100.0
    if J_cond > 50:
        print("WARN: Jacobian conditioning is high: ", J_cond)
    return J


# P-only linear controller on angular velocity.
# direct (ignoring motor lag) control of thrust magnitude.
class OmegaThrustControl(object):
    def __init__(self, dynamics):
        jacobian = quadrotor_jacobian(dynamics)
        self.Jinv = np.linalg.inv(jacobian)

    def action_space(self, dynamics):
        circle_per_sec = 2 * np.pi
        max_rp = 5 * circle_per_sec
        max_yaw = 1 * circle_per_sec
        min_g = -1.0
        max_g = dynamics.thrust_to_weight - 1.0
        low = np.array([min_g, -max_rp, -max_rp, -max_yaw])
        high = np.array([max_g, max_rp, max_rp, max_yaw])
        return spaces.Box(low, high, dtype=np.float32)

    # modifies the dynamics in place.
    # @profile
    def step(self, dynamics, action, dt):
        kp = 5.0  # could be more aggressive
        omega_err = dynamics.omega - action[1:]
        dw_des = -kp * omega_err
        acc_des = GRAV * (action[0] + 1.0)
        des = np.append(acc_des, dw_des)
        thrusts = np.matmul(self.Jinv, des)
        thrusts[thrusts < 0] = 0
        thrusts[thrusts > 1] = 1
        dynamics.step(thrusts, dt)


# TODO: this has not been tested well yet.
class VelocityYawControl(object):
    def __init__(self, dynamics):
        jacobian = quadrotor_jacobian(dynamics)
        self.Jinv = np.linalg.inv(jacobian)

    def action_space(self, dynamics):
        vmax = 20.0  # meters / sec
        dymax = 4 * np.pi  # radians / sec
        high = np.array([vmax, vmax, vmax, dymax])
        return spaces.Box(-high, high, dtype=np.float32)

    # @profile
    def step(self, dynamics, action, dt):
        # needs to be much bigger than in normal controller
        # so the random initial actions in RL create some signal
        kp_v = 5.0
        kp_a, kd_a = 100.0, 50.0

        e_v = dynamics.vel - action[:3]
        acc_des = -kp_v * e_v + npa(0, 0, GRAV)

        # rotation towards the ideal thrust direction
        # see Mellinger and Kumar 2011
        R = dynamics.rot
        zb_des, _ = normalize(acc_des)
        yb_des, _ = normalize(cross(zb_des, R[:, 0]))
        xb_des = cross(yb_des, zb_des)
        R_des = np.column_stack((xb_des, yb_des, zb_des))

        def vee(R):
            return np.array([R[2, 1], R[0, 2], R[1, 0]])

        e_R = 0.5 * vee(np.matmul(R_des.T, R) - np.matmul(R.T, R_des))
        omega_des = np.array([0, 0, action[3]])
        e_w = dynamics.omega - omega_des

        dw_des = -kp_a * e_R - kd_a * e_w
        # we want this acceleration, but we can only accelerate in one direction!
        # thrust_mag = np.dot(acc_des, dynamics.rot[:,2])
        thrust_mag = get_blas_funcs("thrust_mag", [acc_des, dynamics.rot[:, 2]])

        des = np.append(thrust_mag, dw_des)
        thrusts = np.matmul(self.Jinv, des)
        thrusts = np.clip(thrusts, a_min=0.0, a_max=1.0)
        dynamics.step(thrusts, dt)


# this is an "oracle" policy to drive the quadrotor towards a goal
# using the controller from Mellinger et al. 2011
class MellingerController(object):
    # @profile
    def __init__(self, dynamics, sbc_radius, sbc_aggressive, room_box):
        jacobian = quadrotor_jacobian(dynamics)
        self.Jinv = np.linalg.inv(jacobian)
        # Jacobian inverse for our quadrotor
        # Jinv = np.array([[0.0509684, 0.0043685, -0.0043685, 0.02038736],
        #                 [0.0509684, -0.0043685, -0.0043685, -0.02038736],
        #                 [0.0509684, -0.0043685,  0.0043685,  0.02038736],
        #                 [0.0509684,  0.0043685,  0.0043685, -0.02038736]])
        self.action = None

        self.kp_p, self.kd_p = 4.5, 3.5
        self.kp_a, self.kd_a = 200.0, 50.0

        self.rot_des = np.eye(3)

        self.enable_sbc = True
        # maximum_linf_acceleration, max_acc in any dimension
        self.sbc = NominalSBC(
            maximum_linf_acceleration=2.0, aggressiveness=sbc_aggressive,
            radius=sbc_radius, room_box=room_box)
        self.sbc_last_safe_acc = None

        self.step_func = self.step

    # modifies the dynamics in place.
    # @profile

    def step(self, dynamics, acc_des, dt, observation=None):
        # to_goal = goal - dynamics.pos
        # e_p = -clamp_norm(to_goal, 4.0)
        # e_v = dynamics.vel
        # acc_des = -self.kp_p * e_p - self.kd_p * e_v
        sbc_distance_to_boundary = None

        if self.enable_sbc and observation is not None:
            new_acc, sbc_distance_to_boundary = self.sbc.plan(
                observation["self_state"],
                observation["neighbor_descriptions"],
                acc_des)

            if new_acc is not None:
                self.sbc_last_safe_acc = new_acc
                acc_des = new_acc
            else:
                if self.sbc_last_safe_acc is not None:
                    acc_des = self.sbc_last_safe_acc

        # Question: Why do we need to do this???
        acc_des_without_grav = np.array(acc_des)
        acc_des += np.array([0, 0, GRAV])
        xc_des = self.rot_des[:, 0]

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

        thrusts = np.matmul(self.Jinv, des)
        thrusts[thrusts < 0] = 0
        thrusts[thrusts > 1] = 1

        dynamics.step(thrusts, dt)
        self.action = thrusts.copy()
        # TODO: Check with Baskin, originally, it was new_acc
        return self.action, acc_des_without_grav, sbc_distance_to_boundary

    # def action_space(self, dynamics):
    #     circle_per_sec = 2 * np.pi
    #     max_rp = 5 * circle_per_sec
    #     max_yaw = 1 * circle_per_sec
    #     min_g = -1.0
    #     max_g = dynamics.thrust_to_weight - 1.0
    #     low = np.array([min_g, -max_rp, -max_rp, -max_yaw])
    #     high = np.array([max_g, max_rp, max_rp, max_yaw])
    #     return spaces.Box(low, high, dtype=np.float32)

from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario


class Scenario_static_diff_goal(QuadrotorScenario):
    def step(self, infos, pos):
        return infos

__author__ = 'yuwenhao'

import numpy as np
from gym import utils
from gym_dart.envs import dart_env

# swing up and balance of double inverted pendulum
class DartDoubleInvertedPendulumEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        control_bounds = np.array([[1.0],[-1.0]])
        self.action_scale = 40
        dart_env.DartEnv.__init__(
            self, 'inverted_double_pendulum.skel', 2, 8, control_bounds, dt=0.01)
        utils.EzPickle.__init__(self)

        self.init_qpos = np.array(self.robot_skeleton.q).copy()
        self.init_qvel = np.array(self.robot_skeleton.dq).copy()

    def _step(self, a):

        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[0] = a[0] * self.action_scale

        self.do_simulation(tau, self.frame_skip)
        ob = self._get_obs()

        base = self.robot_skeleton.name_to_body['cart'].to_world()[1]
        weight_sz = 0.02 # TODO: Make sure this doesn't change.
        max_height = 0.6
        raw_height = self.robot_skeleton.name_to_body['weight'].to_world()[1]
        # Have the same scaling as the gym env.
        height = 2.0*(raw_height - base - weight_sz)/max_height

        v1, v2 = self.robot_skeleton.dq[1:3]

        alive_bonus = 10.
        dist_penalty = 0.01*ob[0]**2 + (height - 2.)**2
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        reward = alive_bonus - dist_penalty - vel_penalty

        done = bool(height <= 1)
        return ob, reward, done, {}


    def _get_obs(self):
        return np.concatenate([
            self.robot_skeleton.q[:1],
            np.sin(self.robot_skeleton.q[1:]),
            np.cos(self.robot_skeleton.q[1:]),
            self.robot_skeleton.dq
        ]).ravel()

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.init_qpos + \
               self.np_random.uniform(low=-.1, high=.1, size=self.robot_skeleton.ndofs)
        qvel = self.init_qvel + \
               self.np_random.randn(self.robot_skeleton.ndofs) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()


    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -3.5
        self._get_viewer().scene.tb._set_theta(0)
        self.track_skeleton_id = 0

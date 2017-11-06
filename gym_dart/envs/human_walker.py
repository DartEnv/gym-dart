__author__ = 'yuwenhao'

import numpy as np
from gym import utils
from gym_dart.envs import dart_env
import joblib
import os

import time

import pydart2 as pydart

# human model with human-like joint limit
# Refer to https://arxiv.org/abs/1709.08685 for more details
class DartHumanWalkerEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0] * 23, [-1.0] * 23])
        self.action_scale = np.array([120, 120, 120, 100, 60, 60, 120, 120, 120, 100, 60, 60, 100, 100, 100, 80,80,80, 50, 80,80,80, 50])*1.5
        obs_dim = 57

        self.t = 0
        self.target_vel = 1.0
        # state related
        self.contact_info = np.array([0, 0])
        self.include_additional_info = True
        if self.include_additional_info:
            obs_dim += len(self.contact_info)

        dart_env.DartEnv.__init__(self, 'kima/kima_human_edited.skel', 15, obs_dim, self.control_bounds,
                                      disableViewer=True, dt=0.002)

        # add human joint limit
        # Dart with modified joint limit is required: https://github.com/jyf588/dart/tree/human-joint-constraints
        self.use_human_jointlimit = False
        if self.use_human_jointlimit: #
            skel = self.robot_skeleton
            world = self.dart_world
            leftarmConstraint = pydart.constraints.HumanArmJointLimitConstraint(skel.joint('j_bicep_left'),
                                                                                skel.joint('j_forearm_left'), False)
            rightarmConstraint = pydart.constraints.HumanArmJointLimitConstraint(skel.joint('j_bicep_right'),
                                                                                 skel.joint('j_forearm_right'), True)
            leftlegConstraint = pydart.constraints.HumanLegJointLimitConstraint(skel.joint('j_thigh_left'),
                                                                                skel.joint('j_shin_left'),
                                                                                skel.joint('j_heel_left'), False)
            rightlegConstraint = pydart.constraints.HumanLegJointLimitConstraint(skel.joint('j_thigh_right'),
                                                                                 skel.joint('j_shin_right'),
                                                                                 skel.joint('j_heel_right'), True)
            leftarmConstraint.add_to_world(world)
            rightarmConstraint.add_to_world(world)
            leftlegConstraint.add_to_world(world)
            rightlegConstraint.add_to_world(world)

        self.robot_skeleton.set_self_collision_check(False)

        self.sim_dt = self.dt / self.frame_skip

        utils.EzPickle.__init__(self)


    def do_simulation(self, tau, n_frames):
        for _ in range(n_frames):
            self.robot_skeleton.set_forces(tau)
            self.dart_world.step()

    def advance(self, a):
        a*=0
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[6:] = clamped_control * self.action_scale

        self.do_simulation(tau, self.frame_skip)

    def _step(self, a):
        posbefore = self.robot_skeleton.bodynodes[1].com()[0]
        self.advance(np.copy(a))
        posafter = self.robot_skeleton.bodynodes[1].com()[0]
        height = self.robot_skeleton.bodynode('head').com()[1]
        side_deviation = self.robot_skeleton.bodynode('head').com()[2]
        angle = self.robot_skeleton.q[3]

        upward = np.array([0, 1, 0])
        upward_world = self.robot_skeleton.bodynode('head').to_world(np.array([0, 1, 0])) - self.robot_skeleton.bodynode('head').to_world(np.array([0, 0, 0]))
        upward_world /= np.linalg.norm(upward_world)
        ang_cos_uwd = np.dot(upward, upward_world)
        ang_cos_uwd = np.arccos(ang_cos_uwd)

        forward = np.array([1, 0, 0])
        forward_world = self.robot_skeleton.bodynode('head').to_world(np.array([1, 0, 0])) - self.robot_skeleton.bodynode('head').to_world(np.array([0, 0, 0]))
        forward_world /= np.linalg.norm(forward_world)
        ang_cos_fwd = np.dot(forward, forward_world)
        ang_cos_fwd = np.arccos(ang_cos_fwd)

        contacts = self.dart_world.collision_result.contacts
        total_force_mag = 0
        self.contact_info = np.array([0, 0])
        for contact in contacts:
            total_force_mag += np.square(contact.force).sum()
            if contact.skel_id1 + contact.skel_id2 == 1:
                if contact.bodynode1 == self.robot_skeleton.bodynode('l-foot') or contact.bodynode2 == self.robot_skeleton.bodynode('l-foot'):
                    self.contact_info[0] = 1
                if contact.bodynode1 == self.robot_skeleton.bodynode('r-foot') or contact.bodynode2 == self.robot_skeleton.bodynode('r-foot'):
                    self.contact_info[1] = 1


        alive_bonus = 2.0
        vel = (posafter - posbefore) / self.dt
        vel_rew = 2 * (
            self.target_vel - np.abs(self.target_vel - vel))

        action_pen = 0.5 * np.abs(a).sum()
        deviation_pen = 3 * abs(side_deviation)
        reward = vel_rew + alive_bonus - action_pen - deviation_pen


        self.t += self.dt

        s = self.state_vector()

        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height-self.init_height > -0.2) and (height-self.init_height < 1.0) and (abs(ang_cos_uwd) < 2.0) and (abs(ang_cos_fwd) < 2.0)
                    and np.abs(angle) < 1.3 and np.abs(self.robot_skeleton.q[5]) < 0.4 and np.abs(side_deviation) < 0.9)

        if done:
            reward = 0

        ob = self._get_obs()

        broke_sim = False
        if not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all()):
            broke_sim = True

        return ob, reward, done, {'broke_sim': broke_sim, 'vel_rew': vel_rew, 'action_pen': action_pen,
                                  'deviation_pen': deviation_pen, 'done_return': done,
                                  'dyn_model_id': 0, 'state_index': 0}

    def _get_obs(self):
        state = np.concatenate([
            self.robot_skeleton.q[1:],
            np.clip(self.robot_skeleton.dq, -10, 10),
        ])

        if self.include_additional_info:
            state = np.concatenate([state, self.contact_info])

        return state

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.05, high=.05, size=self.robot_skeleton.ndofs)

        self.set_state(qpos, qvel)
        self.t = 0

        self.init_pos = self.robot_skeleton.q[0]

        self.contact_info = np.array([0, 0])

        self.init_height = self.robot_skeleton.bodynode('head').C[1]

        return self._get_obs()

    def viewer_setup(self):
        if not self.disableViewer:
            self._get_viewer().scene.tb.trans[2] = -5.5

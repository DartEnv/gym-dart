from gym.envs.registration import register

register(
    id='dart-v0',
    entry_point='gym_dart.envs:DartEnv',
)

# Dart
# ----------------------------------------
# Obtained from https://github.com/DartEnv/dart-env/blob/master/gym/envs/__init__.py


register(
    id='DartHopper-v1',
    entry_point='gym_dart.envs:DartHopperEnv',
    reward_threshold=3800.0,
    max_episode_steps=1000,
)

register(
    id='DartCartPole-v1',
    entry_point='gym_dart.envs:DartCartPoleEnv',
    reward_threshold=950.0,
    max_episode_steps=1000,
)

register(
    id='DartDoubleInvertedPendulumEnv-v1',
    entry_point='gym_dart.envs:DartDoubleInvertedPendulumEnv',
    max_episode_steps=1000,
)

register(
    id='DartReacher-v1',
    entry_point='gym_dart.envs:DartReacher2dEnv',
    reward_threshold=-3.75,
    max_episode_steps=50,
)

register(
    id='DartReacher3d-v1',
    entry_point='gym_dart.envs:DartReacherEnv',
    reward_threshold=-200,
    max_episode_steps=500,
)

register(
    id='DartDog-v1',
    entry_point='gym_dart.envs:DartDogEnv',
    max_episode_steps=1000,
)

register(
    id='DartCartPoleImg-v1',
    entry_point='gym_dart.envs:DartCartPoleImgEnv',
    reward_threshold=950.0,
    max_episode_steps=2000,
)

register(
    id='DartCartPoleSwingUp-v1',
    entry_point='gym_dart.envs:DartCartPoleSwingUpEnv',
    max_episode_steps=500,
)

register(
    id='DartWalker2d-v1',
    entry_point='gym_dart.envs:DartWalker2dEnv',
    max_episode_steps=1000,
)

register(
    id='DartWalker3d-v1',
    entry_point='gym_dart.envs:DartWalker3dEnv',
    max_episode_steps=1000,
)

register(
    id='DartWalker3dSPD-v1',
    entry_point='gym_dart.envs:DartWalker3dSPDEnv',
    max_episode_steps=1000,
)

register(
    id='DartHumanWalker-v1',
    entry_point='gym_dart.envs:DartHumanWalkerEnv',
    max_episode_steps=300,
)

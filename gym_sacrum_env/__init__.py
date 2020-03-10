from gym.envs.registration import register

register(
    id='sacrum_nav-v0',
    entry_point='gym_sacrum_env.envs:SacrumNavEnv'
)

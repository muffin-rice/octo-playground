model_config = {}

loss_config = {
    # various hyperparameters
    "LR": 2e-4,  # value used in walkthrough
    "MAX_GRAD_NORM": 0.5,  # value used in walkthrough
    "GAMMA": 0.99,  # used to calculate advantage at current step. A ~= r - V + V_{t+1} * gamma
    "GAE_LAMBDA": 0.95,  # used to calculate decay. A_t = r - V + V_{t+1} * gamma + A_{t+1} * gamma * lambda
    "CLIP_EPS": 0.2,
    # used for PPO to guarantee ratio used in backprop does not exceed (1-e, 1+e) unless it's in the right direction
    "CLIP_VAL": 100,  # TODO: is this needed? same as above, but is used for critic loss
    # loss coefficients
    "ACTOR_LOSS": 1,  # value used in walkthrough
    "CRITIC_LOSS": 0.5,  # value used in walkthrough
    "ENTROPY_LOSS": 0.01,  # value used in walkthrough
}

env_config = {
    "metadata": "surprise!",
    # env coefficients
    "STARTING_KEY": 0,
    "ENV_NAME": "overcooked",
    # "ENV_KWARGS": { "layout": "cramped_room" },
    "ENV_KWARGS": {},
    "NUM_ENVS": 3,
    "NUM_EPISODES": 100000,
    "NUM_STEPS": 17,  # episode length. value used in walkthrough
    "NUM_AGENTS": 2,
    # checkpointing
    "SAVE_FILE": "src/ppo_marl/ckpt/checkpoint",
    "CKPT_SAVE": 10000,
    "CONTINUE": False,
    "PREVIOUS_SAVE": "src/ppo_marl/ckpt/checkpoint",  # is only used when continue is true
    # tensorboard
    "JAX_PROFILER_SERVER": 9999,
    "TENSORBOARD_LOGDIR": "data/tensorboard/ppo",
}

visualizer_config = {
    "ENV_NAME": "overcooked",
    "ENV_KWARGS": {},
    "NUM_STEPS": 100,
    "PREVIOUS_SAVE": "src/ppo_marl/ckpt/checkpoint",
}

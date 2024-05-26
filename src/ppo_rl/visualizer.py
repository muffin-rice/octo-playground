# metal does not work. comment out if it does work
import jax

jax.config.update("jax_platform_name", "cpu")

import os
import sys
from flax import struct
import gymnasium as gym
import pickle as pkl

sys.path[0] = os.getcwd()

import gymnax
from gymnash.gymnax.visualize import Visualizer
import jax

from src.ppo_rl.config import visualizer_config
from src.ppo_rl.ppo_utils import get_env_step_function, load_train_state
from src.ppo_rl.ppo_types import (
    create_trajectory_from_transitions,
    TrajectoryState,
    Transition,
)
from src.logger import Logger

LOGGER = Logger("vis.py")


# hack as i can't pull it out of the lib
@struct.dataclass
class EnvState:
    position: float
    velocity: float
    time: int


def get_state(state: EnvState, state_index=0) -> EnvState:
    return EnvState(
        state.position[state_index],
        state.velocity[state_index],
        state.time[state_index],
    )


if __name__ == "__main__":
    LOGGER.info("Hello!")

    # initialize env
    env, env_params = gymnax.make(
        visualizer_config["ENV_NAME"], **visualizer_config["ENV_KWARGS"]
    )
    key, train_state, starting_epoch = load_train_state(
        visualizer_config["PREVIOUS_SAVE"]
    )

    # extract values from train_state
    model, old_model, optim, opt_state = train_state
    model_name = visualizer_config["PREVIOUS_SAVE"].split("/")[-1]

    # initialize values
    key, key_unsplit = jax.random.split(key)
    key_envs = jax.random.split(
        key_unsplit, visualizer_config["NUM_ENVS"]
    )  # non-1 num envs to ensure array shaping works
    # reset env
    env_obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(key_envs, env_params)

    trajectory_state = TrajectoryState(key, env_state, env_obs)

    func_env_step = get_env_step_function(
        (env, env_params), model, visualizer_config["NUM_ENVS"], 0
    )  # 1 env

    # get list of states
    state_seq = [trajectory_state.env_state]
    transitions: [Transition] = []
    for i in range(visualizer_config["NUM_STEPS"]):
        trajectory_state, transition = func_env_step(trajectory_state, None)
        state_seq.append(trajectory_state.env_state)
        transitions.append(transition)

    LOGGER.info(f"Saving {len(state_seq)} states")

    if not os.path.exists(f"data/ppo/{model_name}"):
        os.mkdir(f"data/ppo/{model_name}")

    with open(f"data/ppo/{model_name}/transitions.pkl", "wb") as f:
        pkl.dump(transitions, f)

    gif_dir = f"data/ppo/{model_name}/visualizations"
    if not os.path.exists(gif_dir):
        os.mkdir(gif_dir)

    for env_number in range(visualizer_config["NUM_ENVS"]):
        env = gym.make(visualizer_config["ENV_NAME"], render_mode="human")
        observation, info = env.reset()

        for transition in transitions:
            env.step(transition.action[env_number].item())

        env.close()

    LOGGER.info(
        f"Total reward for agent_1: {create_trajectory_from_transitions(transitions).t_reward.sum()}"
    )

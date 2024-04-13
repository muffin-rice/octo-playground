# metal does not work. comment out if it does work
import jax

jax.config.update("jax_platform_name", "cpu")

import os
import sys
import pickle as pkl

sys.path[0] = os.getcwd()

import JaxMARL.jaxmarl as jaxmarl
from JaxMARL.jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
from JaxMARL.jaxmarl.environments.overcooked.overcooked import State as OvercookedState
import jax

from src.ppo_marl.config import visualizer_config
from src.ppo_marl.ppo_marl_utils import get_env_step_function, load_train_state
from src.ppo_marl.ppo_types import TrajectoryState
from src.logger import Logger

LOGGER = Logger("vis.py")


def reduce_state(state: OvercookedState) -> OvercookedState:
    return OvercookedState(
        state.agent_pos[0],
        state.agent_dir[0],
        state.agent_dir_idx[0],
        state.agent_inv[0],
        state.goal_pos[0],
        state.pot_pos[0],
        state.wall_map[0],
        state.maze_map[0],
        state.time,
        state.terminal,
    )


if __name__ == "__main__":
    LOGGER.info("Hello!")

    # initialize env
    env = jaxmarl.make(visualizer_config["ENV_NAME"], **visualizer_config["ENV_KWARGS"])
    key, train_state, starting_epoch = load_train_state(
        visualizer_config["PREVIOUS_SAVE"]
    )

    # extract values from train_state
    model, old_model, optim, opt_state = train_state
    model_name = visualizer_config["PREVIOUS_SAVE"].split("/")[-1]

    # initialize values
    key, key_unsplit = jax.random.split(key)
    key_envs = jax.random.split(
        key_unsplit, 3
    )  # non-1 num envs to ensure array shaping works
    # reset env
    env_obs, env_state = jax.vmap(env.reset)(key_envs)

    trajectory_state = TrajectoryState(key, env_state, env_obs)

    func_env_step = get_env_step_function(env, model, 3, 2)  # 1 env

    # get list of states
    state_seq = [reduce_state(trajectory_state.env_state)]
    transitions = []
    for i in range(visualizer_config["NUM_STEPS"]):
        trajectory_state, transition = func_env_step(trajectory_state, None)
        state_seq.append(reduce_state(trajectory_state.env_state))
        transitions.append(transition)

    LOGGER.info(f"Saving {len(state_seq)} states")

    with open(f"data/ppo/{model_name}/actions.pkl", "wb") as f:
        pkl.dump(transitions, f)

    viz = OvercookedVisualizer()
    viz.animate(
        state_seq,
        env.agent_view_size,
        filename=f"data/ppo/{model_name}/visualization.gif",
    )

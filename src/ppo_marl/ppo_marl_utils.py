import dill
from typing import Any, Callable, Dict

import distrax
import equinox as eqx
import JaxMARL.jaxmarl as jaxmarl
import JaxMARL.jaxmarl.environments
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, UInt32
import numpy as np

from src.ppo_marl.ppo_types import (
    LossInformation,
    Trajectory,
    Transition,
    TrainState,
    TrajectoryState,
)
from src.logger import Logger, LoggingLevel


def get_env_step_function(
    env: jaxmarl.environments.MultiAgentEnv, model: eqx.Module, num_envs, num_agents
) -> Callable:
    """Wrap env_step in a returnable function as jax.lax.scan requires all objects
    in the carry state to be jax-able, and the env is not jax-able."""

    LOGGER = Logger("env_step", logging_level=LoggingLevel.DEBUG)

    def env_step(trajectory_state: TrajectoryState, _) -> (Any, Transition):
        """Takes a step in the batch of environments given the model's pi and val for a single step(minibatch).
        To be used in jax.lax.scan to grab the entire trajectory.
        Returns shared first arg (trajectory_state) and the transition minibatch"""

        key, env_state, env_obs = trajectory_state
        agent_list: [str] = env_obs.keys()

        # get env_obs as array
        env_obs_array = batchify(env_obs, agent_list, num_envs)

        # predictions for this current state
        model_output = jax.vmap(model)(env_obs_array)
        model_pi: distrax.Categorical = model_output[0]
        model_value: Array = model_output[1]
        LOGGER.debug(
            f"Pi minibatch has num categories {model_pi.num_categories}, "
            f"Value has shape {model_value.shape}"
        )

        # sample from pi
        key, key_action = jax.random.split(key, 2)
        action, log_prob = model_pi.sample_and_log_prob(seed=key_action)
        # put actions into a dictionary with agent name as str and flatten the action array
        # from (num_envs, 1) -> (num_envs,)
        action_unbatch = {
            k: v.flatten()
            for k, v in reverse_batchify(action, agent_list, num_envs).items()
        }

        LOGGER.debug(
            f"Action was chosen with shapes {dtype_as_str(action_unbatch)}\n"
            f"Log prob was calculated with shapes {dtype_as_str(log_prob)}"
        )

        # apply action to env
        key, key_env_unsplit = jax.random.split(key, 2)
        key_env = jax.random.split(key_env_unsplit, num_envs)

        obs_batch, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0)
        )(key_env, env_state, action_unbatch)
        LOGGER.debug(
            f"Observations have shape: {dtype_as_str(obs_batch)}\n"
            f"Reward has shape: {dtype_as_str(reward)}\n"
            f"Done has shape: {dtype_as_str(done)}\n"
            f"Info is: {info}"
        )

        info = jax.tree.map(lambda x: x.reshape((num_agents)), info)
        LOGGER.debug(f"Transformed info into {info}")

        transition = Transition(
            done,
            action_unbatch,
            reward,
            obs_batch,
            info,
            model_value,
            log_prob,
        )

        new_trajectory_state = TrajectoryState(key, env_state, obs_batch)

        return new_trajectory_state, transition

    return env_step


def batchify(
    env_attr: dict, agent_list: [str], num_envs: int, additional_squeeze: bool = False
) -> Float[Array, ""]:
    """JaxMARL envs have attributes agent_id: array. Unpack this dictionary
    into an array to be completely vectorized"""
    # make arr num_agents x num_envs x (optional if trajectory) x (dtype)
    env_attr_array = jnp.stack([env_attr[agent] for agent in agent_list])

    if (
        additional_squeeze
    ):  # if we want to squeeze one additional dim, eg. (optional if trajectory)
        return env_attr_array.reshape(
            (len(agent_list) * num_envs * env_attr_array.shape[2], -1)
        )
    else:
        return env_attr_array.reshape((len(agent_list) * num_envs, -1))


def reverse_batchify(
    env_attr_array: Float[Array, ""], agent_list: [str], num_envs: int
):
    """Reverse of batchify. However, expressiveness is somewhat lost in the array as
    it collapses all dims into one."""
    env_attr_array = env_attr_array.reshape((len(agent_list), num_envs, -1))
    return {agent: env_attr_array[i] for i, agent in enumerate(agent_list)}


def dtype_as_str(x: Any) -> str:
    if isinstance(x, Array):
        return f"shape {x.shape} dtype {x.dtype}"

    if isinstance(x, JaxMARL.jaxmarl.environments.overcooked.overcooked.State):
        return dict_as_str(x.__dict__)

    if isinstance(x, dict):
        return dict_as_str(x)

    if isinstance(x, list):
        # limit to not print too much
        return "[" + "\n\t".join(dtype_as_str(y) for y in x[:5]) + "]"

    if isinstance(x, str):
        return x

    if isinstance(x, (Trajectory, Transition)):
        return dict_as_str(x._asdict())

    if isinstance(x, LossInformation):
        return dict_as_str({k: np.array(v) for k, v in x._asdict().items()})

    if isinstance(x, np.ndarray):
        return str(x)

    return str(type(x))


def dict_as_str(d: Dict) -> str:
    """For an array dict with str: array, transform the dictionary as
    a str: array shape string. Useful as MARL envs are all dicts with
    agent_id as the key."""
    s = "\n\t".join(f"{k}: {dtype_as_str(v)}" for k, v in d.items())
    return "{" + s + "}"


def save_train_state(
    filename: str, key: UInt32[Array, ""], train_state: TrainState, epoch_num: int
) -> None:
    # TODO: save with eqx serialization and avoid dill
    with open(filename + ".ckpt", "wb") as f:
        dill.dump((key, train_state, epoch_num), f)


def load_train_state(filename: str) -> (UInt32[Array, ""], TrainState, int):
    with open(filename + ".ckpt", "rb") as f:
        return dill.load(f)

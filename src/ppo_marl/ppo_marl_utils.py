import jax.numpy as jnp
from jaxtyping import Array, Float
from typing import Any, Dict
from src.ppo_marl.ppo_types import Trajectory, Transition

import JaxMARL.jaxmarl.environments


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

    if isinstance(x, Transition):
        return dict_as_str(
            {
                "done": x.done,
                "action": x.action,
                "reward": x.reward,
                "obs": x.obs,
                "info": x.info,
                "model_value": x.model_value,
                "log_prob": x.log_prob,
            }
        )

    if isinstance(x, Trajectory):
        return dict_as_str(
            {
                "t_done": x.t_done,
                "t_action": x.t_action,
                "t_reward": x.t_reward,
                "t_obs": x.t_obs,
                "t_model_value": x.t_model_value,
                "t_log_prob": x.t_log_prob,
            }
        )

    return str(type(x))


def dict_as_str(d: Dict) -> str:
    """For an array dict with str: array, transform the dictionary as
    a str: array shape string. Useful as MARL envs are all dicts with
    agent_id as the key."""
    s = "\n\t".join(f"{k}: {dtype_as_str(v)}" for k, v in d.items())
    return "{" + s + "}"

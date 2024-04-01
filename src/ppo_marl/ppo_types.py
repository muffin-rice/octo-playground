from typing import NamedTuple, Dict

import jax.numpy as jnp
from jaxtyping import Array, Float, Int
import JaxMARL.jaxmarl as jaxmarl
import JaxMARL.jaxmarl.environments
import equinox as eqx
import optax
import chex


class JaxKey(Array):
    def __init__(self, shape):
        super().__init__(shape)


class Transition(NamedTuple):
    """Data for a single step."""

    done: Dict[str, Int[Array, "num_envs"]]
    action: Dict[str, Float[Array, "num_envs"]]
    reward: Dict[str, Float[Array, "num_envs"]]
    obs: Dict[str, Float[Array, "num_envs obs_dim"]]
    info: {}
    # model info is not indexed by agent
    model_value: Float[Array, "batch"]
    log_prob: Float[Array, "batch action_dim"]


class Trajectory(NamedTuple):
    """Essentially a list of transitions, but batched into an array."""

    t_done: Dict[str, Int[Array, "num_envs d"]]
    t_action: Dict[str, Float[Array, "num_envs d"]]
    t_reward: Dict[str, Float[Array, "num_envs d"]]
    t_obs: Dict[str, Float[Array, "num_envs d obs_dim"]]
    # model info
    t_model_value: Float[Array, "batch d"]
    t_log_prob: Float[Array, "batch d action_dim"]


class TrajectoryState(NamedTuple):
    """State for a trajectory. Maintained as trajectory is being called"""

    key: Array
    env_state: jaxmarl.environments.State
    env_obs: Dict[str, chex.Array]


class TrainState(NamedTuple):
    """Train state to persist. Should be fully contained with make_full_step or whatever that API is."""

    model: eqx.Module
    old_model: eqx.Module
    optim: optax.GradientTransformation
    opt_state: optax.OptState


class LossInformation(NamedTuple):
    """Loss information"""

    actor_loss: Float[Array, ""]
    critic_loss: Float[Array, ""]
    entropy_loss: Float[Array, ""]


def create_trajectory_from_transitions(
    transition_list: [Transition], agent_list: [str]
) -> Trajectory:
    """Helper function to map a list of transitions into a trajectory"""
    return Trajectory(
        {
            agent: jnp.stack(
                [transition.done[agent] for transition in transition_list], axis=1
            )
            for agent in agent_list
        },
        {
            agent: jnp.stack(
                [transition.action[agent] for transition in transition_list], axis=1
            )
            for agent in agent_list
        },
        {
            agent: jnp.stack(
                [transition.reward[agent] for transition in transition_list], axis=1
            )
            for agent in agent_list
        },
        {
            agent: jnp.stack(
                [transition.obs[agent] for transition in transition_list], axis=1
            )
            for agent in agent_list
        },
        jnp.stack([transition.model_value for transition in transition_list], axis=1),
        jnp.stack([transition.log_prob for transition in transition_list], axis=1),
    )


def create_transitions_from_trajectory(
    trajectory: Trajectory, agent_list: [str]
) -> [Transition]:
    """Helper function to map a trajectory into a list of transitions
    Loses info information"""
    num_steps = trajectory.t_done[agent_list[0]].shape[1]
    return [
        Transition(
            {agent: trajectory.t_done[agent][:, i] for agent in agent_list},
            {agent: trajectory.t_action[agent][:, i] for agent in agent_list},
            {agent: trajectory.t_reward[agent][:, i] for agent in agent_list},
            {agent: trajectory.t_obs[agent][:, i] for agent in agent_list},
            {},
            trajectory.t_model_value[:, i],
            trajectory.t_log_prob[:, i],
        )
        for i in range(num_steps)
    ]


def create_transition_list_from_transitions(
    transitions: Transition, agent_list: [str]
) -> [Transition]:
    """Helper function to map the post-scan transition into a list of transitions
    Loses info information"""
    num_steps = transitions.done[agent_list[0]].shape[0]
    return [
        Transition(
            {agent: transitions.done[agent][i, :] for agent in agent_list},
            {agent: transitions.action[agent][i, :] for agent in agent_list},
            {agent: transitions.reward[agent][i, :] for agent in agent_list},
            {agent: transitions.obs[agent][i, :] for agent in agent_list},
            {},
            transitions.model_value[i, :],
            transitions.log_prob[i, :],
        )
        for i in range(num_steps)
    ]

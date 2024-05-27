import chex
import distrax
import equinox as eqx
import gymnax
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int
import optax
from typing import NamedTuple, Dict

from src.logger import Logger, LoggingLevel


class ActorCriticDiscrete(eqx.Module):
    """Actor Critic class for PPO. Discrete action dimension."""

    actor_layers: []
    critic_layers: []
    LOGGER = Logger("actor_critic", logging_level=LoggingLevel.DEBUG)

    def __init__(self, key: Array, observation_dim: int, action_dim: int, config: dict):
        self.LOGGER.info(
            f"Creating model with key {key}, observation dim {observation_dim}, action dim {action_dim}"
        )
        keyx = jax.random.split(key, 10)

        self.actor_layers = [
            eqx.nn.Linear(observation_dim, 64, key=keyx[0]),
            jax.nn.tanh,
            eqx.nn.Linear(64, 64, key=keyx[1]),
            jax.nn.tanh,
            eqx.nn.Linear(64, action_dim, key=keyx[2]),
        ]

        self.critic_layers = [
            eqx.nn.Linear(observation_dim, 64, key=keyx[3]),
            jax.nn.tanh,
            eqx.nn.Linear(64, 64, key=keyx[4]),
            jax.nn.tanh,
            eqx.nn.Linear(64, 1, key=keyx[5]),
        ]

    def __call__(self, x: Array) -> (distrax.Categorical, Array):
        self.LOGGER.debug(f"Input received with shape {x.shape}")
        actor_mean = x
        for actor_layer in self.actor_layers:
            actor_mean = actor_layer(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic_output = x
        for critic_layer in self.critic_layers:
            critic_output = critic_layer(critic_output)

        return pi, jnp.squeeze(critic_output, axis=-1)


class ActorCriticContinuous(eqx.Module):
    """Actor Critic class for PPO. Continuous action dimension"""

    actor_layers: []
    critic_layers: []
    LOGGER = Logger("actor_critic", logging_level=LoggingLevel.DEBUG)

    def __init__(self, key: Array, observation_dim: int, action_dim: int, config: dict):
        self.LOGGER.info(
            f"Creating model with key {key}, observation dim {observation_dim}, action dim {action_dim}"
        )
        keyx = jax.random.split(key, 10)

        self.actor_layers = [
            eqx.nn.Linear(observation_dim, 64, key=keyx[0]),
            jax.nn.tanh,
            eqx.nn.Linear(64, 64, key=keyx[1]),
            jax.nn.tanh,
            eqx.nn.Linear(64, action_dim, key=keyx[2]),
        ]

        self.critic_layers = [
            eqx.nn.Linear(observation_dim, 64, key=keyx[3]),
            jax.nn.tanh,
            eqx.nn.Linear(64, 64, key=keyx[4]),
            jax.nn.tanh,
            eqx.nn.Linear(64, 1, key=keyx[5]),
        ]

    def __call__(self, x: Array) -> (distrax.Categorical, Array):
        self.LOGGER.debug(f"Input received with shape {x.shape}")
        actor_mean = x
        for actor_layer in self.actor_layers:
            actor_mean = actor_layer(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic_output = x
        for critic_layer in self.critic_layers:
            critic_output = critic_layer(critic_output)

        return pi, jnp.squeeze(critic_output, axis=-1)


class JaxKey(Array):
    def __init__(self, shape):
        super().__init__(shape)


class Transition(NamedTuple):
    """Data for a single step."""

    done: Int[Array, "num_envs"]
    action: Float[Array, "num_envs"]
    reward: Float[Array, "num_envs"]
    obs: Float[Array, "num_envs obs_dim"]
    info: {}
    # model info
    model_value: Float[Array, "num_envs"]
    log_prob: Float[Array, "num_envs"]


class Trajectory(NamedTuple):
    """Essentially a list of transitions, but batched into an array."""

    t_done: Int[Array, "num_envs d"]
    t_action: Float[Array, "num_envs d"]
    t_reward: Float[Array, "num_envs d"]
    t_obs: Float[Array, "num_envs d obs_dim"]
    # model info
    t_model_value: Float[Array, "num_envs d"]
    t_log_prob: Float[Array, "num_envs d"]


class TrajectoryState(NamedTuple):
    """State for a trajectory. Maintained as trajectory is being called"""

    key: Array
    env_state: gymnax.EnvState
    env_obs: chex.Array
    done: Bool[Array, "num_envs"]


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
    total_reward: Float[Array, ""]  # just a full sum
    total_done: Float[Array, ""]  # number of failures, just a full sum
    gae: Float[Array, ""]  # reward with decay applied to it


def create_trajectory_from_transitions(transition_list: [Transition]) -> Trajectory:
    """Helper function to map a list of transitions into a trajectory"""
    return Trajectory(
        jnp.stack([transition.done for transition in transition_list], axis=1),
        jnp.stack([transition.action for transition in transition_list], axis=1),
        jnp.stack([transition.reward for transition in transition_list], axis=1),
        jnp.stack([transition.obs for transition in transition_list], axis=1),
        jnp.stack([transition.model_value for transition in transition_list], axis=1),
        jnp.stack([transition.log_prob for transition in transition_list], axis=1),
    )


def create_transition_list_from_transitions(transitions: Transition) -> [Transition]:
    """Helper function to map the post-scan transition into a list of transitions
    Loses info information"""
    num_steps = transitions.done.shape[0]
    return [
        Transition(
            transitions.done[i, :],
            transitions.action[i, :],
            transitions.reward[i, :],
            transitions.obs[i, :],
            {},
            transitions.model_value[i, :],
            transitions.log_prob[i, :],
        )
        for i in range(num_steps)
    ]

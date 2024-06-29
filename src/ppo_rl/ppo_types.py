import chex
import distrax
import equinox as eqx
import gymnax
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int
import optax
from typing import Callable, NamedTuple
from gymnax.environments import spaces

from src.logger import Logger, LoggingLevel

model_logger = Logger("actor_critic", logging_level=LoggingLevel.DEBUG)


def get_categorical_forward_function(action_space: spaces.Discrete) -> (Callable, int):
    model_logger.info(
        f"Creating categorical distribution from action space with {action_space.n} categories"
    )
    return lambda model_output: distrax.Categorical(model_output), action_space.n


def get_clipped_normal_forward_function(action_space: spaces.Box) -> (Callable, int):
    model_logger.info(
        f"Creating clipped normal from action space with min {action_space.low}, max {action_space.high}, shape {action_space.shape}"
    )
    # mean + var
    model_output_size = int(jnp.prod(*action_space.shape) * 2)

    def clipped_normal_function(model_input: jnp.ndarray):
        mu = model_input[: model_output_size // 2].reshape(*action_space.shape)
        # force std positive
        std = jax.nn.relu(
            model_input[model_output_size // 2 :].reshape(*action_space.shape)
        )
        return distrax.ClippedNormal(
            loc=mu, scale=std, minimum=action_space.low, maximum=action_space.high
        )

    return clipped_normal_function, model_output_size


def get_tuple_forward_function(action_space: spaces.Tuple) -> (Callable, int):
    model_logger.info(
        f"Creating tuple of Distrax distributions from action space of length {len(action_space.spaces)}"
    )
    tuple_projections = (
        get_action_forward_function(observation_subsubspace)
        for observation_subsubspace in action_space.spaces
    )

    def tuple_function(model_input: jnp.ndarray):
        input_index = 0
        model_outs = []
        for projection in tuple_projections:
            model_outs.append(
                projection[0](model_input[input_index : input_index + projection[1]])
            )
            input_index += projection[1]

        return model_outs

    return tuple_function, sum(projection[1] for projection in tuple_projections)


def get_dict_forward_function(action_space: spaces.Dict) -> (Callable, int):
    model_logger.info(
        f"Creating dict of Distrax distributions from action space of length {len(action_space.spaces)}"
    )
    dict_projections = {
        key: get_action_forward_function(observation_subsubspace)
        for key, observation_subsubspace in action_space.spaces.items()
    }

    def dict_function(model_input: jnp.ndarray):
        input_index = 0
        model_outs = {}
        for key, projection in dict_projections.items():
            model_outs[key] = projection[0](
                model_input[input_index : input_index + projection[1]]
            )
            input_index += projection[1]

        return model_outs

    return dict_function, sum(
        (projection_value[1] for projection_value in dict_projections.values())
    )


def get_action_forward_function(action_space: spaces.Space) -> (Callable, int):
    if isinstance(action_space, spaces.Discrete):
        return get_categorical_forward_function(action_space)
    if isinstance(action_space, spaces.Box):
        return get_clipped_normal_forward_function(action_space)
    if isinstance(action_space, spaces.Tuple):
        return get_tuple_forward_function(action_space)
    if isinstance(action_space, spaces.Dict):
        return get_dict_forward_function(action_space)

    raise NotImplementedError(f"No implementation found for space {type(action_space)}")


class ActorCriticGymnax(eqx.Module):
    """Actor Critic class for Gymnax spaces. Supports all Gymnax spaces."""

    actor_layers: []
    critic_layers: []
    LOGGER = Logger("actor_critic", logging_level=LoggingLevel.DEBUG)

    def __init__(
        self,
        key: Array,
        observation_dim: int,
        action_space: spaces.Space,
        config: dict,
    ):
        self.LOGGER.info(f"Generating forward function for space {type(action_space)}")
        action_projection, action_dim = get_action_forward_function(action_space)

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
            action_projection,
        ]

        self.critic_layers = [
            eqx.nn.Linear(observation_dim, 64, key=keyx[3]),
            jax.nn.tanh,
            eqx.nn.Linear(64, 64, key=keyx[4]),
            jax.nn.tanh,
            eqx.nn.Linear(64, 1, key=keyx[5]),
        ]

    def __call__(self, x: Array) -> (any, Array):
        self.LOGGER.debug(f"Input received with shape {x.shape}")
        actor_output = x
        for actor_layer in self.actor_layers:
            actor_output = actor_layer(actor_output)

        critic_output = x
        for critic_layer in self.critic_layers:
            critic_output = critic_layer(critic_output)

        return actor_output, jnp.squeeze(critic_output, axis=-1)


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

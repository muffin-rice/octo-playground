# metal does not work. comment out if it does work
import jax

jax.config.update("jax_platform_name", "cpu")

import os
import sys

sys.path[0] = os.getcwd()

from math import prod
from typing import NamedTuple, Any, Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import equinox as eqx
import distrax
import optax
import JaxMARL.jaxmarl as jaxmarl
from JaxMARL.jaxmarl.environments.overcooked import overcooked_layouts

from src.ppo_marl.config import ppo_marl_config as config
from src.ppo_marl.ppo_types import (
    LossInformation,
    TrainState,
    Transition,
    Trajectory,
    TrajectoryState,
    create_trajectory_from_transitions,
    create_transition_list_from_transitions,
)
from src.ppo_marl.ppo_marl_utils import (
    batchify,
    reverse_batchify,
    dtype_as_str,
)
from src.logger import Logger

LOGGER = Logger("train.py")


class ActorCritic(eqx.Module):
    """Actor Critic class for PPO"""

    actor_layers: []
    critic_layers: []

    def __init__(self, key: Array, observation_dim: int, action_dim: int):
        LOGGER.info(
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
        LOGGER.debug(f"Input received with shape {x.shape}")
        actor_mean = x
        for actor_layer in self.actor_layers:
            actor_mean = actor_layer(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic_output = x
        for critic_layer in self.critic_layers:
            critic_output = critic_layer(critic_output)

        return pi, jnp.squeeze(critic_output, axis=-1)


def linear_schedule(count) -> float:
    frac = 1.0 - count / config["NUM_EPISODES"]
    assert 0 <= frac <= 1
    return (
        config["LR"]
        * frac
        / (config["NUM_STEPS"] * config["NUM_AGENTS"] * config["NUM_ENVS"])
    )


def get_env_step_function(
    env: jaxmarl.environments.MultiAgentEnv, model: eqx.Module
) -> Callable:
    """Wrap env_step in a returnable function as jax.lax.scan requires all objects
    in the carry state to be jax-able, and the env is not jax-able."""

    def env_step(trajectory_state: TrajectoryState, _) -> (Any, Transition):
        """Takes a step in the batch of environments given the model's pi and val for a single step(minibatch).
        To be used in jax.lax.scan to grab the entire trajectory.
        Returns shared first arg (trajectory_state) and the transition minibatch"""

        key, env_state, env_obs = trajectory_state
        agent_list: [str] = env_obs.keys()

        # get env_obs as array
        env_obs_array = batchify(env_obs, agent_list, config["NUM_ENVS"])

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
        action_unbatch = {k: v.flatten() for k, v in reverse_batchify(action, agent_list, config["NUM_ENVS"]).items()}

        LOGGER.debug(f"Action was chosen with shapes {dtype_as_str(action_unbatch)}")

        # apply action to env
        key, key_env_unsplit = jax.random.split(key, 2)
        key_env = jax.random.split(key_env_unsplit, config["NUM_ENVS"])

        obs_batch, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0)
        )(key_env, env_state, action_unbatch)
        LOGGER.debug(
            f"Observations have shape: {dtype_as_str(obs_batch)}\n"
            f"Reward has shape: {dtype_as_str(reward)}\n"
            f"Done has shape: {dtype_as_str(done)}\n"
            f"Info is: {info}"
        )

        info = jax.tree_map(lambda x: x.reshape((config["NUM_AGENTS"])), info)
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

        new_trajectory_state = TrajectoryState(
            key,
            env_state,
            obs_batch
        )

        return new_trajectory_state, transition

    return env_step


def calculate_gae(trajectory: Trajectory) -> Float[Array, "batch"]:
    """Given the list of transitions across the trajectory, calculate
    the GAE."""

    class GAECarryState(NamedTuple):
        next_gae: Float[
            Array, "batch"
        ]  # note: this is not the previous value as we lax.scan in reverse
        next_value: Float[Array, "batch"]
        index: int

    def calculate_advantage(
        carry_state: GAECarryState, curr_value: Float[Array, "batch"]
    ) -> (GAECarryState, Float[Array, "batch"]):
        """To be used in jax.lax.scan.
        First param (constantly updated and persisted) is the GAE up until
        that point and the previously used value.
        Second param is the GAE, which will be compiled into a list of all GAEs.
        Third param is index which is used for current-timestamp data.
        We call jax.lax.scan with reverse to make arithmetic easier."""
        next_gae, next_value, index = carry_state
        done, reward = (
            trajectory.t_done[:, -index - 1],
            trajectory.t_reward[:, -index - 1],
        )
        gae = (
            reward
            - curr_value
            + next_value * config["GAMMA"]
            + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * next_gae
        )
        return GAECarryState(next_gae=gae, next_value=curr_value, index=index + 1), gae

    # for a batch x d array of values/rewards, aggregate amongst the d dimension.
    # this means we .scan across the transpose of this array.
    # we should theoretically end up with batch, array of gaes
    inital_carry_state = GAECarryState(
        jnp.zeros_like(trajectory.t_done[:, 0]),
        jnp.zeros_like(trajectory.t_done[:, 0]),
        0,
    )

    _, gaes = jax.lax.scan(
        calculate_advantage,
        inital_carry_state,
        trajectory.t_value,
        reverse=True,
        # unroll = 16 # num steps. TODO: we already have num_steps as limit. necessary? although this is the num steps from the back.
    )
    LOGGER.debug(f"Computed GAE shape: {gaes.shape}")
    return gaes[-1, :]


def calculate_model_logprob(
    model: ActorCritic, trajectory: Trajectory
) -> Float[Array, "batch"]:
    """Given old actor/critic, generate logprob for given trajectory batch."""
    # logits (not logged) for each step, each batch
    model_pi, model_value = jax.vmap(model, in_axes=(0, 0))(trajectory.t_obs)
    # log probs for the trajectory is the sum across the steps
    LOGGER.debug(f"Categorical for each step has shape: {model_pi.batch_shape}")
    return jax.vmap(model_pi.log_prob, in_axes=(0, 0))(trajectory.t_action).sum(axis=1)


def get_actor_loss(
    gae: Float[Array, "batch"],
    new_log_prob: Float[Array, "batch"],
    old_log_prob: Float[Array, "batch"],
) -> Float[Array, ""]:
    ratio = new_log_prob / old_log_prob
    clipped_ratio = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"])

    # the ratio is actually the gain, so we negate it
    return -jnp.minimum(ratio * gae, clipped_ratio * gae).mean()


def get_loss(
    trajectory: Trajectory, model: ActorCritic, old_model: ActorCritic, agent_list: [str]
) -> (Float, LossInformation):
    """Compute loss of entire trajectory"""
    obs_batched = batchify(trajectory.t_obs, agent_list, config["NUM_ENVS"], additional_squeeze=True)
    LOGGER.info(f"Computing loss for observation {dtype_as_str(obs_batched)}")

    # get current model vals
    model_pi, model_value = jax.vmap(model)(obs_batched)

    # log probs for the trajectory is the sum across the steps
    LOGGER.debug(f"Categorical for each step has shape: {model_pi.batch_shape}")
    model_logprob = jax.vmap(model_pi.log_prob)(trajectory.t_action)

    # calculate critic loss
    clipped_model_value = trajectory.t_model_value + (
        model_value - trajectory.t_model_value
    ).clip(-config["CLIP_VAL"], config["CLIP_VAL"])
    # TODO: what is targets?
    unclipped_value_loss = jnp.square(trajectory.t_model_value - model_value)
    clipped_value_loss = jnp.square(trajectory.t_model_value - clipped_model_value)
    # TODO: why the /2?
    critic_loss = jnp.maximum(unclipped_value_loss, clipped_value_loss).mean() / 2

    # calculate actor loss
    actor_loss = get_actor_loss(
        calculate_gae(trajectory),
        model_logprob,
        calculate_model_logprob(old_model, trajectory),
    )

    # entropy
    entropy_loss = model_pi.entropy().mean()

    total_loss = (
        config["CRITIC_LOSS"] * critic_loss
        + config["ACTOR_LOSS"] * actor_loss
        + config["ENTROPY_LOSS"] * entropy_loss
    )

    return total_loss, LossInformation(actor_loss, critic_loss, entropy_loss)


def make_full_step(
    key: Array, env: jaxmarl.environments.MultiAgentEnv, train_state: TrainState
) -> (TrainState, LossInformation):
    """Given a train state, make a full step.
    Return a train_state with aux info such as loss values"""
    # extract values from train_state
    model, old_model, optim, opt_state = train_state

    # initialize values
    key, key_unsplit = jax.random.split(key)
    key_envs = jax.random.split(key_unsplit, config["NUM_ENVS"])
    # reset env
    obs, env_state = jax.vmap(env.reset)(key_envs)
    LOGGER.info(
        f"Finished reset. Observation: {dtype_as_str(obs)}\nEnv State: {dtype_as_str(env_state)}"
    )

    trajectory_state = TrajectoryState(key, env_state, obs)

    func_env_step = get_env_step_function(env, model)
    # run env_step to get list of transitions
    trajectory_state, transition_scan_list = jax.lax.scan(
        func_env_step, trajectory_state, None, length = config["NUM_STEPS"]
    )
    transition_list = create_transition_list_from_transitions(transition_scan_list, env.agents)

    LOGGER.info(f"Completed trajectory with transition list length {len(transition_list)}. "
                f"State timestamps across envs are {trajectory_state.env_state.time} and "
                f"shape of done array is {transition_list[-1].done['agent_1'].shape}")

    assert isinstance(transition_list, list)

    LOGGER.debug(f"Transition list is {dtype_as_str(transition_list)}")

    # construct trajectory from list of transitions
    sampled_trajectory = create_trajectory_from_transitions(transition_list, env.agents)

    LOGGER.debug(f"Trajectory dtype has shape {dtype_as_str(sampled_trajectory)}")

    # perform backwards steps
    (loss_info), grad = eqx.filter_value_and_grad(get_loss, has_aux=True)(
        sampled_trajectory, model, old_model, env.agents
    )
    # TODO: get opt_state typing
    updates, opt_state = optim.update(grad, opt_state, model)
    new_model = eqx.apply_updates(model, updates)

    new_train_state = TrainState(
        new_model,
        model,
        optim,
        opt_state,
    )

    return new_train_state, loss_info


if __name__ == "__main__":
    LOGGER.info("Hello!")
    key = jax.random.PRNGKey(config["STARTING_KEY"])

    # initialize env
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    # initialize model
    key, key_network = jax.random.split(key, 2)

    model = ActorCritic(
        key_network, prod(env.observation_space().shape), env.action_space().n
    )
    old_model = model

    # initialize optim
    optim = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(learning_rate=linear_schedule, eps=1e-5),
    )

    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    train_state = TrainState(model, old_model, optim, opt_state)

    for episode in range(config["NUM_EPISODES"]):
        train_state, loss_info = make_full_step(key, env, train_state)

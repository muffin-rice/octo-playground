# metal does not work. comment out if it does work
import jax

jax.config.update("jax_platform_name", "cpu")

import os
import sys

sys.path[0] = os.getcwd()

from datetime import datetime
from math import prod
from typing import NamedTuple

import equinox as eqx
import JaxMARL.jaxmarl as jaxmarl
from JaxMARL.jaxmarl.environments.overcooked import overcooked_layouts
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import optax
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from src.ppo_marl.config import model_config, loss_config, env_config
from src.ppo_marl.ppo_types import (
    ActorCritic,
    LossInformation,
    TrainState,
    Trajectory,
    TrajectoryState,
    create_trajectory_from_transitions,
    create_transition_list_from_transitions,
)
from src.ppo_marl.ppo_marl_utils import (
    batchify,
    dtype_as_str,
    get_env_step_function,
    load_train_state,
    save_train_state,
)
from src.logger import Logger

LOGGER = Logger("train.py")


def linear_schedule(count) -> float:
    frac = 1.0 - count / env_config["NUM_EPISODES"]
    return (
        loss_config["LR"] * frac / (env_config["NUM_AGENTS"] * env_config["NUM_ENVS"])
    )


def calculate_gae(
    trajectory: Trajectory, agent_list: [str]
) -> (Float[Array, "batch"], Float[Array, "batch num_steps"]):
    """Given the list of transitions across the trajectory, calculate
    the GAE."""
    t_done_batched = batchify(trajectory.t_done, agent_list, env_config["NUM_ENVS"])
    t_reward_batched = batchify(trajectory.t_reward, agent_list, env_config["NUM_ENVS"])

    LOGGER.info(
        f"Batched trajectory done and reward: {dtype_as_str(t_done_batched)} {dtype_as_str(t_reward_batched)}"
    )

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
            t_done_batched[:, -index - 1],
            t_reward_batched[:, -index - 1],
        )
        gae = (
            reward
            - curr_value
            + next_value * loss_config["GAMMA"]
            + loss_config["GAMMA"] * loss_config["GAE_LAMBDA"] * (1 - done) * next_gae
        )
        return GAECarryState(next_gae=gae, next_value=curr_value, index=index + 1), gae

    # for a batch x d array of values/rewards, aggregate amongst the d dimension.
    # this means we .scan across the transpose of this array.
    # we should theoretically end up with batch, array of gaes
    initial_carry_state = GAECarryState(
        jnp.zeros_like(t_done_batched[:, 0], dtype=jnp.float32),
        jnp.zeros_like(t_done_batched[:, 0], dtype=jnp.float32),
        0,
    )

    _, gaes = jax.lax.scan(
        calculate_advantage,
        initial_carry_state,
        trajectory.t_model_value.transpose((1, 0)),
        reverse=True,
        # unroll = 16 # num steps. TODO: we already have num_steps as limit. necessary? although this is the num steps from the back.
    )
    LOGGER.debug(f"Computed GAE: {dtype_as_str(gaes)}")

    # TODO: do we need to reverse this?
    last_gae = gaes[0, :]
    gaes_normalized = (last_gae - last_gae.mean()) / (last_gae.std() + 1e-8)
    return gaes_normalized, gaes.transpose((1, 0))


def calculate_model_logprob(
    model: ActorCritic,
    trajectory: Trajectory,
    agent_list: str,
) -> Float[Array, "batch"]:
    """Given old actor/critic, generate logprob for given trajectory batch."""
    t_obs_batched = batchify(
        trajectory.t_obs, agent_list, env_config["NUM_ENVS"], additional_squeeze=True
    )
    t_action_batched = batchify(
        trajectory.t_action, agent_list, env_config["NUM_ENVS"], additional_squeeze=True
    )

    # logits (not logged) for each step, each batch
    model_pi, model_value = jax.vmap(model)(t_obs_batched)
    # log probs for the trajectory is the sum across the steps
    model_logprobs = model_pi.log_prob(t_action_batched[:, 0])

    LOGGER.debug(
        f"Categoricals for trajectory: {model_pi.num_categories}\n"
        f"Logprobs for trajectory: {dtype_as_str(model_logprobs)}\n"
        f"Actions for trajectory: {dtype_as_str(t_action_batched)}"
    )

    return model_logprobs


def get_critic_loss(
    gae_per_timestamp: Float[Array, "num_envs d"],
    batched_values: Float[Array, "num_envs d"],
    curr_value: Float[Array, "num_envs d"]
) -> Float[Array, ""]:
    gae_extended = gae_per_timestamp.flatten()

    return (curr_value - gae_extended).mean() / 2

    # # calculate critic loss
    # clipped_model_value = t_model_values_batched + (
    #     model_value - t_model_values_batched
    # ).clip(-loss_config["CLIP_VAL"], loss_config["CLIP_VAL"])
    # unclipped_value_loss = jnp.square(gae_extended - model_value)
    # clipped_value_loss = jnp.square(gae_extended - clipped_model_value)
    # # TODO: why the /2?
    # # normalize by env, agents, num steps
    # critic_loss = jnp.maximum(unclipped_value_loss, clipped_value_loss).mean() / 2

def get_actor_loss(
    gae: Float[Array, "batch"],
    new_log_prob: Float[Array, "batch"],
    old_log_prob: Float[Array, "batch"],
) -> Float[Array, ""]:
    ratio = (new_log_prob - old_log_prob).reshape(
        env_config["NUM_ENVS"] * env_config["NUM_AGENTS"], env_config["NUM_STEPS"]
    )
    clipped_ratio = jnp.clip(
        ratio, 1.0 - loss_config["CLIP_EPS"], 1.0 + loss_config["CLIP_EPS"]
    )

    LOGGER.debug(
        f"Calculated ratio {dtype_as_str(ratio)} and clipped ratio {dtype_as_str(clipped_ratio)}"
    )

    # the ratio is actually the gain, so we negate it
    return -jnp.minimum(ratio * gae[:, None], clipped_ratio * gae[:, None]).mean()


def get_loss(
    model: ActorCritic,  # filter_value_and_grad requires backprop'd object to be first
    trajectory: Trajectory,
    old_model: ActorCritic,
    agent_list: [str],
) -> (Float, LossInformation):
    """Compute loss of entire trajectory"""
    # first batch out the trajectory
    t_obs_batched = batchify(
        trajectory.t_obs, agent_list, env_config["NUM_ENVS"], additional_squeeze=True
    )
    t_actions_batched = batchify(
        trajectory.t_action, agent_list, env_config["NUM_ENVS"], additional_squeeze=True
    )
    t_model_values_batched = batchify(
        trajectory.t_action, agent_list, env_config["NUM_ENVS"], additional_squeeze=True
    )
    t_model_values_nonsqueeze = batchify(
        trajectory.t_action,
        agent_list,
        env_config["NUM_ENVS"],
        additional_squeeze=False,
    )
    t_rewards_batched = batchify(
        trajectory.t_reward,
        agent_list,
        env_config["NUM_ENVS"],
        additional_squeeze=False,
    )

    LOGGER.debug(
        f"Batchified objects in trajectory: observation {dtype_as_str(t_obs_batched)}, actions {dtype_as_str(t_actions_batched)}, values {dtype_as_str(t_model_values_batched)}"
    )

    # get current model vals
    model_pi, model_value = jax.vmap(model)(t_obs_batched)

    # log probs for the trajectory is the sum across the steps
    model_logprob = model_pi.log_prob(t_actions_batched[:, 0])

    LOGGER.debug(
        f"Calculating critic loss from logprob {dtype_as_str(model_logprob)} and value {dtype_as_str(model_value)}..."
    )

    gae, gae_all = calculate_gae(trajectory, agent_list)

    critic_loss = get_critic_loss(gae_all, t_model_values_nonsqueeze, model_value)

    LOGGER.info(f"Calculated critic loss: {dtype_as_str(critic_loss)}")

    LOGGER.debug("Calculating actor loss...")

    # calculate actor loss
    actor_loss = get_actor_loss(
        gae,
        model_logprob,
        calculate_model_logprob(old_model, trajectory, agent_list),
    )

    LOGGER.info(f"Calculated actor loss: {dtype_as_str(actor_loss)}")

    # entropy
    entropy_loss = model_pi.entropy().mean()

    total_loss = (
        loss_config["CRITIC_LOSS"] * critic_loss
        + loss_config["ACTOR_LOSS"] * actor_loss
        - loss_config["ENTROPY_LOSS"] * entropy_loss
    )

    return total_loss, LossInformation(
        actor_loss,
        critic_loss,
        entropy_loss,
        t_rewards_batched.sum(),
        gae.sum(),
    )


@eqx.filter_jit
def make_full_step(
    key: Array, env: jaxmarl.environments.MultiAgentEnv, train_state: TrainState
) -> (TrainState, (Float[Array, ""], LossInformation)):
    """Given a train state, make a full step.
    Return a train_state with aux info such as loss values"""
    # extract values from train_state
    model, old_model, optim, opt_state = train_state

    # initialize values
    key, key_unsplit = jax.random.split(key)
    key_envs = jax.random.split(key_unsplit, env_config["NUM_ENVS"])
    # reset env
    obs, env_state = jax.vmap(env.reset)(key_envs)
    LOGGER.info(
        f"Finished reset. Observation: {dtype_as_str(obs)}\nEnv State: {dtype_as_str(env_state)}"
    )

    trajectory_state = TrajectoryState(key, env_state, obs)

    func_env_step = get_env_step_function(
        env,
        model,
        env_config["NUM_ENVS"],
        env_config["NUM_AGENTS"],
        env_config["ZERO_REWARD"],
    )

    LOGGER.info(f"Running {env_config['NUM_STEPS']} steps in env...")

    # run env_step to get list of transitions
    trajectory_state, transition_scan_list = jax.lax.scan(
        func_env_step, trajectory_state, None, length=env_config["NUM_STEPS"]
    )
    transition_list = create_transition_list_from_transitions(
        transition_scan_list, env.agents
    )

    LOGGER.info(
        f"Completed trajectory with transition list length {len(transition_list)}. "
        f"State timestamps across envs are {trajectory_state.env_state.time} and "
        f"shape of done array is {transition_list[-1].done['agent_1'].shape}"
    )

    LOGGER.debug(f"Transition list is {dtype_as_str(transition_list)}")

    # construct trajectory from list of transitions
    sampled_trajectory = create_trajectory_from_transitions(transition_list, env.agents)

    LOGGER.debug(
        f"Created trajectory from transition list: {dtype_as_str(sampled_trajectory)}"
    )

    LOGGER.debug("Calculating loss...")

    # perform backwards steps
    loss_info, grads = eqx.filter_value_and_grad(get_loss, has_aux=True)(
        model, sampled_trajectory, old_model, env.agents
    )
    LOGGER.info(f"Calculated loss: {loss_info[0]}")

    LOGGER.debug("Calculating optimizer update")
    # TODO: get opt_state typing
    updates, opt_state = optim.update(grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)

    new_train_state = TrainState(
        new_model,
        model,
        optim,
        opt_state,
    )

    return new_train_state, loss_info


def write_loss(writer: SummaryWriter, loss_info: LossInformation) -> None:
    writer.add_scalar("loss/actor_loss", np.asarray(loss_info.actor_loss))
    writer.add_scalar("loss/critic_loss", np.asarray(loss_info.critic_loss))
    writer.add_scalar("loss/entropy_loss", np.asarray(loss_info.entropy_loss))
    writer.add_scalar("reward/gae", np.asarray(loss_info.gae))
    writer.add_scalar("reward/total_reward", np.asarray(loss_info.total_reward))


if __name__ == "__main__":
    LOGGER.info("Hello!")
    key = jax.random.PRNGKey(env_config["STARTING_KEY"])

    # initialize env
    env = jaxmarl.make(env_config["ENV_NAME"], **env_config["ENV_KWARGS"])

    if env_config["CONTINUE"]:
        key, train_state, starting_epoch = load_train_state(env_config["PREVIOUS_SAVE"])
        LOGGER.info(
            f"Using old save from {env_config['PREVIOUS_SAVE']} with starting epoch {starting_epoch}"
        )
    else:
        # initialize model
        key, key_network = jax.random.split(key, 2)

        model = ActorCritic(
            key_network,
            prod(env.observation_space().shape),
            env.action_space().n,
            model_config,
        )
        old_model = model

        # initialize optim
        optim = optax.chain(
            optax.clip_by_global_norm(loss_config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
            # optax.adam(learning_rate=loss_config["LR"], eps=1e-5),
        )

        opt_state = optim.init(eqx.filter(model, eqx.is_array))

        train_state = TrainState(model, old_model, optim, opt_state)
        starting_epoch = 0
        LOGGER.info("Training from scratch.")

    writer = SummaryWriter(
        f"{env_config['TENSORBOARD_LOGDIR']}/{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}"
    )

    # jax.profiler.start_server(env_config["JAX_PROFILER_SERVER"]) # tensorboard server
    # LOGGER.info(f"Starting profiler server. Continue?")
    # input()

    for episode in range(env_config["NUM_EPISODES"]):
        key, key_step = jax.random.split(key, 2)
        curr_episode = starting_epoch + episode
        train_state, loss_info = make_full_step(key_step, env, train_state)
        write_loss(writer, loss_info[1])

        if loss_info[1].total_reward.item() > 0:
            LOGGER.info(
                f"Non-zero reward on episode {episode}: {loss_info[1].total_reward.item()}"
            )

        if curr_episode % env_config["CKPT_SAVE"] == 0:
            LOGGER.info(
                f"Checkpointed episode {starting_epoch + episode} with total loss {np.array(loss_info[0])}, total reward {np.array(loss_info[1].total_reward)}"
            )
            save_train_state(
                f'{env_config["SAVE_FILE"]}_{curr_episode}',
                key_step,
                train_state,
                curr_episode,
            )

    ending_epoch = starting_epoch + env_config["NUM_EPISODES"]
    LOGGER.info(
        f"Finished training epochs {starting_epoch} to {ending_epoch}; saving to file {env_config['SAVE_FILE']}"
    )

    save_train_state(
        env_config["SAVE_FILE"],
        key,
        train_state,
        ending_epoch,
    )

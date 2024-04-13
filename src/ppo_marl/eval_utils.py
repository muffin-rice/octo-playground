import jax

from src.ppo_marl.ppo_marl_utils import batchify
from src.ppo_marl.ppo_types import (
    ActorCritic,
    LossInformation,
    TrainState,
    Trajectory,
    TrajectoryState,
    Transition,
    create_trajectory_from_transitions,
    create_transition_list_from_transitions,
)


# specifically for overcooked
def percentage_of_transitions_done(transition_list: [Transition]) -> float:
    return (
        transition_list[-1].done["__all__"].sum()
        / transition_list[-1].done["__all__"].shape[0]
    ).item()


def forward_pass_model(model, transition_list, index, agent_list, num_envs):
    obs = transition_list[index].obs
    return jax.vmap(model)(batchify(obs, agent_list, num_envs))

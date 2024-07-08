import torch


from resources.environment.cost_sharing import CostSharing


# torch.nn.Module.load_state_dict()


# # Restore model parameters.
# for agent_id in range(self.num_agents):
#     policy_actor_state_dict = torch.load(
#         'results'
#         + "/actor_agent"
#         + str(agent_id)
#         + ".pt"
#     )
#     self.actor[agent_id].actor.load_state_dict(policy_actor_state_dict)
# if not self.algo_args["render"]["use_render"]:
#     policy_critic_state_dict = torch.load(
#         str(self.algo_args["train"]["model_dir"]) + "/critic_agent" + ".pt"
#     )
#     self.critic.critic.load_state_dict(policy_critic_state_dict)
#     if self.value_normalizer is not None:
#         value_normalizer_state_dict = torch.load(
#             str(self.algo_args["train"]["model_dir"])
#             + "/value_normalizer"
#             + ".pt"
#         )
#         self.value_normalizer.load_state_dict(value_normalizer_state_dict)


DETAILS = [
        {"name": "FIRM0", "costs_fixed": 1000, "cash_flow": 100000, "inventory": 50000},
        {"name": "FIRM1", "costs_fixed": 1000, "cash_flow": 100000, "inventory": 50000},
]

env = CostSharing(details_firms=DETAILS, render_mode='human', price_is_constant=True, cost_function='concave', portion_cost=0.75)
env.reset()

while True:
    actions = {0: [10555], 1: [10555]}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    if all(terminations.values()) or all(truncations.values()):
        break
    env.render()

from copy import copy, deepcopy

from resources.environment.cost_sharing import CostSharing, find_index_week


# HETEROGENEOUS-AGENT REINFORCEMENT LEARNING (HARL) ENVIRONMENT
class CostSharingEnvironment:
    """
    A class representing a cost sharing environment for heterogeneous agents.

    Args:
        args (dict): A dictionary containing the arguments for initializing the environment.

    Attributes:
        args (dict): The arguments for initializing the environment.
        env (CostSharing): The cost sharing environment.
        agents (list): The list of possible agents in the environment.
        n_agents (int): The number of agents in the environment.
        share_observation_space (list): The repeated state space for each agent.
        observation_space (list): The unwrapped observation spaces for each agent.
        action_space (list): The unwrapped action spaces for each agent.
        step_start (int): The starting step index.
        step_end (int): The ending step index.
        step_curr (int): The current step index.
        _seed (int): The seed value for random number generation.

    Methods:
        step(actions): Takes a step in the environment.
        reset(): Resets the environment.
        get_avail_actions(): Returns the available actions for all agents.
        get_avail_agent_actions(agent_id): Returns the available actions for a specific agent.
        seed(seed): Sets the seed value for random number generation.
        render(): Renders the environment.
        close(): Closes the environment.
        repeat(space): Repeats a space for each agent.
        wrap(list_info_agents): Wraps a list of agent information into a dictionary.
        unwrap(dict_info_agents): Unwraps a dictionary of agent information into a list.
    """

    def __init__(self, args):
        """
        Initializes the CostSharingEnv class.
        Args:
            args (dict): A dictionary containing the arguments for initializing the environment.
        """
        self.args = deepcopy(args)

        self.env = CostSharing(**self.args)
        self.env.reset()

        self.agents = self.env.possible_agents
        self.n_agents = self.env.n_agents
        self.share_observation_space = self.repeat(self.env.state_space)
        self.observation_space = self.unwrap(self.env.observation_spaces)
        self.action_space = self.unwrap(self.env.action_spaces)
        self.step_start = 0 if "time_start" not in self.args else find_index_week(*self.args["time_start"])
        self.step_end = 3944 if "time_end" not in self.args else find_index_week(*self.args["time_end"])
        self.step_curr = self.step_start
        self._seed = self.env.get_seed()

    def step(self, actions):
        """
        Takes a step in the environment.
        Args:
            actions (list): The list of actions for each agent.
        Returns:
            tuple: A tuple containing the local observations, global state, rewards, dones, info, and available actions.
        """
        obs_local, rew, term, trunc, info = self.env.step(self.wrap(actions))
        self.agents = self.env.agents
        self.step_curr += 1

        if self.step_curr == self.step_end:
            trunc = {agent: True for agent in self.agents}
            for agent in self.agents:
                info[agent]["bad_transition"] = True

        dones = {agent: term[agent] or trunc[agent] for agent in self.agents}
        obs_local = self.unwrap(obs_local)
        state_global = self.repeat(self.env.get_state())
        total_reward = sum([rew[agent] for agent in self.agents])
        rewards = [[total_reward]] * self.n_agents

        return (
            obs_local,
            state_global,
            rewards,
            self.unwrap(dones),
            self.unwrap(info),
            self.get_avail_actions(),
        )

    def reset(self):
        """
        Resets the environment.
        Returns:
            tuple: A tuple containing the observations, state, and available actions.
        """
        self._seed += 1
        self.step_curr = self.step_start
        obs = self.unwrap(self.env.reset(seed=self._seed))
        state = self.repeat(self.env.get_state())
        return obs, state, self.get_avail_actions()
    
    def get_avail_actions(self):
        """
        Returns the available actions for all agents. Because the action space is continuous, agents have unlimited possibilities.
        Returns:
            None
        """
        return None

    def get_avail_agent_actions(self, agent_id):
        """
        Returns the available actions for a specific agent.
        Args:
            agent_id (int): The ID of the agent.
        Returns:
            list: The available actions for the agent.
        """
        return [1] * self.action_space[agent_id].n

    def seed(self, seed):
        """
        Sets the seed value for random number generation.
        Args:
            seed (int): The seed value.
        """
        self._seed = seed
        self.env.reset(seed=seed)

    def render(self):
        """
        Renders the environment.
        """
        self.env.render()

    def close(self):
        """
        Closes the environment.
        """
        self.env.close()
    
    def repeat(self, space):
        """
        Repeats a space for each agent.
        Args:
            space: The space to repeat.
        Returns:
            list: The repeated space for each agent.
        """
        return [space for _ in range(self.n_agents)]
    
    def wrap(self, list_info_agents):
        """
        Wraps a list of agent information into a dictionary.
        Args:
            list_info_agents (list): The list of agent information.
        Returns:
            dict: The wrapped agent information in a dictionary.
        """
        dict_info_agents = {}
        for i, agent in enumerate(self.agents):
            dict_info_agents[agent] = list_info_agents[i]
        return dict_info_agents

    def unwrap(self, dict_info_agents):
        """
        Unwraps a dictionary of agent information into a list.
        Args:
            dict_info_agents (dict): The dictionary of agent information.
        Returns:
            list: The unwrapped agent information in a list.
        """
        list_info_agents = []
        for agent in self.agents:
            list_info_agents.append(dict_info_agents[agent])
        return list_info_agents

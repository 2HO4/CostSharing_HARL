"""

Cost-sharing environment for the cost-sharing game with (heterogeneous) firms.

"""

# IMPORTS
# Default libraries
from copy import copy, deepcopy
import functools
from typing import Iterable, Tuple, Union, Callable
from math import ceil, floor
import warnings

# Data manipulation libraries
import pandas as pd
import numpy as np

# Reinforcement learning libraries
import gymnasium as gym
from gymnasium.spaces import Box
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

# Cost-sharing components
from resources.agent import Firm
from resources.economics.price_calculation import calculate_price
from resources.economics.cost_functions import build_cost_function, scale_cost, normalize_cost
from resources.economics.cost_sharing_algorithms import use_cost_sharing


# PETTINGZOO ENVIRONMENT
class CostSharing(ParallelEnv):
    """
    CostSharing is an environment class that represents a cost-sharing game. It is a PettingZoo parallel environment.

    Attributes:
        metadata (dict): Metadata for the environment, including render modes and name.
        details_firms (list): Details of the firms participating in the game.
        date_start (int): Start date of the game.
        date_end (int): End date of the game.
        emissions_max (float): Maximum emissions allowed in the game.
        seed (int): Seed for random number generation.
        price_is_constant (bool or float): Flag indicating whether the price of gasoline is constant.
        price_ceiling (float): The maximum price of gasoline in the game.
        price_floor (float): The minimum price of gasoline in the game.
        discount_demand (float): The discount rate of the remaining unsatisfied demand.
        portion_cost (float): The value that manipulates the scaling of the total cost of producing gasoline.
        portion_reward (float): The portion of a firm's earnings that becomes the reward.
        portion_punishment (float): The portion of total earnings that is taken as a punishment when the demand is not satisfied.
        reward_final (float): The reward given to a firm when it ends the game.
        quota_production (int): The number of barrels of gasoline that a single firm needs to produce in a week.
        cap_production (int): The maximum number of barrels of gasoline that a single firm can produce in a week.
        hide_others_moves (bool): Flag indicating whether past production of firms should be hidden from each firm.
        cost_function (str or function): Cost function used in the game.
        algorithm (function): Cost-sharing algorithm used in the game.
        render_mode (str): Render mode for visualization.

    Methods:
        observation_space(agent): Get the observation space for the specified agent.
        action_space(agent): Get the action space for the specified agent.
        update_observations(): Update the observations of the environment.
        step(actions): Execute a single step in the environment based on the given actions.
        render(): Render the current state of the environment.
        get_seed(): Get the seed used in the game.
        get_state(): Get the current state of the game.
        reset(seed=None, options=None): Reset the game to its initial state.

    """

    metadata = {
        "render_modes": ["human"],
        "name": "cost_sharing_v4",
    }

    def __init__(
            self, 
            details_firms: Union[str, Iterable[dict]]=None, 
            date_start: Tuple[int, int]=None, 
            date_end: Tuple[int, int]=None, 
            emissions_max: float=None, 
            seed: int=None, 
            price_is_constant: Union[bool, float]=None, 
            price_ceiling: float=None,
            price_floor: float=None,
            discount_demand: float=None, 
            portion_cost: float=None, 
            portion_reward: float=None,
            portion_punishment: float=None,
            reward_final: float=None,
            quota_production: int=None,
            cap_production: int=None, 
            hide_others_moves: bool=None,
            cost_function: Union[str, Callable]=None, 
            algorithm: Union[str, Callable]=None, 
            render_mode: str=None
        ):
        """
        Initialize the CostSharing environment.
        Args:
            details_firms (list): Details of the firms participating in the game.
            date_start (tuple): Start time of the game.
            date_end (tuple): End time of the game.
            emissions_max (float): Maximum emissions allowed in the game.
            seed (int): Seed for random number generation.
            price_is_constant (bool or float): Flag indicating whether the price of gasoline is constant. If it is a float, then it is the 'constant' price of gasoline.
            price_ceiling (float): The maximum price of gasoline in the game.
            price_floor (float): The minimum price of gasoline in the game.
            discount_demand (float): The discount rate of the remaining unsatisfied demand. For example, if it is 0.25, then 1/4 of the remaining demand will be carried to the next period.
            portion_cost (float): The value that manipulates the scaling of the total cost of producing gasoline.
            portion_reward (float): The portion of a firm's earnings that becomes the reward.
            portion_punishment (float): The portion of total earnings that is taken as a punishment when the demand is not satisfied.
            reward_final (float): The reward given to a firm when it ends the game.
            quota_production (int): The number of barrels of gasoline that a single firm needs to produce in a week.
            cap_production (int): The maximum number of barrels of gasoline that a single firm can produce in a week.
            hide_others_moves (bool): Flag indicating whether past production of firms should be hidden from each firm.
            cost_function (str or function): Cost function of the number of barrels produced which is used in the game.
            algorithm (str or function): Cost-sharing algorithm used in the game.
            render_mode (str): Render mode for visualization.
        """
        if details_firms is None:
            details_firms = DETAILS_OF_FIRMS_COMPETITION_HETEROGENEOUS
        elif type(details_firms) == dict:
            details_firms = create_details_firms(**details_firms)
        else:
            try:
                details_firms = align_names(list(details_firms))
            except:
                raise ValueError("The details of the firms need to be a list of dictionaries.")
        
        n_agents = len(details_firms)
        cost_function = 'convex' if cost_function is None else cost_function
        algorithm = 'serial' if algorithm is None else algorithm
        
        # Environment's configuration
        self._seed = 0 if seed is None else seed%10
        self._price_is_constant = False if price_is_constant is None else float(price_is_constant)
        
        if self._price_is_constant < 0:
            raise ValueError("The constant price of gasoline cannot be negative.")
        if self._price_is_constant and (price_ceiling is not None or price_floor is not None):
            warnings.warn("The constant price of gasoline is set. The price ceiling and floor will be ignored.")
        else:
            self._price_ceiling = np.inf if price_ceiling is None else price_ceiling
            self._price_floor = 0 if price_floor is None else price_floor
            if self._price_ceiling < 0:
                raise ValueError("The price ceiling cannot be negative.")
            if self._price_floor < 0:
                raise ValueError("The price floor cannot be negative.")
            if self._price_ceiling < self._price_floor:
                raise ValueError("The price ceiling cannot be lower than the price floor.")
            
        self._discount_demand = 0.1 if discount_demand is None else discount_demand
        self._portion_cost = 1 if portion_cost is None else portion_cost
        self._portion_reward = 0.00000001 if portion_reward is None else portion_reward
        self._portion_punishment = 0.00000001 if portion_punishment is None else portion_punishment
        self._reward_final = 0 if reward_final is None else reward_final
        self._quota_production = 0 if quota_production is None else quota_production
        self._cap_production = np.inf if cap_production is None else cap_production
        self._hide_others_moves = True if hide_others_moves is None else hide_others_moves

        # Default RL attributes
        self.render_mode = render_mode
        self.possible_agents = list(range(n_agents))
        self.n_agents = n_agents
        self.agent_name_mapping = dict(zip(self.possible_agents, self.possible_agents))
        self.state_space = Box(low=0, high=np.array([5000, 16, np.inf, np.inf] + [1]*n_agents), dtype=np.float32)
        self.observation_spaces = {a: self.observation_space(a) for a in self.possible_agents}
        self.action_spaces = {a: self.action_space(a) for a in self.possible_agents}
        
        # Firms
        self._details_firms = details_firms
        self._firms = {i: Firm(**self._details_firms[i]) for i in self.possible_agents}

        # Number of barrels of gasoline in the global market
        self._n_barrels_total = sum(firm.get_inventory() for firm in self._firms.values())
        
        # Time data (in 3944 weeks) starting from the 28th week of 2024
        self._date_start = (2024, 28) if date_start is None else date_start
        self._date_end = (2100, 20) if date_end is None else date_end  # it should represent the first week of 2100, but is set to the 20th week due to the rounding of the number of weeks in a year to 52
        self._time_start = 0 if date_start is None else find_index_week(*date_start)
        self._time_end = 3944 if date_end is None else find_index_week(*date_end)
        self._time_remaining = self._time_end - self._time_start
        
        # Emissions data (in billions of tonnes of CO2)
        self._emissions_max = 999999999 if emissions_max is None else emissions_max
        self._emissions = pd.read_csv('./forecasts/forecast_cumulativeghgusemissions.csv').iloc[self._time_start]['Point.Forecast']

        # In order to reach the maximum emissions within the time frame:
        self._emissions_per_week = (self._emissions_max - self._emissions)/self._time_remaining  # average emissions (in billions of tonnes of CO2) per week
        self._n_barrels_per_week = int(self._emissions_per_week/BILLION_EMISSIONS_PER_THOUSAND_BARRELS)  # average number (in thousands) of barrels of gasoline produced per week
        
        # Cost-sharing components
        if type(cost_function) == str:
            self._cost_function = build_cost_function(characteristic=cost_function, normalize=self._n_barrels_per_week, scale=1)
            self._characteristic_cost_function = cost_function
        elif callable(cost_function):
            self._cost_function = normalize_cost(cost_function=cost_function, factor=self._n_barrels_per_week)
            self._characteristic_cost_function = cost_function.__name__
        else:
            raise ValueError("Cost function needs to be a function or a string representing the property of the function.")
        
        if type(algorithm) == str:
            self._cost_sharing_algorithm = use_cost_sharing(algorithm)
            self._characteristic_algorithm = algorithm
        elif callable(algorithm):
            self._cost_sharing_algorithm = algorithm
            self._characteristic_algorithm = algorithm.__name__
        else:
            raise ValueError("Cost sharing algorithm needs to be a function or a string representing the property of the algorithm.")

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """
        Returns the observation space for the given agent. It is a continuous space with the following dimensions:
        - Emissions: the total emissions produced in the game.
        - Event: the event that affects the demand for gasoline.
        - Demand forecast: the forecasted demand for gasoline.
        - Supply: the total supply of gasoline in the market.
        - Status of all firms in the economy: 1 if the firm is alive, 0 if the firm is dead.
        - Inventory: the inventory of the firm.
        - Cash flow: the cash flow of the firm.
        """
        # The status of all firms in the economy is either whether they are alive or their last actions/produtions
        status_firms = 1 if self._hide_others_moves else self._cap_production

        return Box(
            low=0, 
            high=np.array(
                [5000, 16, np.inf, np.inf] +  # environment's data: emissions, event, demand forecast, supply
                [status_firms]*self.n_agents +  # status of all firms in the economy
                [np.inf, np.inf]  # firm's specific data: inventory, cash flow
            ),
            dtype=np.float32
        )
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: int):
        """
        Returns the action space for the specified agent. It is a continuous 1-dimensional space which represents the quantity of gasoline to produce.
        """
        return Box(low=self._quota_production, high=self._cap_production, shape=(1, ))
    
    def update_observations(self):
        self._state = [
            self._emissions_curr, 
            self._event_curr, 
            self._demand_forecast_curr, 
            self._supply_curr
        ] + self._statuses_firms_prev
        
        observations = {
            agent: self._state + [self._firms_curr[agent].get_inventory(), self._firms_curr[agent].get_cash_flow()] 
            for agent in self.agents
        }

        return observations

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.
        Parameters:
        - seed (int): The seed for random number generation. If provided, it will be used to generate a new seed between 0 and 9.
        - options (dict): Additional options for resetting the environment.
        Returns:
        - observations (object): The initial observations of the environment for each agent after resetting.
        """
        self.agents = copy(self.possible_agents)
        self._agents_remaining = copy(self.possible_agents)

        self._firms_curr = deepcopy(self._firms)

        self._supply_curr = self._n_barrels_total

        self._time_curr = self._time_start
        
        self._emissions_curr = self._emissions
        
        self._seed = seed%10 if seed is not None else self._seed
        self._event, self._demand_forecast, self._demand_real = pd.read_csv(f'./forecasts/forecast_demand_withshock{self._seed}.csv', index_col=0).iloc[self._time_start][:3]
        self._event_curr = self._event
        self._demand_forecast_curr = self._demand_forecast
        self._demand_real_curr = self._demand_real

        # Previous results
        self._production_total_prev = self._n_barrels_total
        self._price_prev = self._price_floor
        self._costs_emissions_shared_prev = 0
        self._statuses_firms_prev = [1 if self._hide_others_moves else 0]*self.n_agents

        return self.update_observations()

    def step(self, actions):
        """
        Executes a single step in the environment.
        Args:
            actions (dict): A dictionary containing the actions for each agent.
        Returns:
            tuple: A tuple containing the following elements:
                - observations (dict): A dictionary containing the observations for each agent.
                - rewards (dict): A dictionary containing the rewards for each agent.
                - terminations (dict): A dictionary indicating whether each agent has terminated.
                - truncations (dict): A dictionary indicating whether each agent has reached a truncation condition.
                - infos (dict): A dictionary containing additional information for each agent.
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            infos = {a: {} for a in self.agents}
            # self.agents = []
            return {}, {}, {}, {}, infos
        
        # Rewards for each agent
        rewards = {agent: 0 for agent in self.agents}
        
        # Termination conditions during the game
        terminations = {agent: True for agent in self.agents}

        # Truncation conditions when the game ends after an externally defined condition, i.e. reaches the last time frame (overwrites termination conditions)
        truncations = {agent: False for agent in self.agents}
        
        # Get production quantities
        quantities_to_produce = {}
        quantities_negative = {}
        for agent in self._agents_remaining:
            if actions[agent][0] < 0:
                quantities_negative[agent] = floor(actions[agent][0])
                terminations[agent] = True
                rewards[agent] = -self._reward_final
            elif actions[agent][0] is None:
                quantities_to_produce[agent] = 0
            else:
                quantities_to_produce[agent] = round(actions[agent][0])
        
        production_total = sum(quantities_to_produce.values())
        self._supply_curr += production_total
        n_barrels_total = self._supply_curr

        # Calculate price of gasoline
        if self._price_is_constant:
            price = self._price_is_constant
        else:
            price = calculate_price(self._supply_curr, self._demand_forecast_curr, self._demand_real_curr)
            if price > self._price_ceiling:
                price = self._price_ceiling
            elif price < self._price_floor:
                price = self._price_floor

        # Calculate remaining supply and demand of gasoline
        if self._supply_curr >= self._demand_real_curr:
            n_barrels_sold = self._demand_real_curr
            self._supply_curr -= self._demand_real_curr
            self._demand_real_curr = 0
        else:
            if self._n_barrels_per_week > self._demand_real_curr:
                punishment = (self._n_barrels_per_week - self._demand_real_curr)*price/len(self._agents_remaining)*self._portion_punishment
                for agent in self._agents_remaining:
                    rewards[agent] -= punishment
            n_barrels_sold = self._supply_curr
            self._demand_real_curr = int((self._demand_real_curr - self._supply_curr)*self._discount_demand)
            self._supply_curr = 0

        # Calculate total emissions
        emissions_produced = BILLION_EMISSIONS_PER_THOUSAND_BARRELS*production_total
        self._emissions_curr += emissions_produced

        # Calculate shared costs of emissions
        cost_function_curr = scale_cost(self._cost_function, factor=price*self._n_barrels_per_week*self._portion_cost)
        costs_emissions_shared = self._cost_sharing_algorithm(cost_function_curr, quantities_to_produce)

        # Apply actions
        agents_remaining = []
        statuses_firms = [0]*self.n_agents
        
        for agent, quantity_to_produce in quantities_to_produce.items():
            firm = self._firms_curr[agent]
            
            # Update firm's production and sales
            firm.produce(quantity_to_produce)

            quantity_to_sell = min(firm.get_inventory(), ceil(firm.get_inventory()/n_barrels_total*n_barrels_sold) if n_barrels_sold > 0 else 0)
            firm.sell(quantity_to_sell)
            
            # Calculate net earnings
            net_earnings = price*quantity_to_sell - costs_emissions_shared[agent]

            # Update firm's cash flow
            firm.earn(net_earnings)
            rewards[agent] += net_earnings*self._portion_reward

            # Check if the firm overproduces and goes bankrupt
            if firm.get_cash_flow() < 0:
                rewards[agent] = -self._reward_final
            else:
                terminations[agent] = False
                agents_remaining.append(agent)
            
            statuses_firms[agent] = quantity_to_produce
        
        for agent, quantity_negative in quantities_negative.items():
            firm = self._firms_curr[agent]
            
            # Update firm's production and sales
            firm.produce(quantity_negative)
            firm.sell("all")

            # Calculate net earnings
            net_earnings = price*quantity_negative

            # Set firm's cash flow to zero
            firm.earn(-firm.get_cash_flow())
            rewards[agent] += net_earnings*self._portion_reward
            
            statuses_firms[agent] = quantity_negative
        
        # Check if the game ends:
        self._time_curr += 1
        ## Due to reaching the last time frame
        if self._time_curr == self._time_end:
            truncations = {a: True for a in self.agents}
            observations = {a: [0]*(6 + self.n_agents) for a in self.agents}
            if self._emissions_curr <= self._emissions_max:  # the total emissions are within the limit so all remaining agents receive a reward
                for agent in agents_remaining:
                    rewards[agent] += self._reward_final
            else:
                for agent in agents_remaining:
                    rewards[agent] -= self._reward_final
            infos = {a: {} for a in self.agents}
            
            return observations, rewards, terminations, truncations, infos
        
        ## Due to no more agents
        if not agents_remaining:
            observations = {a: [0]*(6 + self.n_agents) for a in self.agents}
            infos = {a: {} for a in self.agents}
            
            return observations, rewards, terminations, truncations, infos
        
        ## Due to excessive emissions
        elif self._emissions_curr > self._emissions_max:
            terminations = {a: True for a in self.agents}
            for agent in agents_remaining:
                rewards[agent] = -self._reward_final
            observations = {a: [0]*(6 + self.n_agents) for a in self.agents}
            infos = {a: {} for a in self.agents}
            
            return observations, rewards, terminations, truncations, infos
        
       # New observations
        self._event_curr, self._demand_forecast_curr, demand_real_curr = pd.read_csv(f'./forecasts/forecast_demand_withshock{self._seed}.csv', index_col=0).iloc[self._time_curr][:3]
        self._demand_real_curr += demand_real_curr
        observations = self.update_observations()

        # Dummy infos
        infos = {a: {} for a in self.agents}

        # Update agents
        self._agents_remaining = agents_remaining

         # Record previous results
        self._production_total_prev = production_total
        self._price_prev = price
        self._costs_emissions_shared_prev = {agent: round(costs_emissions_shared[agent], 2) for agent in self._agents_remaining}
        
        if self._hide_others_moves:
            self._statuses_firms_prev = [0]*self.n_agents
            for agent in self._agents_remaining:
                self._statuses_firms_prev[agent] = 1
        else:
            self._statuses_firms_prev = statuses_firms

        return observations, rewards, terminations, truncations, infos


    def render(self):
        """
        Renders the current state of the environment.
        """
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return
        
        if self._time_curr == 1:
            print(
                f"\n{'='*122}\n" + \
                f"|{' '*51}COST-SHARING GAME{' '*52}|\n" + \
                f"{'='*122}\n" + \
                f"| The game is a cost-sharing game where firms produce gasoline and share the costs of emissions.{' '*25}|\n" + \
                f"| The goal is to maximize profits while minimizing emissions.{' '*60}|\n" + \
                f"| The game ends when the total emissions exceed the limit, all firms go bankrupt, or we reach the last week.{' '*13}|\n" + \
                f"{'-'*122}"
            )

            print(
                f"\n\nGAME SETTINGS:\n" + \
                f"Time frame: week {self._date_start[1]}/{self._date_start[0]} ({self._time_start}) to week {self._date_end[1]}/{self._date_end[0]} ({self._time_end}) -> {self._time_end - self._time_start} weeks\n" + \
                f"Maximum emissions: {self._emissions_max} billions of tonnes of CO2\n" + \
                f"Ideal emissions per week: {self._emissions_per_week} billions of tonnes of CO2\n" + \
                f"Ideal number of barrels produced per week: {self._n_barrels_per_week} thousands of barrels\n" + \
                f"Seed: {self._seed}\n" + \
                f"Price is constant: {self._price_is_constant}\n" + \
                f"Portion of cost: {self._portion_cost}\n" + \
                f"Quota on production: {self._quota_production}\n" + \
                f"Cap on production: {self._cap_production}\n" + \
                f"Hide others' moves: {self._hide_others_moves}\n" + \
                f"Discount factor of remaining demand: {self._discount_demand}\n" + \
                f"Cost function: {self._characteristic_cost_function}\n" + \
                f"Cost-sharing algorithm: {self._characteristic_algorithm}\n"
            )

            print(
                "\n\nINITIAL ENVIRONMENT:\n" + \
                f"Alive agents: {self.possible_agents}\n- " + \
                "- ".join(f"{str(self._firms[agent])}\n" for agent in self.possible_agents) + \
                "\n" + \
                f"State:\n" + \
                f"- EMISSIONS: {self._emissions}\n" + \
                f"- EVENT: {self._event}\n" + \
                f"- DEMAND_FORECAST: {self._demand_forecast}\n" + \
                f"- DEMAND_REAL: {self._demand_real}\n" + \
                f"- SUPPLY: {self._n_barrels_total}\n"
            )
        
        if len(self._agents_remaining) > 0:
            agents_dead = list(set(self.possible_agents) - set(self._agents_remaining))
            
            print(
                f"\n\nWEEK {self._time_curr}:\n" + \
                f"Results:\n" + \
                f"- PRODUCTION: {self._production_total_prev}\n" + \
                f"- PRICE: {self._price_prev}\n" + \
                f"- SHARED COSTS OF EMISSIONS: {self._costs_emissions_shared_prev}\n" + \
                f"- FIRMS' STATUSES: {dict(enumerate(self._statuses_firms_prev))}\n" + \
                f"\n" + \
                f"Alive agents: {self._agents_remaining}\n- " + \
                "- ".join(f"{str(self._firms_curr[agent])}\n" for agent in self._agents_remaining) + \
                "\n" + \
                f"Dead agents: {agents_dead}\n- " + \
                "- ".join(f"{str(self._firms_curr[agent])}\n" for agent in agents_dead) + \
                "\n" + \
                f"State:\n" + \
                f"- EMISSIONS: {self._emissions_curr}\n" + \
                f"- EVENT: {self._event_curr}\n" + \
                f"- DEMAND_FORECAST: {self._demand_forecast_curr}\n" + \
                f"- DEMAND_REAL: {self._demand_real_curr}\n" + \
                f"- SUPPLY: {self._supply_curr}\n"
            )
        else:
            print("\n\nGAME OVER\n")
    
    def get_seed(self):
        return self._seed
    
    def get_state(self):
        return self._state
    
    def close(self):
        pass
    

if __name__ == "__main__":
    from pettingzoo.test import parallel_api_test


    details = [
        {'name': 'small0', 'costs_fixed': 5000, 'cash_flow': 50000}, 
        {'name': 'small1', 'costs_fixed': 5000, 'cash_flow': 200000}, 
        {'name': 'med0', 'costs_fixed': 6000, 'cash_flow': 300000}, 
        {'name': 'med1', 'costs_fixed': 7800, 'cash_flow': 300000},
        {'name': 'big0', 'costs_fixed': 10000, 'cash_flow': 1000000, 'inventory': 10000}, 
        {'name': 'big1', 'costs_fixed': 10000, 'cash_flow': 1000000, 'inventory': 20000}
    ]
    
    env = CostSharing(details, price_is_constant=False, render_mode="human")
    parallel_api_test(env, num_cycles=1_000_000)


# UTILITIES
# Constants
BILLION_EMISSIONS_PER_THOUSAND_BARRELS = 0.373254*(10**3)/(10**9)  # billions of tonnes of CO2 emissions per thousand of gasoline barrels

DETAILS_OF_FIRMS_MONOPOLY_HETEROGENEOUS = [{"name": "FIRM", "costs_fixed": 1000, "cash_flow": 40000, "inventory": 20000}]
DETAILS_OF_FIRMS_MONOPOLY_HOMOGENEOUS = [{"name": "FIRM", "costs_fixed": 1000, "cash_flow": 40000, "inventory": 20000}]

DETAILS_OF_FIRMS_DUOPOLY_HETEROGENEOUS = [
    {"name": "SMALL", "costs_fixed":  500, "cash_flow": 20000, "inventory": 10000},
    {"name": "BIG  ", "costs_fixed": 1500, "cash_flow": 80000, "inventory": 40000},
]

DETAILS_OF_FIRMS_DUOPOLY_HOMOGENEOUS = [
    {"name": "FIRM0", "costs_fixed": 1000, "cash_flow": 40000, "inventory": 20000},
    {"name": "FIRM1", "costs_fixed": 1000, "cash_flow": 40000, "inventory": 20000},
]

DETAILS_OF_FIRMS_TRIPOLY_HETEROGENEOUS = [
    {"name": "SMALL ", "costs_fixed":  500, "cash_flow": 20000, "inventory": 10000},
    {"name": "MEDIUM", "costs_fixed": 1000, "cash_flow": 40000, "inventory": 20000},
    {"name": "BIG   ", "costs_fixed": 1500, "cash_flow": 80000, "inventory": 40000},
]

DETAILS_OF_FIRMS_TRIPOLY_HOMOGENEOUS = [
    {"name": "FIRM0", "costs_fixed": 1000, "cash_flow": 40000, "inventory": 20000},
    {"name": "FIRM1", "costs_fixed": 1000, "cash_flow": 40000, "inventory": 20000},
    {"name": "FIRM2", "costs_fixed": 1000, "cash_flow": 40000, "inventory": 20000},
]

DETAILS_OF_FIRMS_OLIGOPOLY_HETEROGENEOUS = [
    {"name": "TINY", "costs_fixed":  250, "cash_flow": 10000, "inventory":  5000},
    {"name": "SMALL", "costs_fixed":  500, "cash_flow": 20000, "inventory": 10000},
    {"name": "MEDIUM", "costs_fixed": 1000, "cash_flow": 40000, "inventory": 20000},
    {"name": "BIG", "costs_fixed": 1500, "cash_flow": 80000, "inventory": 40000},
]

DETAILS_OF_FIRMS_OLIGOPOLY_HOMOGENEOUS = [
    {"name": "FIRM0", "costs_fixed": 1000, "cash_flow": 40000, "inventory": 20000},
    {"name": "FIRM1", "costs_fixed": 1000, "cash_flow": 40000, "inventory": 20000},
    {"name": "FIRM2", "costs_fixed": 1000, "cash_flow": 40000, "inventory": 20000},
    {"name": "FIRM3", "costs_fixed": 1000, "cash_flow": 40000, "inventory": 20000},
]

DETAILS_OF_FIRMS_COMPETITION_HETEROGENEOUS = [
    {"name": "SMALL                         ", "costs_fixed":  500, "cash_flow": 10000},
    {"name": "SMALL_WITH_CASHFLOW           ", "costs_fixed":  500, "cash_flow": 20000}, 
    {"name": "SMALL_WITH_INVENTORY          ", "costs_fixed":  500, "cash_flow": 10000, "inventory": 10000},
    {"name": "SMALL_WITH_CASHFLOW_INVENTORY ", "costs_fixed":  500, "cash_flow": 20000, "inventory": 10000},
    
    {"name": "MEDIUM                        ", "costs_fixed": 1000, "cash_flow": 20000},
    {"name": "MEDIUM_WITH_CASHFLOW          ", "costs_fixed": 1000, "cash_flow": 40000}, 
    {"name": "MEDIUM_WITH_INVENTORY         ", "costs_fixed": 1000, "cash_flow": 20000, "inventory": 20000},
    {"name": "MEDIUM_WITH_CASHFLOW_INVENTORY", "costs_fixed": 1000, "cash_flow": 40000, "inventory": 20000},
    
    {"name": "BIG                           ", "costs_fixed": 1500, "cash_flow": 40000},
    {"name": "BIG_WITH_CASHFLOW             ", "costs_fixed": 1500, "cash_flow": 80000}, 
    {"name": "BIG_WITH_INVENTORY            ", "costs_fixed": 1500, "cash_flow": 40000, "inventory": 40000},
    {"name": "BIG_WITH_CASHFLOW_INVENTORY   ", "costs_fixed": 1500, "cash_flow": 80000, "inventory": 40000}
]

DETAILS_OF_FIRMS_COMPETITION_HOMOGENEOUS = [
    {"name": "FIRM00", "costs_fixed": 1000, "cash_flow": 40000, "inventory": 20000},
    {"name": "FIRM01", "costs_fixed": 1000, "cash_flow": 40000, "inventory": 20000},
    {"name": "FIRM02", "costs_fixed": 1000, "cash_flow": 40000, "inventory": 20000},
    {"name": "FIRM03", "costs_fixed": 1000, "cash_flow": 40000, "inventory": 20000},
    {"name": "FIRM04", "costs_fixed": 1000, "cash_flow": 40000, "inventory": 20000},
    {"name": "FIRM05", "costs_fixed": 1000, "cash_flow": 40000, "inventory": 20000},
    {"name": "FIRM06", "costs_fixed": 1000, "cash_flow": 40000, "inventory": 20000},
    {"name": "FIRM07", "costs_fixed": 1000, "cash_flow": 40000, "inventory": 20000},
    {"name": "FIRM08", "costs_fixed": 1000, "cash_flow": 40000, "inventory": 20000},
    {"name": "FIRM09", "costs_fixed": 1000, "cash_flow": 40000, "inventory": 20000},
    {"name": "FIRM10", "costs_fixed": 1000, "cash_flow": 40000, "inventory": 20000},
    {"name": "FIRM11", "costs_fixed": 1000, "cash_flow": 40000, "inventory": 20000},
]


# Helper functions
def create_environment(class_env, aec=True, render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    """

    internal_render_mode = render_mode if render_mode != "ansi" else "human"

    if aec:
        env = initialize_environment_aec(class_env, render_mode=internal_render_mode)

    else:
        env = class_env(render_mode=internal_render_mode)

    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)

    # This wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)

    # Provides a wide vareity of helpful user errors
    env = wrappers.OrderEnforcingWrapper(env)

    return env


def initialize_environment_aec(class_env, render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env.
    """

    env = class_env(render_mode=render_mode)
    env = parallel_to_aec(env)

    return env


def find_index_week(year, week):
    """
    Calculates the index of a given week in a year.

    Parameters:
    year (int): The year.
    week (int): The week number.

    Returns:
    int: The index of the week in the year.

    Example:
    >>> find_index_week(2024, 30)
    2
    """
    return (year - 2024)*52 + week - 28


def align_names(details):
    """
    Aligns the names of the firms so all names have the same length.

    Parameters:
    details (list): The details of the firms.

    Returns:
    list: The details of the firms with aligned names.

    Example:
    >>> align_names([
        {"name": "SMALL", "costs_fixed": 1000, "cash_flow": 10000}, 
        {"name": "BIG", "costs_fixed": 3000, "cash_flow": 40000}
    ])
    [
        {"name": "SMALL", "costs_fixed": 1000, "cash_flow": 10000}, 
        {"name": "BIG  ", "costs_fixed": 3000, "cash_flow": 40000}
    ]
    """

    length = max(len(detail["name"]) for detail in details)

    return [{**detail, "name": detail["name"] + " "*(length - len(detail["name"]))} for detail in details]


def create_details_firms(
        n_firms: Union[int, str], 
        type_firms: str, 
        costs_fixed_average: int=None, 
        cash_flow_average: int=None, 
        inventory_average: int=None
    ):
    """
    Creates the details of the firms participating in the game.

    Parameters:
    n_firms (int or str): The number of firms in the game.
    type_firms (str): The type of firms in the game. It can be either 'heterogeneous' or 'homogeneous'.

    Returns:
    list: The details of the firms participating in the game.

    Example:
    >>> np.random.seed(0)
    >>> create_details_firms(3, 'heterogeneous')
    [
        {'name': 'FIRM0', 'costs_fixed': 1294, 'cash_flow': 45335, 'inventory': 26525},
        {'name': 'FIRM1', 'costs_fixed': 1373, 'cash_flow': 64901, 'inventory': 13485},
        {'name': 'FIRM2', 'costs_fixed': 1158, 'cash_flow': 37982, 'inventory': 19312}
    ]
    """

    if type(n_firms) == str:
        if n_firms.lower() not in ['monopoly', 'duopoly', 'tripoly', 'oligopoly', 'competition']:
            raise ValueError("The number of firms needs to be either an integer or one of the following: 'monopoly', 'duopoly', 'tripoly', 'oligopoly', 'competition'.")
        if type_firms.lower() not in ['heterogeneous', 'homogeneous']:
            raise ValueError("The type of firms needs to be either 'heterogeneous' or 'homogeneous'.")
        return align_names(globals()[f"DETAILS_OF_FIRMS_{n_firms.upper()}_{type_firms.upper()}"])

    if costs_fixed_average is None:
        costs_fixed_average = 1000

    if cash_flow_average is None:
        cash_flow_average = 40000

    if inventory_average is None:
        inventory_average = 20000

    n_digits = ceil(np.log10(n_firms))

    if type_firms.lower() == 'homogeneous':
        return align_names([
            {"name": f"FIRM{i:0{n_digits}d}", "costs_fixed": costs_fixed_average, "cash_flow": cash_flow_average, "inventory": inventory_average}
            for i in range(n_firms)
        ])

    details_firms = []

    for i in range(n_firms):
        details_firms.append({
            "name": f"FIRM{i:0{n_digits}d}", 
            "costs_fixed": max(0, round(np.random.normal(loc=costs_fixed_average, scale=costs_fixed_average/6))), 
            "cash_flow": max(0, round(np.random.normal(loc=cash_flow_average, scale=cash_flow_average/3))), 
            "inventory": max(0, round(np.random.normal(loc=inventory_average, scale=inventory_average/3)))
        })
    
    return align_names(details_firms)

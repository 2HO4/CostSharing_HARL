"""
Calculation of the price of motor gasoline in the United States

"""

from typing import Callable

QUANTITY_MEDIAN = 62769  # median quantity (in thousands of barrels) of motor gasoline supplied in the United States


def calculate_price(quantity_supplied: int, quantity_demanded_forecast: int, quantity_demanded_real: int, function_demand: Callable[[int], float]=None) -> float:
    """
    Calculate the price of motor gasoline in the United States.

    Args:
        quantity_supplied: The total supply of gasoline (in thousands of barrels).
        quantity_demanded_forecast: The forecasted demand for gasoline (in thousands of barrels).
        quantity_demanded_real: The real demand for gasoline (in thousands of barrels).

    Returns:
        The price of gasoline in the United States.

    """
    
    if function_demand is None:
        function_demand = function_demand_linear

    return round(function_demand(quantity_supplied + quantity_demanded_forecast - quantity_demanded_real), 2)


def calculate_price_simple(quantity_supplied: float) -> float:
    return round(3.7567 - 1/22769*quantity_supplied, 2)


def function_demand_linear(quantity: int) -> float:
    return 1 - 1/22769*(quantity - QUANTITY_MEDIAN)

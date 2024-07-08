"""
Cost functions that takes into account the negative externalities caused by the supplies of petroleum, in particular, those of finished motor gasoline.

"""


from bisect import bisect_left
from typing import Callable
 

QUANTITY_MEDIAN = 62769  # median quantity of thousands of gasoline barrels supplied in the US


def build_cost_function(cost_function: Callable=None, characteristic: str='convex', normalize: float=QUANTITY_MEDIAN, scale=QUANTITY_MEDIAN):
    """
    Builds a cost function based on the given parameters.

    Args:
        cost_function (Callable, optional): The cost function to be used. If not provided, a default cost function based on the characteristic will be used. Defaults to None.
        characteristic (str, optional): The characteristic of the cost function. Defaults to 'convex'.
        factor (float, optional): The factor to be applied to the cost function which represents the ideal amount of production. Defaults to QUANTITY_MEDIAN.
        scale (float, optional): The scale to be applied to the cost function which represents the total value of the ideal amount of production. Defaults to 1*QUANTITY_MEDIAN

    Returns:
        Callable: The built cost function.

    Raises:
        ValueError: If the specified characteristic cost function is not found.
    """
    
    try:
        func = globals()[f'cost_{characteristic}'] if cost_function is None else cost_function
    
    except NameError:
        raise ValueError(f'{characteristic.upper()} cost function not found.')
    
    return scale_cost(normalize_cost(func, factor=normalize), factor=scale)


def scale_cost(cost_function, factor=QUANTITY_MEDIAN):
    """
    Scales a given cost function by a specified scale factor.

    Parameters:
    cost_function (function): The cost function to be scaled.
    scale_factor (float): The factor by which to scale the cost function.

    Returns:
    function: A lambda function that represents the scaled cost function.
    """

    return lambda x: cost_function(x) * factor


def normalize_cost(cost_function, factor=QUANTITY_MEDIAN):
    """
    Normalizes the cost function by dividing it by the cost at the given quantity.

    Parameters:
    - cost_function: A function that calculates the cost based on the quantity.
    - quantity: The quantity at which the cost is calculated.

    Returns:
    - A lambda function that takes a new quantity and returns the normalized cost.
    """

    return lambda x: cost_function(x) / cost_function(factor)


def cost_convex(n_barrels_supplied: int) -> float:
    """
    Convex cost function on the number of barrels of motor gasoline supplied.
    """

    return 0.001*n_barrels_supplied**2


def cost_linear(n_barrels_supplied: int) -> float:
    """
    Linear cost function on the number of barrels of motor gasoline supplied.
    """

    return n_barrels_supplied


def cost_concave(n_barrels_supplied: int) -> float:
    """
    Concave cost function on the number of barrels of motor gasoline supplied.
    """
    
    return 1000000/(n_barrels_supplied + 1)


def cost_piecewise(n_barrels_supplied: int) -> float:
    """
    Piecewise cost function on the number of barrels of motor gasoline supplied.
    """

    costs_barrel_per_step = {0: 0, 0.005: 0.05, 0.01: 0.1, 0.015: 0.2, 0.02: 0.35, 0.025: 0.55, 0.03: 0.8, 0.035: 1.1, 0.04: 1.45, 0.045: 1.85, 0.05: 2.3, 0.055: 2.8, 0.06: 3.35, 0.065: 4, 0.07: 4.75, 0.075: 5.6, 0.08: 6.55, 0.085: 7.6, 0.09: 8.75, 0.095: 10, 0.1: 11.35}

    return costs_barrel_per_step[round(5*round(n_barrels_supplied/5, 3))]*n_barrels_supplied


def cost_step(n_barrels_supplied: int) -> float:
    """
    Step cost function on the number of barrels of motor gasoline supplied.
    """

    costs_barrel_per_step = {0: 0, 0.005: 0.05, 0.01: 0.1, 0.015: 0.2, 0.02: 0.35, 0.025: 0.55, 0.03: 0.8, 0.035: 1.1, 0.04: 1.45, 0.045: 1.85, 0.05: 2.3, 0.055: 2.8, 0.06: 3.35, 0.065: 4, 0.07: 4.75, 0.075: 5.6, 0.08: 6.55, 0.085: 7.6, 0.09: 8.75, 0.095: 10, 0.1: 11.35}

    step = round(5*round(n_barrels_supplied/5, 3))

    return costs_barrel_per_step[step]*step


def cost_sigmoid(n_barrels_supplied: int) -> float:
    """
    Sigmoid cost function on the number of barrels of motor gasoline supplied.
    """

    return 100000/(1 + 10000**(-n_barrels_supplied))


if __name__ == '__main__':
    print(build_cost_function(characteristic='convex', normalize=QUANTITY_MEDIAN, scale=1.1*QUANTITY_MEDIAN)(QUANTITY_MEDIAN))
    print(cost_convex(QUANTITY_MEDIAN))
    print(normalize_cost(cost_convex, factor=QUANTITY_MEDIAN)(QUANTITY_MEDIAN))
    print(cost_convex(QUANTITY_MEDIAN*1.5))
    print(normalize_cost(cost_convex, factor=QUANTITY_MEDIAN)(0.09))

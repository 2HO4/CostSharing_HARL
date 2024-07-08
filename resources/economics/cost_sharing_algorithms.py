"""

Cost-sharing algorithms for the production of a discrete good.

"""

from typing import Callable


def use_cost_sharing(characteristic: str=None) -> Callable[[Callable, dict], dict]:
    if characteristic is None:
        characteristic = 'serial'

    try:
        return globals()[f'cost_sharing_{characteristic}']
    
    except NameError:
        raise ValueError(f'{characteristic.upper()} cost-sharing function not found.')


def cost_sharing_serial(cost_function: Callable[[int], float], quantities_to_produce: dict) -> dict:
    n = len(quantities_to_produce)
    agents_ordered = sorted(quantities_to_produce.keys(), key=lambda k: quantities_to_produce[k])
    
    q_n = tuple(
        sum(quantities_to_produce[agents_ordered[j]] for j in range(i)) + 
        (n - i)*quantities_to_produce[agents_ordered[i]] 
        for i in range(n)
    )

    return {
        agents_ordered[i]: cost_function(q_n[i])/(n - i) - 
        sum(cost_function(q_n[k])/((n - k)*(n - k - 1)) for k in range(i)) 
        for i in range(n)
    }


def cost_sharing_average(cost_function, quantity_to_produce: dict) -> dict:
    q_total = sum(quantity_to_produce.values())
    cost_total = cost_function(q_total)

    return {a: cost_total/q_total*quantity_to_produce[a] for a in quantity_to_produce}


if __name__ == '__main__':
    def c(q):
        return q if q <= 10 else 10*q - 90
    
    # q = (6, 10, 3)
    q = {1: 6, 3: 10, 4: 3}

    print(cost_sharing_serial(c, q))
    print(cost_sharing_average(c, q), '\n')

    # Cost sharing of emissions
    from cost_functions import build_cost_function
    
    ## Billions of barrels
    cost_func = build_cost_function(normalize=30, scale=0.8*30)
    n_barrels = {1: 00, 2: 00, 4: 10, 5: 20}
    costs_shared = use_cost_sharing()(cost_func, n_barrels)
    print(sum(costs_shared.values()))
    print(costs_shared)
    print('-'*50)
    
    # Thousands of barrels
    q_median = 31155
    price = 1.1
    cost_func = build_cost_function(normalize=q_median, scale=price*q_median)
    n_barrels = {1: 20000, 2: 10000, 3: 5000, 4: 5000}  # sum = 40000
    costs_shared = use_cost_sharing()(cost_func, n_barrels)
    print(sum(costs_shared.values()))
    print(costs_shared, end='\n\n')

    n_barrels = {1: 10000, 2: 11000, 3: 6000, 4: 4155}  # sum = 31155
    costs_shared = use_cost_sharing()(cost_func, n_barrels)
    print(sum(costs_shared.values()))
    print(costs_shared, end='\n\n')

    n_barrels = {1: 8000, 2: 7000, 3: 6000, 4: 4155}  # sum = 15155
    costs_shared = use_cost_sharing()(cost_func, n_barrels)
    print(sum(costs_shared.values()))
    print(costs_shared, end='\n\n')

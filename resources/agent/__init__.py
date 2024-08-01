# Class of a firm in the cost-sharing game reinforcement learning environment. This firm can produce, sell, and earn money.
class Firm:
    def __init__(self, name, costs_fixed, cash_flow, inventory=0):
        self._name = name
        self._costs_fixed = costs_fixed
        self._cash_flow = cash_flow
        self._inventory = inventory
        self._last_actions = None
    
    def get_name(self):
        return self._name
    
    def get_inventory(self):
        return self._inventory
    
    def get_cash_flow(self):
        return self._cash_flow
    
    def get_costs_fixed(self):
        return self._costs_fixed
        
    def produce(self, quantity):
            """
            Produces a specified quantity of barrels. Each production requires a fixed cost.
            Args:
                quantity (int): The quantity of barrels to produce.
            """
            if quantity > 0:
                self._cash_flow -= self._costs_fixed
                self._inventory += quantity
                self._last_actions = f'produced {quantity}K barrel(s)'
            elif not quantity:
                self._cash_flow -= self._costs_fixed//2
                self._last_actions = 'produced nothing'
            else:
                self._last_actions = 'produced {quantity}K barrel(s) (INVALID)'
    
    def sell(self, quantity):
        """
        Sell a specified quantity of barrels.
        Args:
            quantity (int or str): The quantity of barrels to sell. If "all" is provided, all barrels will be sold.
        """
        if quantity == "all":
            self._inventory = 0
            self._last_actions += ', sold all barrels'
        elif quantity:
            self._inventory -= quantity
            self._last_actions += f', sold {quantity}K barrel(s)'
        else:
            self._last_actions += ', sold nothing'
    
    def earn(self, amount):
        self._cash_flow += amount
        self._last_actions += f", earned USD {round(amount, 2)}K"
        
        return amount
    
    def __str__(self):
        return f'{self._name}: USD {round(self._cash_flow, 1)}K in cash flow and {self._inventory}K units in inventory, last actions: {self._last_actions}'
    
    def __repr__(self):
        return f'Firm {self._name}: {{costs_fixed: {self._costs_fixed}, cash_flow: {self._cash_flow}, inventory: {self._inventory}}}'


if __name__ == '__main__':
    firm = Firm('A', 100, 1000, 10)
    print(firm.__repr__())
    print(firm)
    print(firm.produce(5))
    print(firm)
    print(firm.sell(3, 20))
    print(firm)
    print(firm.sell(3, 20))
    print(firm)

    
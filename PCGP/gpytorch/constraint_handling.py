import torch


class ConstraintsModifications():
    """helper class to handle constraints on parameters and their transformations"""
    def __init__(self, constraints):
        """
        :param constraints: gpytorch.constraints.Constraint or list of gpytorch.constraints.Constraint
        """
        if not isinstance(constraints, list):
            constraints = [constraints]
        self.constraints = constraints

    def read_constraint(self, index = 0):
        """
        Read out constraint type and value at given index (standard index = 0)
        
        :param index: int, index of constraint to read
        :return: tuple (str, float), constraint type as string and constraint value
        """
        if self.constraints[index] == False:
            return None, 0.0
        string = self.constraints[index].__repr__()
        for i in range(len(string)):
                if string[i] == "(":
                    il = i
                elif string[i] == ")":
                    ih = i
        condition = string[:il]
        if il+1!=ih: value = eval(string[il+1:ih]) 
        else: value = 0.
        return condition, value
    
    def is_fulfilled(self, value, index = 0): 
        """checks if given value fulfills constraint at given index (standard index = 0)
        :param value: float, value to check
        :param index: int, index of constraint to check
        :return: bool, True if constraint is fulfilled, False otherwise"""
        condition, limit = self.read_constraint(index = index)
        if condition == "GreaterThan" and value <= limit:
                return False
        elif condition == "LessThan" and value >= limit:
                return False
        elif condition == "Positive" and value <= 0:
                return False
        elif condition == "Interval":
            lower, upper = limit
            if value < lower or value > upper:
                return False
            else: 
                return True
        else: 
            return True

    def init_val_from_constraint(self, index = 0): 
        """generates a "standard" initial value that fulfills the constraint at given index (standard index = 0)"""
        condition, values = self.read_constraint(index = index)
        if condition == "GreaterThan":
            init_val = values + 1
        elif condition == "LessThan":
            init_val = values - 1
        elif condition == "Positive":
            init_val = values + 1
        elif condition == "Interval":
            lower, upper = values
            init_val = lower + (upper-lower)/2. 
        else: init_val = 0.0
        return init_val  
    
    def inverse_derivatives(self, transformed_parameter, index = 0):
        """returns first and second derivative of inverse transform corresponding to constraint, evaluated at transformed_parameter value
        :param transformed_parameter: torch.tensor, value at which to evaluate the derivatives
        :param index: int, index of constraint to consider
        :return: tuple (torch.tensor, torch.tensor), first and second derivative of inverse transform"""

        condition, value = self.read_constraint(index = index)
        if condition == "Interval":
            lower, upper = value
            fx = (transformed_parameter - lower)/(upper - lower)
            derivative = (1/fx + 1/(1-fx))/(upper - lower)
            second_derivative = (-1/fx**2 + 1/(1-fx)**2)/(upper - lower)**2
        elif condition == "LessThan":
            fx = -(transformed_parameter - value)
            derivative = 1/-torch.expm1(-fx)
            second_derivative = 1/(torch.expm1(-fx)+torch.expm1(fx))
        elif condition == "Positive" or condition == "GreaterThan":
             fx = transformed_parameter - value
             derivative = 1/(-torch.expm1(-fx))
             second_derivative = -1/(torch.expm1(-fx)+torch.expm1(fx))
        else: 
            derivative = torch.tensor([1])
            second_derivative = torch.tensor([0]) 

        return derivative, second_derivative    

     
       
 

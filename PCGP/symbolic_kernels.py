import sympy as sp

class symbolic_parametrization_kernel:
    """
    Base class that calculates the symbolic kernel matrix based on a user specified parametrization matrix B and a base_kernel in n = number_of_input_dimensions dimensions.
    """
    def __init__(self, B, eff_input_dims = None, base_kernel = None, shared_base_kernel = False):
        """
        Parameters

        :param B: sp.matrices.Matrix, user specified parametrization matrix
        :param number_of_input_dimensions: int, number of input dimensions. either this or base_kernel_arguments needs to be specified
        :param base_kernel_arguments: list of strings containing expressions with indexed inputs of the form "x[0]". either this or number_of_input_dimensions needs to be specified.
        :param base_kernel: positive semi-definite and symmetric function of x and y, returning sympy object for symbolic input. If not specified, RBF kernel is used.
        """     
 
        self.number_of_input_dimensions = eff_input_dims
          
        self.x = sp.symbols(f'x1:{self.number_of_input_dimensions + 1}')   # creates (x1, x2,..)
        self.y = sp.symbols(f'y1:{self.number_of_input_dimensions + 1}') 
        self.Dx = sp.symbols(f'Dx1:{self.number_of_input_dimensions + 1}') # corresponding derivatives
        self.Dy = sp.symbols(f'Dy1:{self.number_of_input_dimensions + 1}')
        self.set_of_derivatives = set(self.Dx) | set(self.Dy)
        self.B = B
        self.Bx = self.B(self.Dx, self.x)
        self.By = self.B(self.Dy, self.y)
        
        self.num_tasks = self.Bx.shape[0]
        self.nr_of_parametrization_vectors = self.Bx.shape[1]
        self.expanded_BinBjn = {}
        for n in range(self.nr_of_parametrization_vectors):
            self.expanded_BinBjn[n] = sp.expand(self.Bx[:, n] * self.By[:, n].T)
        #build base kernel and parameter maps for the different parametrization vectors:
        if base_kernel:
            base_kernel_symbols = base_kernel(self.x, self.y).free_symbols- set(self.x)- set(self.y)
            self.base_kernel_template = base_kernel(self.x, self.y)
            if shared_base_kernel:
                self.kernel_parameter_maps = [
                {sym: sym for sym in base_kernel_symbols}
                    for _ in range(self.nr_of_parametrization_vectors)]
            else:
                self.kernel_parameter_maps = []
                for n in range(self.nr_of_parametrization_vectors):
                    for sym in base_kernel_symbols:
                        new_sym = sp.symbols(f"{sym}_{n}")
                        self.kernel_parameter_maps.append({sym: new_sym})        
        else:
            amplitude, lengthscale = sp.symbols("amplitude lengthscale")
            self.base_kernel_template = amplitude * sp.exp(-1/(2*lengthscale) * sum((xi - yi)**2 for xi, yi in zip(self.x, self.y)))
            
            if shared_base_kernel:
                self.kernel_parameter_maps = [
                {amplitude: amplitude, lengthscale: lengthscale}
                    for _ in range(self.nr_of_parametrization_vectors)]
            else:
                self.kernel_parameter_maps = []
                for n in range(self.nr_of_parametrization_vectors):
                    amp_n, ls_n = sp.symbols(f"amplitude_{n} lengthscale_{n}")
                    self.kernel_parameter_maps.append({
                        amplitude: amp_n,
                        lengthscale: ls_n})
        #automatically define all symbols that are not inputs or derivatives as parameters
        base_kernel_symbols = set().union(*(mapping.values() for mapping in self.kernel_parameter_maps))
        self.parameters = { 
            str(sym): None
            for sym in (base_kernel_symbols.union(self.Bx.free_symbols) - set(self.Dx)- set(self.x)- set(self.y))}
        

    def get_symbolic_kernel(self): 
        all_items_to_differentiate, sorted_items = self._needed_derivatives()
        template_diff = self._differentiate_base_kernel(all_items_to_differentiate)
        symbolic_kernel = self._substitute_differential_terms(sorted_items, template_diff)
        return sp.simplify(symbolic_kernel)
        
    def _needed_derivatives(self):
        all_items_to_differentiate = {}
        sorted_items = {} #keys: n values: nr_of_parametrization vectors, values: list of tuples (key: written operator, value: list of tuples (input, order of differentiation)) sorted by order of differentiation
        diff_conversion = {Dx: x for Dx, x in zip(self.Dx+self.Dy, self.x+self.y)} #differentiate wrt x or y 
        for n in range(self.nr_of_parametrization_vectors):
            symbolic_expr = self.expanded_BinBjn[n]
            used_differential_terms = {}

            for i in range(self.num_tasks):
                for j in range(self.num_tasks):
                    terms = sp.Add.make_args(symbolic_expr[i,j])
                    for term in terms:
                        factors = sp.Mul.make_args(term)
                        diff_factors = []
                        needed_differentiations = []
                        for f in factors:
                            #differentiate between first derivatives and higher order
                            if isinstance(f, sp.Symbol) and f in self.set_of_derivatives:
                                diff_factors.append(f)
                                needed_differentiations.append((diff_conversion[f], 1))
                            elif isinstance(f, sp.Pow):
                                base, exp = f.args
                                if base in self.set_of_derivatives:
                                    diff_factors.append(f)
                                    needed_differentiations.append((diff_conversion[base], exp))
                        #make correct dict entries between written operator and sympy-friendly needed differentiations
                        if diff_factors:
                            op = sp.Mul(*diff_factors)
                            used_differential_terms[op] = needed_differentiations
            all_items_to_differentiate.update(used_differential_terms)
            sorted_items[n] = sorted(used_differential_terms.items(), key=lambda item: sorted([p for _, p in item[1]], reverse=True), reverse=True)
        return all_items_to_differentiate, sorted_items
        
    def _differentiate_base_kernel(self, all_items_to_differentiate):
        #differentiate the base kernel template according to all needed differentiations and store in dict for substitution later
        template_diff = {}
        for key in all_items_to_differentiate:
                template_diff[key] = sp.diff(self.base_kernel_template, *all_items_to_differentiate[key]) 
        return template_diff
    
    def _substitute_differential_terms(self, sorted_items, template_diff):  
        #operator_kernel = {}
        differential_substitutions = {}
        symbolic_matrix = sp.zeros(self.num_tasks, self.num_tasks)
        for n in range(self.nr_of_parametrization_vectors):       
            differential_substitutions[n] = {}
            for key, _ in sorted_items[n]:
                differential_substitutions[n][key] = template_diff[key].subs(self.kernel_parameter_maps[n])
       
            base_kernel_n = self.base_kernel_template.subs(self.kernel_parameter_maps[n])
            
            for i in range(self.num_tasks):
                for j in range(self.num_tasks):
                    expr = self.expanded_BinBjn[n][i, j]
                    
                    result = 0
                    for term in sp.Add.make_args(expr):
                        new_term = term
                        if not any(new_term.has(D) for D in self.set_of_derivatives):
                            new_term = new_term * base_kernel_n
                        else:
                            for key, value in sorted_items[n]:
                                if new_term.has(key):
                                    new_term = new_term.subs(key, differential_substitutions[n][key])
                        result += new_term
                    symbolic_matrix[i, j] += result
        return symbolic_matrix
    

class symbolic_mercer_kernel:
    """
    Base class that calculates the symbolic mercer kernel based on user specified base_functions and (optionally) the covariance matrix Sigma in number_of_input_dimensions dimensions.
    """
     
    def __init__(self, base_functions, Sigma = None, number_of_input_dimensions = 1):
        """
        Parameters
        
        :param base_functions: sp.matrices.Matrix, user specified matrix containing the base functions as columns
        :param parameters: dictionary including all parameters (as keys) in B and the base-kernel hyperparameters A (amplitude) and l (lengthscale)
        :param number_of_input_dimensions: int, number of input dimensions
        """
        self.base_functions = base_functions
        self.Sigma = Sigma
        self.x = sp.symbols(f'x1:{number_of_input_dimensions + 1}')  # creates (x1, x2,..)
        self.y = sp.symbols(f'y1:{number_of_input_dimensions + 1}')  
        self.parameters = {
            str(sym): None
            for sym in (
                self.base_functions(self.x).free_symbols
                - set(self.x)
            )
        }
    
    def get_symbolic_kernel(self): 
        """
        Calculates the symbolic kernel matrix based on the base functions and the covariance matrix Sigma.
        """
        if self.Sigma is None:
            mercer_kernel = sp.simplify(self.base_functions(self.x)@self.base_functions(self.y).transpose())
        else:
            mercer_kernel = sp.simplify(self.base_functions(self.x)@self.Sigma@self.base_functions(self.y).transpose())
        return mercer_kernel


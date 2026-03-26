import sympy as sp

class symbolic_parametrization_kernel:
    """
    Base class that calculates the symbolic kernel matrix based on a user specified parametrization matrix B and a base_kernel in n = number_of_input_dimensions dimensions.
    """
    def __init__(self, B, number_of_input_dimensions=None, base_kernel_arguments = None, base_kernel = None):
        """
        Parameters

        :param B: sp.matrices.Matrix, user specified parametrization matrix
        :param number_of_input_dimensions: int, number of input dimensions. either this or base_kernel_arguments needs to be specified
        :param base_kernel_arguments: list of strings containing expressions with indexed inputs of the form "x[0]". either this or number_of_input_dimensions needs to be specified.
        :param base_kernel: positive semi-definite and symmetric function of x and y, returning sympy object. If not specified, RBF kernel is used.
        """     
   
        if base_kernel_arguments:
            self.number_of_input_dimensions = len(base_kernel_arguments)
        elif number_of_input_dimensions:
            self.number_of_input_dimensions = number_of_input_dimensions
        else:
            raise ValueError("Either number of dimensions or base_kernel_arguments (explicit argument list) must be provided")
        
        self.x = sp.symbols(f'x1:{self.number_of_input_dimensions + 1}')   # creates (x1, x2,..)
        self.y = sp.symbols(f'y1:{self.number_of_input_dimensions + 1}') 
        self.Dx = sp.symbols(f'Dx1:{self.number_of_input_dimensions + 1}') #corresponding derivatives
        self.Dy = sp.symbols(f'Dy1:{self.number_of_input_dimensions + 1}')
        if not base_kernel:
            amplitude, lengthscale = sp.symbols("amplitude, lengthscale")
            self.base_kernel = amplitude*sp.exp(-1/(2*lengthscale)*sum((xi - yi)**2 for xi, yi in zip(self.x, self.y)))
        else:
            self.base_kernel = base_kernel(self.x, self.y)
        self.B = B

        self.parameters = { #automatically define all symbols that are not inputs or derivatives as parameters
            str(sym): None
            for sym in (
                self.base_kernel.free_symbols
                .union(self.B(self.Dx, self.x).free_symbols)
                - set(self.Dx)
                - set(self.x)
                - set(self.y)
            )
        }
        self.num_tasks = self.B(self.Dx, self.x).shape[0]

    
    def get_symbolic_kernel(self): 
        operator_kernel = self.B(self.Dx, self.x)*self.B(self.Dy, self.y).T 
        diff_conversion = {Dx: x for Dx, x in zip(self.Dx+self.Dy, self.x+self.y)}
        used_differential_terms = {}
        for i in range(self.num_tasks):
            for j in range(self.num_tasks):
                symbolic_expr = sp.expand(operator_kernel[i,j])
                terms = sp.Add.make_args(symbolic_expr)
                for term in terms:
                    factors = sp.Mul.make_args(term)
                    diff_factors = []
                    needed_differentiations = []
                    for f in factors:
                        if isinstance(f, sp.Symbol) and (f in self.Dx or f in self.Dy):
                            diff_factors.append(f)
                            needed_differentiations.append((diff_conversion[f], 1))
                        elif isinstance(f, sp.Pow):
                            base, exp = f.args
                            if base in self.Dx or base in self.Dy:
                                diff_factors.append(f)
                                needed_differentiations.append((diff_conversion[base], exp))
                    if diff_factors:
                        op = sp.Mul(*diff_factors)
                        used_differential_terms[op] = needed_differentiations
                   
        sorted_items = sorted(used_differential_terms.items(), key=lambda item: sorted([p for _, p in item[1]], reverse=True), reverse=True)
        differential_substitutions = {}
        for key, value in sorted_items:
            differential_substitutions[key] = sp.diff(self.base_kernel, *value)
        
        #define differentials correctly:
        for i in range(self.num_tasks):
            for j in range(self.num_tasks):
                terms_of_BBT_entry = sp.matrices.Matrix(sp.Add.make_args(sp.expand(operator_kernel[i,j])))
                for l in range(len(terms_of_BBT_entry)):
                    term = terms_of_BBT_entry[l]
                    if all(str(Dxi) not in str(term) for Dxi in self.Dx) and all(str(Dyi) not in str(term) for Dyi in self.Dy): #needs to be able to handle symbols or numbers alone
                            terms_of_BBT_entry[l] = terms_of_BBT_entry[l].subs(term, term*self.base_kernel)
                    else:
                        for key, value in sorted_items:
                            if str(key) in str(terms_of_BBT_entry[l]):
                                terms_of_BBT_entry[l] = terms_of_BBT_entry[l].subs(key, differential_substitutions[key])
                               
                operator_kernel[i,j] = sum(terms_of_BBT_entry)
        return operator_kernel
    
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
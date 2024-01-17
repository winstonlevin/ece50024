import numpy as np
import scipy as sp

# Exercise 3 (a) -------------------------------------------------------------------------------------------------------
x = np.linspace(-1, 1, 50)
l1 = sp.special.eval_legendre(1, x)
l2 = sp.special.eval_legendre(2, x)
l3 = sp.special.eval_legendre(3, x)
l4 = sp.special.eval_legendre(4, x)

np.random.seed(3)


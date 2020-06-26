import numpy as np
import math
import sys
import random as rd

def gen_matrix():
    n = rd.randint(1, 100)
    m = rd.randint(1, 100)

    c = np.random.rand(n) # objective function
    b = np.random.rand(m)
    a = np.random.rand(m, n)

    return a, b, c

# def gen_matrices(k):
    
#     res = []
    
#     for _ in range(k):
#         n = rd.randint(1, 100)
#         m = rd.randint(1, 100)
#         mat = gen_matrix(n, m)
#         res.append(mat)
    
#     return res

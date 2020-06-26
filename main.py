import sys
import numpy as np
import math
import time
import simplex_method
from simplex_method import simplex
from generate_matrix import gen_matrix
from scipy.optimize import linprog

if __name__ == '__main__':
    dura_self_implement = []
    dura_lib = []
    for _ in range(100):
        a, b, c = gen_matrix()
        start = time.time()
        simplex(c, a, b)
        end = time.time()

        duration = end - start
        dura_self_implement.append(duration)

        start = time.time()
        opt = linprog(c=c, A_ub=a, b_ub=b, method="simplex")
        end = time.time()
        duration = end - start
        dura_lib.append(duration)

    mean_self_implement = sum(dura_self_implement) / len(dura_self_implement)
    mean_lid = sum(dura_lib) / len(dura_lib)




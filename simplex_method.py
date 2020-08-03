import numpy as np
import math
import random
import sys


def e(l, p):
    v = np.zeros(l)
    v[p] = 1
    return v


def primal_simplex(c, A, b, modified={}):
    status = ''
    n = len(c)
    m = len(b)
    # init
    if modified == {}:
        N_index = np.array(range(1, n + 1))
        B_index = np.array(range(n + 1, n + m + 1))

        B = np.identity(m)
        A = np.concatenate([A, B], axis=1)
        N = A[:, (N_index - 1)]
        B = A[:, (B_index - 1)]

        c_b = np.zeros(m)
        c_n = c
        c = np.concatenate([c_n, c_b])

        x_b = b
        z_n = - c_n

        x = np.concatenate([np.zeros(n), x_b])
        z = np.concatenate([z_n, np.zeros(m)])
        s, t = 0, 0
        i, j = 0, 0
    else:
        x = modified['x']
        z = modified['z']

        B_index = modified['B_index']
        N_index = modified['N_index']

        c = np.concatenate([c, np.zeros(m)])
        A = np.concatenate([A, np.identity(m)], axis=1)

        N = A[:, (N_index - 1)]
        B = A[:, (B_index - 1)]

        c_b = c[B_index - 1]
        c_n = c[N_index - 1]

        x_b = np.dot(np.linalg.inv(B), b)
        z_n = np.dot(np.dot(np.linalg.inv(B), N).T, c_b) - c_n

        x = np.zeros(m + n)
        z = np.zeros(m + n)

        x[B_index - 1] = x_b
        z[N_index - 1] = z_n

    while True:
        if np.all(z_n >= 0):
            status = 'optimal'
            # print(z_n)
            break

        j = np.where(z_n < 0)[0][0]
        e_j = e(n, j)
        delta_xb = np.dot(np.dot(np.linalg.inv(B), N), e_j)

        t = max(delta_xb / x_b) ** -1
        i = np.argmax(delta_xb / x_b)
        e_i = e(m, i)

        delta_z_n = - np.dot(np.dot(np.linalg.inv(B), N).T, e_i)

        s = z_n[j] / delta_z_n[j]

        x_b = x_b - t * delta_xb
        z_n = z_n - s * delta_z_n

        x[B_index - 1] = x_b
        z[N_index - 1] = z_n

        j = N_index[j]
        i = B_index[i]

        x[j - 1] = t
        z[i - 1] = s

        # print(np.where(B_index == i), j)
        B_index[np.where(B_index == i)[0][0]] = j
        N_index[np.where(N_index == j)[0][0]] = i

        B_index = np.sort(B_index)
        N_index = np.sort(N_index)

        x_b = x[B_index - 1]
        z_n = z[N_index - 1]

        B = A[:, (B_index-1)]
        N = A[:, (N_index-1)]

    obj = np.dot(c.T, x)

    result = {
        'status': status,
        'optimal value': obj,
        'x': x,
        'B_index': B_index,
        'N_index': N_index
    }
    return result


def dual_simplex(c, A, b):
    status = ''
    n = len(c)
    m = len(b)
    # init
    N_index = np.array(range(1, n + 1))
    B_index = np.array(range(n + 1, n + m + 1))

    B = np.identity(m)
    A = np.concatenate([A, B], axis=1)
    N = A[:, (N_index - 1)]
    B = A[:, (B_index - 1)]

    c_b = np.zeros(m)
    c_n = c
    c = np.concatenate([c_n, c_b])

    x_b = b
    z_n = - c_n

    x = np.concatenate([np.zeros(n), x_b])
    z = np.concatenate([z_n, np.zeros(m)])
    s, t = 0, 0
    i, j = 0, 0

    while True:
        if np.all(x_b >= 0):
            status = 'optimal'
            break

        i = np.where(x_b < 0)[0][0]
        e_i = e(m, i)
        # delta_xb = np.dot(np.dot(np.linalg.inv(B), N), e_j)
        delta_z_n = - np.dot(np.dot(np.linalg.inv(B), N).T, e_i)

        s = max(delta_z_n / z_n) ** -1
        j = np.argmax(delta_z_n / z_n)
        e_j = e(n, j)

        # delta_z_n = - np.dot(np.dot(np.linalg.inv(B), N).T, e_i)
        delta_xb = np.dot(np.dot(np.linalg.inv(B), N), e_j)

        t = x_b[i] / delta_xb[i]

        x_b = x_b - t * delta_xb
        z_n = z_n - s * delta_z_n

        x[B_index - 1] = x_b
        z[N_index - 1] = z_n

        j = N_index[j]
        i = B_index[i]

        x[j - 1] = t
        z[i - 1] = s

        # print(np.where(B_index == i), j)
        B_index[np.where(B_index == i)[0][0]] = j
        N_index[np.where(N_index == j)[0][0]] = i

        B_index = np.sort(B_index)
        N_index = np.sort(N_index)

        x_b = x[B_index - 1]
        z_n = z[N_index - 1]

        B = A[:, (B_index-1)]
        N = A[:, (N_index-1)]

    obj = np.dot(c.T, x)

    result = {
        'status': status,
        'optimal value': obj,
        'x': x,
        'z': z,
        'B_index': B_index,
        'N_index': N_index
    }
    return result


def two_phase(c, A, b):
    c_mod = np.ones(len(c)) * -1
    modified_prob = dual_simplex(c_mod, A, b)

    N_index = modified_prob['N_index']
    B_index = modified_prob['B_index']

    result = primal_simplex(c, A, b, modified=modified_prob)
    return result


def simplex_method(c, A, b):
    x_b = b
    z_n = -c
    if np.all(x_b >= 0) and np.any(z_n < 0):
        print(primal_simplex(c, A, b))
    elif np.all(z_n >= 0) and np.any(x_b < 0):
        print(dual_simplex(c, A, b))
    else:
        print(two_phase(c, A, b))

if __name__ == '__main__':
    sys.stdout = open('out.txt', 'w')

    c = np.array([2, -6, 0])
    A = np.array([
        [-1, -1, -1],
        [2, -1, 1]
    ])
    b = np.array([-2, 1])
    print(simplex_method(c, A, b))

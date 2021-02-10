import time

import numpy as np
from sympy import S
from sympy import solveset
from sympy import Symbol

from utils import cheby_poly
from utils import report


def codebook(args):
    N, d1, d2 = args.N, args.d1, args.d2

    codebook = np.zeros([N, d1, d2])
    consts = (np.arange(N) + 1) / (N + 5)

    x = Symbol("x", real=True)

    start = time.time()

    for i, const in enumerate(consts):
        equation = cheby_poly(x, order=d1) - const
        a = list(solveset(equation, x, domain=S.Reals))
        for j, aa in enumerate(a):
            equation = cheby_poly(x, order=d2) - aa
            b = list(solveset(equation, x, domain=S.Reals))

            # rearrange codes.
            if (args.experiment == 2) and (j == 1):
                b[0], b[-1] = b[-1], b[0]
            if (args.experiment == 3) and (j == 1):
                b[0], b[-1] = b[-1], b[0]
            elif (args.experiment == 3) and (j == 2):
                b = [b[1], b[3], b[0], b[2]]

            codebook[i, j] = b

    report(args, start, name="Codebook")

    return codebook


def encode(Ap, Bp, codebook, args):
    dim1_split, dim2 = Ap[0].shape
    dim2, dim3_split = Bp[0].shape
    N, d1, d2, m, n = args.N, args.d1, args.d2, args.m, args.n

    Aenc = np.zeros([N, d2, dim1_split, dim2])
    Benc = np.zeros([N, d1, dim2, dim3_split])

    start = time.time()

    for i in range(N):
        for j in range(d2):
            code = codebook[i, 0, j]
            for k in range(m):
                Aenc[i, j] = Aenc[i, j] + Ap[k] * pow(cheby_poly(code, order=d1), k)

    for i in range(N):
        for j in range(d1):
            code = codebook[i, j, 0]
            for k in range(n):
                Benc[i, j] = Benc[i, j] + Bp[k] * pow(cheby_poly(code, order=d2), k)

    report(args, start, name="Encode")

    return Aenc, Benc


def get_Van_inv(codebook, gets, args):
    N, d1, d2, m, n = args.N, args.d1, args.d2, args.m, args.n
    Van = np.zeros([m * n, m * n])

    start = time.time()

    for i in range(m * n):
        ids = gets[i]
        code = codebook[ids[0] - 1, ids[1], ids[2]]
        fx = cheby_poly(code, order=d1)
        gx = cheby_poly(code, order=d2)

        if args.experiment == 1:
            Van[i, 0] = 1
            Van[i, 1] = fx
            Van[i, 2] = gx
            Van[i, 3] = gx * fx
        elif args.experiment == 2:
            Van[i, 0] = 1
            Van[i, 1] = fx
            Van[i, 2] = fx * fx

            Van[i, 3] = gx
            Van[i, 4] = fx * gx
            Van[i, 5] = fx * fx * gx

            Van[i, 6] = gx * gx
            Van[i, 7] = fx * gx * gx
            Van[i, 8] = fx * fx * gx * gx
        elif args.experiment == 3:
            Van[i, 0] = 1
            Van[i, 1] = fx
            Van[i, 2] = fx * fx
            Van[i, 3] = fx * fx * fx

            Van[i, 4] = gx
            Van[i, 5] = fx * gx
            Van[i, 6] = fx * fx * gx
            Van[i, 7] = fx * fx * fx * gx

            Van[i, 8] = gx * gx
            Van[i, 9] = fx * gx * gx
            Van[i, 10] = fx * fx * gx * gx
            Van[i, 11] = fx * fx * fx * gx * gx

            Van[i, 12] = gx * gx * gx
            Van[i, 13] = fx * gx * gx * gx
            Van[i, 14] = fx * fx * gx * gx  * gx
            Van[i, 15] = fx * fx * fx * gx  * gx  * gx

    Van_inv = np.linalg.inv(Van)

    report(args, start, name="Van inverse")

    return Van_inv


def decode(Cenc, codebook, Van_inv, args):
    m, n = args.m, args.n
    _, dim1_split, dim3_split = Cenc.shape

    I = np.identity(dim1_split)
    Van_inv_kron = np.kron(Van_inv, I)

    Cenc = Cenc.reshape(-1, Cenc.shape[-1])
    Cenc = Cenc[: dim1_split * m * n, :]

    start = time.time()

    C = np.matmul(Van_inv_kron, Cenc)
    C = np.vsplit(C, m * n)
    C = [np.vstack(C[i * m : (i + 1) * m]) for i in range(n)]
    C = np.hstack(C)

    report(args, start, name="Decode")

    return C

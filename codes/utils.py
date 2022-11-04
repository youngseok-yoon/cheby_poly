import time
import numpy as np


def cheby_poly(input, order=2):  # orderth chebyshev
    if order == 1:
        return input
    elif order == 2:
        return 2 * pow(input, 2) - 1
    elif order == 3:
        return 4 * pow(input, 3) - 3 * input
    elif order == 4:
        return 8 * pow(input, 4) - 8 * pow(input, 2) + 1
    elif order == 5:
        return 16 * pow(input, 5) - 20 * pow(input, 3) + 5 * input
    elif order == 6:
        return 32 * pow(input, 6) - 48 * pow(input, 4) + 18 * pow(input, 2) - 1
    elif order == 7:
        return 64 * pow(input, 7) - 112 * pow(input, 5) + 56 * pow(input, 3) - 7 * input
    elif order == 8:
        return 128 * pow(input, 8) - 256 * pow(input, 6) + 160 * pow(input, 4) - 32 * pow(input, 2) + 1
    elif order == 9:
        return 256 * pow(input, 9) - 576 * pow(input, 7) + 432 * pow(input, 5) - 120 * pow(input, 3) + 9 * input
    else:
        raise ValueError(f"not supported order {order} for chebyshev polynomial")


def report(args, start, name):
    spent = format(time.time() - start, ".5f")
    print(f"{name} time: {spent} seconds.")
    args.computation_time[name] = float(spent)
    pass


def straggle(gap=10):
    t = time.time()
    while time.time() < t + gap:
        a = 100 + 100
    pass


def split_matrix(A, B, args):
    # dim1_split = int(dim1 / m), dim3_split = int(dim3 / n)
    # Ap: m length list of dim1_split * dim2
    # Bp: n length list of dim2 * dim3_split
    Ap = np.split(A, args.m)
    Bp = np.hsplit(B, args.n)

    return Ap, Bp


def get_shape(args):
    A_enc_shape = [args.d2, int(args.dim1 / args.m), args.dim2]
    B_enc_shape = [args.d1, args.dim2, int(args.dim3 / args.n)]
    C_rec_shape = [int(args.dim1 / args.m), int(args.dim3 / args.n)]

    return A_enc_shape, B_enc_shape, C_rec_shape


def compute(Aenc, Benc):
    return np.matmul(Aenc, Benc)

import argparse
import threading
import time

import numpy as np
from mpi4py import MPI

import utils
import functions


def main(args):
    A_enc_shape, B_enc_shape, C_rec_shape = utils.get_shape(args)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size > 1:
        args.N = size - 1

    if rank == 0:
        # set stragglers.
        stragglers = list(np.random.uniform(0, 1, args.N) <= args.straggle_prob)
        args.stragglers = {}
        for i, straggler in enumerate(stragglers):
            comm.send(straggler, dest=i + 1, tag=99)
            args.stragglers[f"Worker {i + 1}"] = bool(straggler)

        args.computation_time = {}

        # get codebook. codebook: N * d1 * d2.
        codebook = functions.codebook(args)

        # set input matrice, A: dim1 * dim2, B: dim2 * dim3
        A = np.random.randint(0, 256, (args.dim1, args.dim2))
        B = np.random.randint(0, 256, (args.dim2, args.dim3))

        # split matrice.
        Ap, Bp = utils.split_matrix(A, B, args)

        # encode splited matrice.
        Aenc, Benc = functions.encode(Ap, Bp, codebook, args)

        # start parallel computing. Set buffers.
        Crec = []
        for i in range(args.N * args.d1 * args.d2):
            Crec.append(np.zeros(C_rec_shape))

        start = time.time()

        reqA, reqB, reqC = [], [], []
        for i in range(args.N):
            # send Aenc and Benc, prepare for Crec.
            reqA.append(comm.Isend(Aenc[i], dest=i + 1, tag=3))
            reqB.append(comm.Isend(Benc[i], dest=i + 1, tag=4))
            for j in range(args.d1):
                for k in range(args.d2):
                    reqC.append(comm.Irecv(Crec[i * args.d1 * args.d2 + j * args.d2 + k], source=i + 1, tag=7 + j * args.d2 + k))

        MPI.Request.Waitall(reqA)
        MPI.Request.Waitall(reqB)

        utils.report(args, start, name="Send")

        # wait and syncronize for all workers to receive matrice.
        comm.Barrier()

        # receive n_required computed results
        Cenc, gets = [], []
        for _ in range(args.m * args.n):
            gets.append(MPI.Request.Waitany(reqC))

        utils.report(args, start, name="Send and receive")

        for i in gets:
            Cenc.append(Crec[i])

        args.gets = [[i // (args.d1 * args.d2) + 1, (i // args.d2) % args.d1, i % args.d2] for i in gets]
        print(f"Get {args.gets} from workers.")

        # get Van inverse matrix and decode, Van_inv: (m * n) * (m * n).
        Cenc = np.stack(Cenc)
        Van_inv = functions.get_Van_inv(codebook, args.gets, args)
        C = functions.decode(Cenc, codebook, Van_inv, args)

        MPI.Request.Waitall(reqC)

        for i in range(args.N):
            spent = comm.recv(source=i + 1, tag=100)
            args.computation_time[f"Worker {i + 1}"] = float(spent)

        print(args)

    else:
        straggler = comm.recv(source=0, tag=99)

        Aenc, Benc = np.zeros(A_enc_shape), np.zeros(B_enc_shape)
        reqA = comm.Irecv(Aenc, source=0, tag=3)
        reqB = comm.Irecv(Benc, source=0, tag=4)

        reqA.wait()
        reqB.wait()
        
        comm.Barrier()

        start = time.time()

        if straggler:
            print(f'Straggler in worker {rank}.')
            thread = threading.Thread(target=utils.straggle)
            thread.start()

        Cenc, reqC = [], []
        for j in range(args.d1):
            for k in range(args.d2):
                C = utils.compute(Aenc[k], Benc[j])
                if len(reqC) > 0:
                    reqC[-1].wait()
                reqC.append(comm.Isend(C, dest=0, tag=7 + j * args.d2 + k))

        spent = format(time.time() - start, ".5f")
        print(f"Worker {rank} of {args.N} computation time: {spent} seconds.")

        MPI.Request.Waitall(reqC)

        comm.send(spent, dest=0, tag=100)

    pass


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--experiment", type=int, default=2, help="experiment id, one from [1, 2, 3], use if parameter not contained in arguement should be changed")
    parser.add_argument("--id", type=str, default="test", help="experiment id")

    parser.add_argument("--dim1", type=int, default=1800, help="height of input matrix A")
    parser.add_argument("--dim2", type=int, default=1800, help="width of input matrix A, height of input matrix B")
    parser.add_argument("--dim3", type=int, default=1800, help="width of input matrix B")

    parser.add_argument("--m", type=int, default=3, help="A division")
    parser.add_argument("--n", type=int, default=3, help="B division")

    parser.add_argument("--d1", type=int, default=2, help="degree of the first polynomial")
    parser.add_argument("--d2", type=int, default=3, help="degree of the second polynomial")

    parser.add_argument("--straggle_prob", type=float, default=0.2, help="probabilitiy of straggle for each worker, 0 <= prob <= 1, 0 if there is no straggle")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    main(args)

    pass

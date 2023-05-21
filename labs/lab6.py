import time
import numpy as np
from icecream import ic
from mpi4py import MPI


def get_hilbert_matrix(n):
    return np.array([[1 / (i + 1 + j + 1 - 1) for i in range(n)] for j in range(n)])


alg="by_column"
N = 512
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
ROOT_RANK = COMM.Get_size() - 1
NC = N // ROOT_RANK
A = []
L = np.eye(N)
U = []
hist=[]
times = [0.0] * ROOT_RANK
if RANK == ROOT_RANK:
    A = get_hilbert_matrix(N)
    U = np.copy(A)

start_time = MPI.Wtime()
for j in range(N - 1):
    if RANK == ROOT_RANK:
        M = np.eye(N)

        mxi, mxj = j, j
        for k in range(j, N):
            #if U[k][j] > U[mxi][j]:
            #    mxi = k
            if U[j][k] > U[j][mxj]:
                mxj = k
        hist = [[mxi, mxj, j, j]] + hist
        U[:, [j, mxj]] = U[:, [mxj, j]]
        #ic('qqq', U)
        U[[j, mxi], :] = U[[mxi, j], :]

        for i in range(j + 1, ROOT_RANK):
            if j + 1 == ROOT_RANK - 1:
                break
            m = U[i][j] / U[j][j]
            #ic(i, U[i], j, N)
            for tg in range(NC):
                COMM.send((U[i], U[:, j], L[i], m), dest=i, tag=tg)
        for i in range(j + 1, ROOT_RANK):
            if j + 1 == ROOT_RANK - 1:
                break
            for tg in range(NC):
                U[i], L[i] = COMM.recv(source=i, tag=tg)
        #ic(j, U, L)
    else:
        for tg in range(NC):
            row, col, l_row, m = COMM.recv(source=ROOT_RANK, tag=tg)
            for k in range(j, N):
                # if j == 0:
                #     ic(RANK, k, row[k], col[k], m, row[k] - col[k] * m)
                row[k] = row[k] - col[k] * m
                if RANK*NC+tg > j:
                    l_row[j] = m
            COMM.send((row, l_row), dest=ROOT_RANK, tag=tg)
    end_time = MPI.Wtime()
    times[RANK - 1] = end_time - start_time

if RANK == ROOT_RANK:
    # ic(A)
    Alu = np.matmul(L, U)
    # ic(L)
    # ic(U)
    # ic(Alu)
    for idx in hist:
        Alu[:, [idx[3], idx[1]]] = Alu[:, [idx[1], idx[3]]]
        Alu[[idx[2], idx[0]], :] = Alu[[idx[0], idx[2]], :]

    # ic(Alu)
    err = abs(np.vectorize(abs)(np.matmul(L, U)).sum() - A.sum())
    ic(N, err, sum(times), alg, ROOT_RANK)
    with open("lab6.txt", "a") as file:
        file.write(f"{N}\t{ROOT_RANK}\t{err}\t{sum(times)}\t{alg}\n")
        ic('written')

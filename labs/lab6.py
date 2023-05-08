import numpy as np
from icecream import ic
from mpi4py import MPI


def get_hilbert_matrix(n):
    return np.array([[i + 1 + j + 1 - 1 for i in range(n)] for j in range(n)])


def multiply_to_num(row, number):
    return [a * number for a in row]


def divide_to_num(row, number):
    return [a / number for a in row]


def divide_column(matrix, ci, number):
    column = [row[ci] for row in matrix]
    divided_column = divide_to_num(column, number)
    for row in matrix:
        row[ci] = divided_column[ci]
    return matrix


def step(i, step_row, subtract_row):
    mi = subtract_row[i] / step_row[i]
    mul_step_row = multiply_to_num(step_row, mi)
    return (mi, [e - mul
                 for mul, e in zip(mul_step_row,
                                   subtract_row)])


COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
MATRIX = None
ROOT_RANK = 0
N = 0
if RANK == ROOT_RANK:
    N = 4
    MATRIX = get_hilbert_matrix(N)
    COMM.bcast(MATRIX, root=ROOT_RANK)
    COMM.bcast(N, root=ROOT_RANK)
    L = []
    U = []
    for i in range(N-1):
        for j, row in enumerate(MATRIX):
            print(('root send', j, i))
            COMM.isend((MATRIX[i], row), dest=j, tag=i)

COMM.Barrier()

if RANK != ROOT_RANK:
    MATRIX = COMM.bcast(MATRIX, root=ROOT_RANK)
    N = COMM.bcast(N, root=ROOT_RANK)
    for i in range(N-1):
        print(('they get', ROOT_RANK, i))
        step_row, sub_row = COMM.irecv(source=ROOT_RANK,
                                       tag=i).wait()
        mi, new_row = step(i, step_row, sub_row)

COMM.Barrier()

if RANK != ROOT_RANK:
    for i in range(N-1):
        print(('they send', ROOT_RANK, i))
        COMM.isend((mi, new_row), dest=ROOT_RANK)

COMM.Barrier()

if RANK == ROOT_RANK:
    for i in range(N-1):
        for j in range(N):
            print(('root get', j))
            mi, new_row = COMM.irecv(source=j).wait()
            L.append(mi)
            MATRIX[j] = new_row
    print(L)
    U = MATRIX
    print(U)




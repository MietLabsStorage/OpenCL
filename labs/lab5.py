import math
from mpi4py import MPI
import matplotlib.pyplot as plt
import icecream as ic


def rectangle(a, b, h, f):
    def s1(x):
        return f(a + (x - 1 / 2) * h)

    def s2(x):
        return 0

    def solve(sum12):
        return h * sum12

    return s1, s2, solve


def trapezium(a, b, h, f):
    first_part = (f(a) + f(b)) / 2

    def s1(x):
        return f(a + x * h)

    def s2(x):
        return 0

    def solve(sum12):
        return h * (first_part + sum12)

    return s1, s2, solve


def simpson(a, b, h, f):
    s3 = (f(a) + f(b)) / 2

    def inner(a, b, h, f):
        def left(x):
            return 2 * f(a + (x - 1 / 2) * h)

        def right(x):
            return f(a + x * h)

        def solve(sum12):
            return h / 3 * (s3 + sum12)

        return left, right, solve

    return inner(a, b, h, f)


def df(x):
    return 1 / (1 + x * x)


i_funcs = [[rectangle, 'Rect.'], [trapezium, 'Trap.'], [simpson, 'Simp.']]

for i_func in i_funcs:
    ns = []
    errs = []
    ts = []
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    N_PROC = COMM.Get_size() - 1
    for i in range(3, 10):
        n = 2 ** i

        REPEATS_IN_PROCESS = n // N_PROC
        times = [0.0] * N_PROC

        A = 0
        B = 1
        H = (B - A) / n

        func = i_func[0](A, B, H, df)
        chunk1_func = func[0]
        chunk2_func = func[1]
        reduce_func = func[2]

        start_time = MPI.Wtime()
        if RANK < N_PROC:
            for _ in range(0, REPEATS_IN_PROCESS):
                chunk1_res = chunk1_func(RANK)
                chunk2_res = chunk2_func(RANK)
                COMM.send(chunk1_res, dest=N_PROC)
                COMM.send(chunk2_res, dest=N_PROC)

        res = 0
        if RANK == N_PROC:
            d_sum = 0
            for _ in range(2 * n):
                d_sum += COMM.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
            res = 4 * reduce_func(d_sum)
        end_time = MPI.Wtime()
        elapsed_time = end_time - start_time
        times[RANK - 1] = elapsed_time

        if RANK == N_PROC:
            time = sum(times)
            err = math.fabs(math.pi - res)
            ic.ic(i_func[1], N_PROC, n, err, time)
            ns.append(n)
            errs.append(err)
            ts.append(time)

    if RANK == N_PROC:
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        ax1.plot(ns, errs)
        ax1.title.set_text(i_func[1] + ' Отклонение от pi')
        ax2.plot(ns, ts)
        ax2.title.set_text(i_func[1] + ' Время выполнения')
        plt.show(block=False)

plt.show(block=True)

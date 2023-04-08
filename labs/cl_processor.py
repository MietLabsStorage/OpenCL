from time import perf_counter_ns
import numpy as np
import pyopencl as cl


def create_context_and_queue():
    context = cl.create_some_context()
    queue = cl.CommandQueue(context)
    return context, queue


def build_program(context, kernel):
    return cl.Program(context, kernel).build()


def get_buffer_w(context, size):
    return cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=size)


def bind_to_buffer(context, host_buf):
    return cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=host_buf)


def execute_kernel(kernel, context, queue, sample, *args):
    out_array = get_buffer_w(context, sample.nbytes)
    timer_start = perf_counter_ns()
    kernel(queue, sample.shape, None, *args, out_array).wait()
    timer_stop = perf_counter_ns()
    output_array = np.empty_like(sample)
    cl.enqueue_copy(queue, output_array, out_array)
    return output_array, timer_stop - timer_start

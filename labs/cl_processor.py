from time import perf_counter_ns
import numpy as np
import pyopencl as cl

device = cl.get_platforms()[1].get_devices()[0]
print(device)


def create_context_and_queue():
    context = cl.Context(devices=[device])
    queue = cl.CommandQueue(context)
    return context, queue


def build_program(context, kernel, options=None):
    return cl.Program(context, kernel).build(options=options if options else [])


def get_buffer_w(context, size):
    return cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=size)


def bind_to_buffer(context, host_buf):
    return cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=host_buf)


def get_rw_pipe(ctx, element_size=1, capacity=1024):
    return cl.Pipe(ctx, cl.mem_flags.READ_WRITE, np.int32(element_size), np.int32(capacity), ())


def execute_kernel(kernel, context, queue, sample, *args):
    out_array = get_buffer_w(context, sample.nbytes)
    timer_start = perf_counter_ns()
    kernel(queue, sample.shape, None, *args, out_array).wait()
    timer_stop = perf_counter_ns()
    output_array = np.empty_like(sample)
    cl.enqueue_copy(queue, output_array, out_array)
    return output_array, timer_stop - timer_start


def execute_n_kernels(kernels, context, queue, sample, *args):
    out_array = get_buffer_w(context, sample.nbytes)
    timer_start = perf_counter_ns()
    for kernel in kernels:
        kernel(queue, sample.shape, None, *args, out_array).wait()
    timer_stop = perf_counter_ns()
    output_array = np.empty_like(sample)
    cl.enqueue_copy(queue, output_array, out_array)
    return output_array, timer_stop - timer_start


def execute_kernel_local(kernel, context, queue, sample, *args):
    out_array = get_buffer_w(context, sample.nbytes)
    local_array = cl.LocalMemory(out_array.size)
    timer_start = perf_counter_ns()
    kernel(queue, sample.shape, None, *args, local_array, out_array).wait()
    timer_stop = perf_counter_ns()
    output_array = np.empty_like(sample)
    cl.enqueue_copy(queue, output_array, out_array)
    return output_array, timer_stop - timer_start

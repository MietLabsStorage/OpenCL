import cl_processor as clp
import numpy as np
import cv2 as cv2
from time import perf_counter_ns

__colorsPalette = 3


def process_image(image_name, kernel_text, kernel_name, new_image_name, blur=0):
    image = cv2.imread(image_name, cv2.IMREAD_COLOR)
    processed, time = process_buffer(kernel_text, np.asarray(image), kernel_name, blur)
    cv2.imwrite(new_image_name, processed)
    return time


def process_monochrome_image(image_name, kernel_text, kernel_name, new_image_name, blur=0, is_local=False):
    image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    processed, time = process_monochrome_buffer(kernel_text, np.asarray(image), kernel_name, blur, is_local)
    cv2.imwrite(new_image_name, processed)
    return time


def to_grayscale_cv(image_name, new_image_name):
    timer_start = perf_counter_ns()
    image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(new_image_name, image)
    timer_stop = perf_counter_ns()
    return timer_stop - timer_start


def process_monochrome_buffer(kernel_text, data, kernel_name, blur, is_local):
    context, queue = clp.create_context_and_queue()
    prog = clp.build_program(context, kernel_text)
    cl_func = getattr(prog, kernel_name)
    rows_count = data.shape[0]
    d_row = data.flatten()
    processed, d_time = process_monochrome_rows(cl_func, d_row, context, queue, blur, data.shape[0], data.shape[1], is_local)
    processed_data = np.array(np.array_split(processed, rows_count))
    return processed_data, d_time


def process_buffer(kernel_text, data, kernel_name, *args, **kwargs):
    context, queue = clp.create_context_and_queue()
    build_options = kwargs['options'] if 'options' in kwargs else None
    prog = clp.build_program(context, kernel_text, build_options)

    if 'multiple' in kwargs and kwargs['multiple']:
        cl_func = [getattr(prog, f) for f in kernel_name]
    else:
        cl_func = getattr(prog, kernel_name)

    cl_func = getattr(prog, kernel_name)
    all_time = 0
    processed_rows = []
    for d_row in data:
        processed, d_time = process_rows(cl_func, d_row, context, queue, 'multiple' in kwargs)
        all_time += d_time
        processed_rows.append(processed)
    processed_data = np.array(processed_rows)
    return processed_data, all_time


def process_monochrome_rows(func, rows, context, queue, blur, row_len, col_len, is_local):
    row = clp.bind_to_buffer(context, rows)
    if is_local:
        processed_row, time = clp.execute_kernel_local(
            func,
            context,
            queue,
            rows,
            np.int32(row_len),
            np.int32(col_len),
            row,
            blur
        )
    else:
        processed_row, time = clp.execute_kernel(
            func,
            context,
            queue,
            rows,
            np.int32(row_len),
            np.int32(col_len),
            row,
            blur
        )
    return np.array(processed_row), time


def process_rows(func, rows, context, queue, multiple=False):
    flatten_rows = rows.flatten()
    flatten_rows_len = np.int32(flatten_rows.shape[0])
    row = clp.bind_to_buffer(context, flatten_rows)
    if multiple:
        processed_row, time = clp.execute_n_kernels(
            func,
            context,
            queue,
            flatten_rows,
            flatten_rows_len,
            row,
            np.int32(10))
    else:
        processed_row, time = clp.execute_kernel(
            func,
            context,
            queue,
            flatten_rows,
            flatten_rows_len,
            row,
            np.int32(10))
    return np.array(np.array_split(processed_row, len(processed_row) // __colorsPalette)), time
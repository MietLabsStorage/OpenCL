import cl_processor as clp
import numpy as np
import cv2 as cv2

__colorsPalette = 3


def process_image(image_name, kernel_text, kernel_name, new_image_name):
    image = cv2.imread(image_name, cv2.IMREAD_COLOR)
    processed, time = process_buffer(kernel_text, np.asarray(image), kernel_name)
    cv2.imwrite(new_image_name, processed)
    return time


def to_grayscale_cv(image_name, new_image_name):
    image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(new_image_name, image)


def process_buffer(kernel_text, data, kernel_name):
    context, queue = clp.create_context_and_queue()
    prog = clp.build_program(context, kernel_text)
    cl_func = getattr(prog, kernel_name)
    all_time = 0
    processed_rows = []
    for d_row in data:
        processed, d_time = process_rows(cl_func, d_row, context, queue)
        all_time += d_time
        processed_rows.append(processed)
    processed_data = np.array(processed_rows)
    return processed_data, all_time


def process_rows(func, rows, context, queue):
    flatten_rows = rows.flatten()
    flatten_rows_len = np.int32(flatten_rows.shape[0])
    row = clp.bind_to_buffer(context, flatten_rows)
    processed_row, time = clp.execute_kernel(
        func,
        context,
        queue,
        flatten_rows,
        flatten_rows_len,
        row)
    return np.array(np.array_split(processed_row, len(processed_row) // __colorsPalette)), time

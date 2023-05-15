from matplotlib import pyplot as plt

import cl_processor as clp
import numpy as np
from image_processor import process_buffer
from icecream import ic
import cv2 as cv2


box_blur = {
    "signature": """
            __kernel void box_blur(int rgb_row_len, __global char *rgb_row, int blur_size, __global char *blur_row)
            """,
    "defines": """
              #define RED 2
              #define GREEN 1
              #define BLUE 0
    
              #define CREATE_SUM_NAME(COLOR) CREATE_READABLE_SUM_NAME(COLOR)
              #define CREATE_READABLE_SUM_NAME(COLOR) sum_ ## COLOR
    
              #define GET_COLOR_INDEX(COLOR, PTR, PIXEL_INDEX) PTR[PIXEL_INDEX + COLOR]
              #define FOREACH_PIXEL(PIXEL_INDEX, BODY) \
                for(int PIXEL_INDEX = start; PIXEL_INDEX < end; PIXEL_INDEX += 3){ \
                  BODY; \
                }
    
              #define BLUR(COLOR) \
                FOREACH_PIXEL(PIXEL_INDEX, CREATE_SUM_NAME(COLOR) += GET_COLOR_INDEX(COLOR, rgb_row, PIXEL_INDEX);) \
                CREATE_SUM_NAME(COLOR) /= blur_size;
           """,
    "body": """
           int gid = get_global_id(0);
           int start = gid * 3 * blur_size;
           int end = start + 3 * blur_size;
           
           int CREATE_SUM_NAME(RED) = 0;
           int CREATE_SUM_NAME(GREEN) = 0;
           int CREATE_SUM_NAME(BLUE) = 0;

           if(start > rgb_row_len){
             return;
           }

           BLUR(RED)
           BLUR(GREEN)
           BLUR(BLUE)
           """,
    "kernel": "box_blur"
}

to_grayscale= {
    "signature": """
            __kernel void to_grayscale(int bgr_row_len, __global char *bgr_row, __global char *grayscale_row)
            """,
    "body": """           
            int gid = get_global_id(0);
            int bgr_chunk = gid * 3;
            if (bgr_chunk <= bgr_row_len) 
            {
                char b = bgr_row[bgr_chunk];
                char g = bgr_row[bgr_chunk + 1];
                char r = bgr_row[bgr_chunk + 2];
                char k = (char)(0.2126 * r + 0.7152 * g + 0.0722 * b);
                grayscale_row[bgr_chunk] = k;
                grayscale_row[bgr_chunk + 1] = k;
                grayscale_row[bgr_chunk + 2] = k;
            }            
           """,
    "kernel": "to_grayscale"
}


def by_buffers(image_name):
    image = cv2.imread(image_name, cv2.IMREAD_COLOR)
    data = np.asarray(image)
    box_blur_code = f"""
        {box_blur['defines']}
        {box_blur['signature']}
        {{
          {box_blur['body']}
          FOREACH_PIXEL(PIXEL_INDEX, GET_COLOR_INDEX(RED, blur_row, PIXEL_INDEX) = CREATE_SUM_NAME(RED))
          FOREACH_PIXEL(PIXEL_INDEX, GET_COLOR_INDEX(GREEN, blur_row, PIXEL_INDEX) = CREATE_SUM_NAME(GREEN))
          FOREACH_PIXEL(PIXEL_INDEX, GET_COLOR_INDEX(BLUE, blur_row, PIXEL_INDEX)  = CREATE_SUM_NAME(BLUE))
        }}
    """
    blur, box_blur_time = process_buffer(box_blur_code, data, box_blur['kernel'], np.int32(5))

    grayscale_code = f"""
        {to_grayscale['signature']}
         {{
           {to_grayscale['body']}
         }}
    """
    grayscale, grayscale_time = process_buffer(grayscale_code, blur, to_grayscale['kernel'])

    cv2.imwrite("by_buffer_"+image_name, grayscale)
    return box_blur_time + grayscale_time


def by___pipes(image_name):
    image = cv2.imread(image_name, cv2.IMREAD_COLOR)
    data = np.asarray(image)
    blur_and_grey_text = f"""
        #define SPIN_READ_PIPE(PIPE, DEST) if(read_pipe(PIPE, &DEST)){{mem_fence(CLK_GLOBAL_MEM_FENCE);}};
        #define SPIN_WRITE_PIPE(PIPE, DEST) if(write_pipe(PIPE, &DEST)){{mem_fence(CLK_GLOBAL_MEM_FENCE);}};
        {box_blur['defines']}
        __kernel void box_blur(int rgb_row_len,
                               __global char *rgb_row,
                               int blur_size,
                               write_only pipe char gray_pipe,
                               read_only pipe char blur_pipe,
                               __global char *blur_row)
        {{
          {box_blur['body']}
          char red = CREATE_SUM_NAME(RED);
          char green = CREATE_SUM_NAME(GREEN);
          char blue = CREATE_SUM_NAME(BLUE);
          FOREACH_PIXEL(PIXEL_INDEX,
                         SPIN_WRITE_PIPE(gray_pipe, red);
                         SPIN_WRITE_PIPE(gray_pipe, green);
                         SPIN_WRITE_PIPE(gray_pipe, blue);)
        }}
         __kernel void to_grayscale(int rgb_row_len,
                                      __global char *rgb_row,
                                      int blur_size,
                                      read_only pipe char gray_pipe,
                                      write_only pipe char blur_pipe,
                                      __global char *blur_row)
         {{
           int gid = get_global_id(0);
           char red, green, blue;

           int start = gid * 3 * blur_size;
           int end = start + 3 * blur_size;
           if(start > rgb_row_len){{
             return;
           }}
           FOREACH_PIXEL(PIXEL_INDEX,
                         SPIN_READ_PIPE(gray_pipe, red);
                         SPIN_READ_PIPE(gray_pipe, green);
                         SPIN_READ_PIPE(gray_pipe, blue);
                         char k = (char)(0.2126 * red + 0.7152 * green + 0.0722 * blue);
                         GET_COLOR_INDEX(RED, blur_row, PIXEL_INDEX)=k;
                         GET_COLOR_INDEX(GREEN, blur_row, PIXEL_INDEX)=k;
                         GET_COLOR_INDEX(BLUE, blur_row, PIXEL_INDEX)=k;)
        }}
        """
    ctx, queue = clp.create_context_and_queue()
    pipe = clp.get_rw_pipe(ctx, 1024, 66000)
    blurred_and_grey_data, blurred_and_grey_time = process_buffer(blur_and_grey_text,
                                                                  data,
                                                                  [box_blur["kernel"],
                                                                   to_grayscale["kernel"]],
                                                                  np.int32(5),
                                                                  pipe,
                                                                  pipe,
                                                                  multiple=True,
                                                                  options=['-cl-std=CL2.0'],
                                                                  ctx=ctx,
                                                                  queue=queue)
    cv2.imwrite("by_pipe_"+image_name, blurred_and_grey_data)
    return blurred_and_grey_time


szs = [32, 64, 128, 256, 512, 720]

by_bufs = []
by_pipes = []
for sz in szs:
    nm = f"image{sz}.jpg"
    by_bufs.append(by_buffers(nm))
    by_pipes.append(by___pipes(nm))
    ic(sz, by_bufs[-1], by_pipes[-1])

fig, ax = plt.subplots()
ax.plot(szs, by_bufs, label='by bufs')
ax.plot(szs, by_pipes, label='by pipes')
plt.legend()
plt.show()

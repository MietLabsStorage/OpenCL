import numpy as np
import image_processor as ip
from icecream import ic
import matplotlib.pyplot as plt


def box_blur(blur_size):
    time = ip.process_monochrome_image(
        "Helmet.jpg",
        """
            __kernel void box_blur(int bgr_rows_count, int bgr_cols_count, __global char *bgr_row, int blur_size, __global char *blur_row)
            {
                int gid = get_global_id(0);
                int x = gid % bgr_cols_count;
                int y = gid / bgr_rows_count;
                if (gid <= bgr_rows_count*bgr_cols_count) 
                {
                    int radius = blur_size / 2;
                    int r = 0;
                    int c = 0;
                    for (int ir = y-radius; ir <= y+radius; ir++)
                        for (int ic = x-radius; ic <= x+radius; ic++)
                        {                            
                            if (ir < 0 || ir >= bgr_rows_count || ic < 0 || ic >= bgr_cols_count)
                            {
                                
                            }
                            else 
                            {
                                r += bgr_row[ir * bgr_cols_count + ic];
                                c++;
                            }
                        }
                    r /= c;

                    blur_row[gid] = r;
                }
            }
        """,
        "box_blur",
        f"box_blur{blur_size}.jpg",
        np.int32(blur_size))
    return time


def box_blur_const(blur_size):
    time = ip.process_monochrome_image(
        "Helmet.jpg",
        """
            __kernel void box_blur(int bgr_rows_count, int bgr_cols_count, __constant char *bgr_row, int blur_size, __global char *blur_row)
            {
                int gid = get_global_id(0);
                int x = gid % bgr_cols_count;
                int y = gid / bgr_rows_count;
                if (gid <= bgr_rows_count*bgr_cols_count) 
                {
                    int radius = blur_size / 2;
                    int r = 0;
                    int c = 0;
                    for (int ir = y-radius; ir <= y+radius; ir++)
                        for (int ic = x-radius; ic <= x+radius; ic++)
                        {                            
                            if (ir < 0 || ir >= bgr_rows_count || ic < 0 || ic >= bgr_cols_count)
                            {

                            }
                            else 
                            {
                                r += bgr_row[ir * bgr_cols_count + ic];
                                c++;
                            }
                        }
                    r /= c;

                    blur_row[gid] = r;
                }
            }
        """,
        "box_blur",
        f"box_blur_c{blur_size}.jpg",
        np.int32(blur_size))
    return time


def box_blur_local(blur_size):
    time = ip.process_monochrome_image(
        "Helmet.jpg",
        """
            __kernel void box_blur(int bgr_rows_count, int bgr_cols_count, __global char *bgr_row, int blur_size, __local char *blur_row, __global char *output)
            {
                int gid = get_global_id(0);
                int x = gid % bgr_cols_count;
                int y = gid / bgr_rows_count;
                if (gid <= bgr_rows_count*bgr_cols_count) 
                {
                    int radius = blur_size / 2;
                    int r = 0;
                    int c = 0;
                    for (int ir = y-radius; ir <= y+radius; ir++)
                        for (int ic = x-radius; ic <= x+radius; ic++)
                        {                            
                            if (ir < 0 || ir >= bgr_rows_count || ic < 0 || ic >= bgr_cols_count)
                            {

                            }
                            else 
                            {
                                r += bgr_row[ir * bgr_cols_count + ic];
                                c++;
                            }
                        }
                    r /= c;
                    
                    blur_row[gid] = r;
                }
                
                barrier(CLK_LOCAL_MEM_FENCE);
                output[gid] = blur_row[gid];
            }
        """,
        "box_blur",
        f"box_blur_l{blur_size}.jpg",
        np.int32(blur_size),
        True)
    return time


x1 = []
y1 = []
for i in range(2, 5 + 1):
    sz = 2**i
    t = box_blur(sz)
    ic(i, sz, t)
    x1.append(sz)
    y1.append(t)

x2 = []
y2 = []
for i in range(2, 12 + 1):
    sz = 2**i
    t = box_blur_local(sz)
    ic(i, sz, t)
    x2.append(sz)
    y2.append(t)

x3 = []
y3 = []
for i in range(2, 12 + 1):
    sz = 2 ** i
    t = box_blur_const(sz)
    ic(i, sz, t)
    x3.append(sz)
    y3.append(t)

fig, ax = plt.subplots()
ax.plot(x1, y1, label='Global')
ax.plot(x2, y2, label='Local')
ax.plot(x3, y3, label='Constant')
plt.legend()
plt.show()

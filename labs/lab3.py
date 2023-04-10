import numpy as np
import image_processor as ip
from icecream import ic
import matplotlib.pyplot as plt


def box_blur(blur_size):
    time = ip.process_monochrome_image(
        "image.jpg",
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
                    if (r > 200)
                        r = 127;
                    blur_row[gid] = r;
                }
            }
        """,
        "box_blur",
        f"box_blur{blur_size}.jpg",
        np.int32(blur_size))
    return time


x = []
y = []
for i in range(2, 9 + 1):
    sz = 2**i
    t = box_blur(sz)
    ic(i, sz, t)
    x.append(sz)
    y.append(t)
fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()

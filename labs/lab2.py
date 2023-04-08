import image_processor as ip


def to_negative():
    time = ip.process_image(
        "image.jpg",
        """
            __kernel void to_negative(int bgr_row_len, __global char *bgr_row, __global char *negative_row)
            {
                int gid = get_global_id(0);
                int bgr_chunk = gid * 3;
                int max_intensity = 255;
                if (bgr_chunk <= bgr_row_len) 
                {
                    negative_row[bgr_chunk] = max_intensity - bgr_row[bgr_chunk];
                    negative_row[bgr_chunk + 1] = max_intensity - bgr_row[bgr_chunk + 1];
                    negative_row[bgr_chunk + 2] = max_intensity - bgr_row[bgr_chunk + 2];
                }
            }
        """,
        "to_negative",
        "negative.jpg")
    return time


def to_grayscale_cl():
    time = ip.process_image(
        "image.jpg",
        """
            __kernel void to_grayscale(int bgr_row_len, __global char *bgr_row, __global char *grayscale_row)
            {
                int gid = get_global_id(0);
                int bgr_chunk = gid * 3;
                if (bgr_chunk <= bgr_row_len) 
                {
                    double b = bgr_row[bgr_chunk];
                    double g = bgr_row[bgr_chunk + 1];
                    double r = bgr_row[bgr_chunk + 2];
                    char k = (char)(0.2126 * r + 0.7152 * g + 0.0722 * b);
                    grayscale_row[bgr_chunk] = k;
                    grayscale_row[bgr_chunk + 1] = k;
                    grayscale_row[bgr_chunk + 2] = k;
                }
            }
        """,
        "to_grayscale",
        "grayscale_cl.jpg")
    return time


to_negative()
ip.to_grayscale_cv("image.jpg", "grayscale_cv.jpg")
to_grayscale_cl()
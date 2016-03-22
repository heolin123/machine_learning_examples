#!/usr/bin/env python2
import PIL.Image # python-imaging
import PIL.ImageStat # python-imaging
import Xlib.display # python-xlib
import numpy as np
import sys

def read_image(image_path):
    return np.asarray(PIL.Image.open(image_path))

def get_histogram(image_array):
    result = np.array([0.0 for _ in xrange(256)])
    for x in image_array:
        result[x] += 1.0
    return result

def get_histograms(image):
    w, h, d = image.shape
    image_flat = image.reshape(w * h, d)
    histogram_r = get_histogram(image_flat[:, 0])
    histogram_g = get_histogram(image_flat[:, 1])
    histogram_b = get_histogram(image_flat[:, 2])

    histogram_r_norm = (histogram_r / float(sum(histogram_r))).reshape(256, 1)
    histogram_g_norm = (histogram_g / float(sum(histogram_g))).reshape(256, 1)
    histogram_b_norm = (histogram_b / float(sum(histogram_b))).reshape(256, 1)

    return np.concatenate((histogram_r_norm, histogram_g_norm, histogram_b_norm), axis=0)

def main():
    image = read_image(sys.argv[1])
    print get_histograms(image)


if __name__ == "__main__":
    main()

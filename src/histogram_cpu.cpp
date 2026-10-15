#include "histogram_cpu.h"

void computeHistogramCPU(const unsigned char* image, int width, int height, int histogram[256]) {
    for (int i = 0; i < 256; i++) {
        histogram[i] = 0;
    }

    int size = width * height;
    for (int i = 0; i < size; i++) {
        histogram[image[i]]++;
    }
}
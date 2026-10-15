#include "equalization_cpu.h"
#include "histogram_cpu.h"
#include <cmath>

void equalizeHistogramCPU(const unsigned char* input, unsigned char* output, int width, int height) {
    int histogram[256];
    computeHistogramCPU(input, width, height, histogram);

    int totalPixels = width * height;

    float cdf[256];
    cdf[0] = static_cast<float>(histogram[0]);

    for (int i = 1; i < 256; i++) {
        cdf[i] = cdf[i - 1] + histogram[i];
    }

    float cdfMin = 0.0f;
    for (int i = 0; i < 256; i++) {
        if (cdf[i] > 0) {
            cdfMin = cdf[i];
            break;
        }
    }

    unsigned char lut[256];
    for (int i = 0; i < 256; i++) {
        float value = (cdf[i] - cdfMin) / (totalPixels - cdfMin);
        value = value * 255.0f;

        if (value < 0.0f) value = 0.0f;
        if (value > 255.0f) value = 255.0f;

        lut[i] = static_cast<unsigned char>(std::round(value));
    }

    for (int i = 0; i < totalPixels; i++) {
        output[i] = lut[input[i]];
    }
}
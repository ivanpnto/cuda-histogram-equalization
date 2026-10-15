#ifndef HISTOGRAM_GPU_CUH
#define HISTOGRAM_GPU_CUH

// -------------------------------------------------
// Version 2: Baseline GPU histogram
// Single global histogram updated with atomicAdd
// -------------------------------------------------
void computeHistogramGPUBaseline(const unsigned char* image, int width, int height, int histogram[256]);
float computeHistogramGPUBaselineTimed(const unsigned char* image, int width, int height, int histogram[256]);

// -------------------------------------------------
// Version 3: Optimized GPU histogram
// Per-block local histograms in shared memory
// -------------------------------------------------
float computeHistogramGPUOptimizedTimed(const unsigned char* image, int width, int height, int histogram[256]);

float computeHistogramGPUOptimizedTimedWithBlockSize(
    const unsigned char* image,
    int width,
    int height,
    int histogram[256],
    int threadsPerBlock
);

float computeHistogramGPUSharedSkipZeroTimedWithBlockSize(
    const unsigned char* image,
    int width,
    int height,
    int histogram[256],
    int threadsPerBlock
);

float computeHistogramGPUSingleThreadMergeTimedWithBlockSize(
    const unsigned char* image,
    int width,
    int height,
    int histogram[256],
    int threadsPerBlock
);

float computeHistogramGPUReducedGridTimedWithBlockSize(
    const unsigned char* image,
    int width,
    int height,
    int histogram[256],
    int threadsPerBlock,
    int blocksPerSM
);

float computeHistogramGPUBinReductionTimedWithBlockSize(
    const unsigned char* image,
    int width,
    int height,
    int histogram[256],
    int threadsPerBlock
);

// -------------------------------------------------
// Common GPU stage: LUT application
// -------------------------------------------------
void applyLUTGPU(const unsigned char* input, unsigned char* output, int width, int height, const unsigned char lut[256]);
float applyLUTGPUTimed(const unsigned char* input, unsigned char* output, int width, int height, const unsigned char lut[256]);

#endif
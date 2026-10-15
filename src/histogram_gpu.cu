#include "histogram_gpu.cuh"
#include <cuda_runtime.h>
#include <iostream>

// -------------------------------------------------
// Version 2: Baseline GPU histogram kernel
// Each thread updates a single global histogram
// using atomicAdd. This version is simple and
// correct, but suffers from contention.
// -------------------------------------------------
__global__ void histogramKernelBaseline(const unsigned char* image, int size, int* histogram) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned char pixel = image[idx];
        atomicAdd(&histogram[pixel], 1);
    }
}

// -------------------------------------------------
// Version 3: Optimized GPU histogram kernel
// Each block builds a local histogram in shared
// memory and only performs a final accumulation
// into the global histogram.
// -------------------------------------------------
__global__ void histogramKernelOptimized(const unsigned char* image, int size, int* histogram) {
    __shared__ unsigned int localHist[256];

    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    // Initialize all 256 bins, even when blockDim.x < 256
    for (int bin = tid; bin < 256; bin += blockDim.x) {
        localHist[bin] = 0;
    }
    __syncthreads();

    // Accumulate local histogram in shared memory
    for (int i = globalIdx; i < size; i += stride) {
        unsigned char pixel = image[i];
        atomicAdd(&localHist[pixel], 1);
    }
    __syncthreads();

    // Accumulate local block histogram into global histogram
    for (int bin = tid; bin < 256; bin += blockDim.x) {
        atomicAdd(&histogram[bin], localHist[bin]);
    }
}

// -------------------------------------------------
// Extra optimization attempt 1:
// Shared-memory histogram with zero-bin skipping
// during the local-to-global merge.
// -------------------------------------------------
__global__ void histogramKernelSharedSkipZero(
    const unsigned char* image,
    int size,
    int* histogram
) {
    __shared__ unsigned int localHist[256];

    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    // Initialize all 256 local histogram bins
    for (int bin = tid; bin < 256; bin += blockDim.x) {
        localHist[bin] = 0;
    }

    __syncthreads();

    // Build local histogram in shared memory
    for (int i = globalIdx; i < size; i += stride) {
        unsigned char pixel = image[i];
        atomicAdd(&localHist[pixel], 1);
    }

    __syncthreads();

    // Merge local histogram into global histogram.
    // Difference from the original optimized kernel:
    // skip the global atomicAdd if the local bin is zero.
    for (int bin = tid; bin < 256; bin += blockDim.x) {
        unsigned int value = localHist[bin];

        if (value > 0) {
            atomicAdd(&histogram[bin], value);
        }
    }
}

// -------------------------------------------------
// Extra optimization attempt 2:
// Shared-memory histogram with single-thread merge.
// Only thread 0 of each block merges the 256 local bins
// into the global histogram.
// -------------------------------------------------
__global__ void histogramKernelSingleThreadMerge(
    const unsigned char* image,
    int size,
    int* histogram
) {
    __shared__ unsigned int localHist[256];

    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    // Initialize local histogram
    for (int bin = tid; bin < 256; bin += blockDim.x) {
        localHist[bin] = 0;
    }

    __syncthreads();

    // Build local histogram in shared memory
    for (int i = globalIdx; i < size; i += stride) {
        unsigned char pixel = image[i];
        atomicAdd(&localHist[pixel], 1);
    }

    __syncthreads();

    // Single-thread local-to-global merge.
    // Only one thread per block performs all 256 global atomic updates.
    if (tid == 0) {
        for (int bin = 0; bin < 256; bin++) {
            unsigned int value = localHist[bin];

            if (value > 0) {
                atomicAdd(&histogram[bin], value);
            }
        }
    }
}

// -------------------------------------------------
// Extra optimization attempt 4:
// Bin-oriented histogram reduction.
// Each block computes one histogram bin.
// This version avoids atomicAdd, but each bin scans
// the whole image.
// -------------------------------------------------
__global__ void histogramKernelBinReduction(
    const unsigned char* image,
    int size,
    int* histogram
) {
    __shared__ unsigned int partialCounts[1024];

    int tid = threadIdx.x;
    int bin = blockIdx.x;

    unsigned int localCount = 0;

    // Each block is responsible for one bin.
    // Threads inside the block scan the image and count
    // how many pixels are equal to that bin.
    for (int i = tid; i < size; i += blockDim.x) {
        if (image[i] == static_cast<unsigned char>(bin)) {
            localCount++;
        }
    }

    partialCounts[tid] = localCount;

    __syncthreads();

    // Parallel reduction inside the block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partialCounts[tid] += partialCounts[tid + stride];
        }

        __syncthreads();
    }

    // Thread 0 writes the final count for this bin.
    // No atomicAdd is needed because each bin is written
    // by exactly one block.
    if (tid == 0) {
        histogram[bin] = static_cast<int>(partialCounts[0]);
    }
}

// -------------------------------------------------
// Common GPU stage: LUT application kernel
// Each thread transforms one pixel independently.
// -------------------------------------------------
__global__ void applyLUTKernel(const unsigned char* input, unsigned char* output, int size, const unsigned char* lut) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = lut[input[idx]];
    }
}

// -------------------------------------------------
// Wrapper for baseline histogram (non-timed)
// -------------------------------------------------
void computeHistogramGPUBaseline(const unsigned char* image, int width, int height, int histogram[256]) {
    computeHistogramGPUBaselineTimed(image, width, height, histogram);
}

// -------------------------------------------------
// Wrapper for LUT application (non-timed)
// -------------------------------------------------
void applyLUTGPU(const unsigned char* input, unsigned char* output, int width, int height, const unsigned char lut[256]) {
    applyLUTGPUTimed(input, output, width, height, lut);
}

// -------------------------------------------------
// Version 2: Baseline GPU histogram timed
// -------------------------------------------------
float computeHistogramGPUBaselineTimed(const unsigned char* image, int width, int height, int histogram[256]) {
    const int size = width * height;

    unsigned char* d_image = nullptr;
    int* d_histogram = nullptr;
    cudaError_t err;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    err = cudaMalloc(&d_image, size * sizeof(unsigned char));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc d_image failed: " << cudaGetErrorString(err) << "\n";
        return -1.0f;
    }

    err = cudaMalloc(&d_histogram, 256 * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc d_histogram failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_image);
        return -1.0f;
    }

    cudaEventRecord(start);

    err = cudaMemcpy(d_image, image, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy H2D failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_image);
        cudaFree(d_histogram);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0f;
    }

    err = cudaMemset(d_histogram, 0, 256 * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "cudaMemset failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_image);
        cudaFree(d_histogram);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0f;
    }

    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    histogramKernelBaseline<<<blocks, threadsPerBlock>>>(d_image, size, d_histogram);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Baseline kernel launch failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_image);
        cudaFree(d_histogram);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0f;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Baseline kernel execution failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_image);
        cudaFree(d_histogram);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0f;
    }

    err = cudaMemcpy(histogram, d_histogram, 256 * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy D2H failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_image);
        cudaFree(d_histogram);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0f;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_image);
    cudaFree(d_histogram);

    return ms;
}

// -------------------------------------------------
// Version 3: Optimized GPU histogram timed
// Convenience wrapper with a fixed block size.
// The main benchmark uses the configurable version.
// -------------------------------------------------
float computeHistogramGPUOptimizedTimedWithBlockSize(
    const unsigned char* image,
    int width,
    int height,
    int histogram[256],
    int threadsPerBlock
) {
    const int size = width * height;

    if (threadsPerBlock <= 0 || threadsPerBlock > 1024) {
        std::cerr << "Invalid threadsPerBlock value: " << threadsPerBlock << "\n";
        return -1.0f;
    }

    unsigned char* d_image = nullptr;
    int* d_histogram = nullptr;
    cudaError_t err;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    err = cudaMalloc(&d_image, size * sizeof(unsigned char));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc d_image failed: " << cudaGetErrorString(err) << "\n";
        return -1.0f;
    }

    err = cudaMalloc(&d_histogram, 256 * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc d_histogram failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_image);
        return -1.0f;
    }

    cudaEventRecord(start);

    err = cudaMemcpy(d_image, image, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy H2D failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_image);
        cudaFree(d_histogram);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0f;
    }

    err = cudaMemset(d_histogram, 0, 256 * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "cudaMemset failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_image);
        cudaFree(d_histogram);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0f;
    }

    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    histogramKernelOptimized<<<blocks, threadsPerBlock>>>(d_image, size, d_histogram);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Optimized kernel launch failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_image);
        cudaFree(d_histogram);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0f;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Optimized kernel execution failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_image);
        cudaFree(d_histogram);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0f;
    }

    err = cudaMemcpy(histogram, d_histogram, 256 * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy D2H failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_image);
        cudaFree(d_histogram);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0f;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_image);
    cudaFree(d_histogram);

    return ms;
}

// -------------------------------------------------
// Version 3: Optimized GPU histogram timed
// Default final configuration: 256 threads per block.
// -------------------------------------------------
float computeHistogramGPUOptimizedTimed(const unsigned char* image, int width, int height, int histogram[256]) {
    return computeHistogramGPUOptimizedTimedWithBlockSize(image, width, height, histogram, 1024);
}

// -------------------------------------------------
// Extra optimization 1: shared-memory skip-zero histogram timed
// Configurable block size version.
// -------------------------------------------------
float computeHistogramGPUSharedSkipZeroTimedWithBlockSize(
    const unsigned char* image,
    int width,
    int height,
    int histogram[256],
    int threadsPerBlock
) {
    const int size = width * height;

    if (threadsPerBlock <= 0 || threadsPerBlock > 1024) {
        std::cerr << "Invalid threadsPerBlock value: " << threadsPerBlock << "\n";
        return -1.0f;
    }

    unsigned char* d_image = nullptr;
    int* d_histogram = nullptr;
    cudaError_t err;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    err = cudaMalloc(&d_image, size * sizeof(unsigned char));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc d_image failed: " << cudaGetErrorString(err) << "\n";
        return -1.0f;
    }

    err = cudaMalloc(&d_histogram, 256 * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc d_histogram failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_image);
        return -1.0f;
    }

    cudaEventRecord(start);

    err = cudaMemcpy(d_image, image, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy H2D failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_image);
        cudaFree(d_histogram);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0f;
    }

    err = cudaMemset(d_histogram, 0, 256 * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "cudaMemset failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_image);
        cudaFree(d_histogram);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0f;
    }

    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    histogramKernelSharedSkipZero<<<blocks, threadsPerBlock>>>(d_image, size, d_histogram);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Skip-zero kernel launch failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_image);
        cudaFree(d_histogram);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0f;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Skip-zero kernel execution failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_image);
        cudaFree(d_histogram);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0f;
    }

    err = cudaMemcpy(histogram, d_histogram, 256 * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy D2H failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_image);
        cudaFree(d_histogram);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0f;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_image);
    cudaFree(d_histogram);

    return ms;
}

// -------------------------------------------------
// Extra optimization 2: single-thread merge histogram timed
// Configurable block size version.
// -------------------------------------------------
float computeHistogramGPUSingleThreadMergeTimedWithBlockSize(
    const unsigned char* image,
    int width,
    int height,
    int histogram[256],
    int threadsPerBlock
) {
    const int size = width * height;

    if (threadsPerBlock <= 0 || threadsPerBlock > 1024) {
        std::cerr << "Invalid threadsPerBlock value: " << threadsPerBlock << "\n";
        return -1.0f;
    }

    unsigned char* d_image = nullptr;
    int* d_histogram = nullptr;
    cudaError_t err;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    err = cudaMalloc(&d_image, size * sizeof(unsigned char));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc d_image failed: " << cudaGetErrorString(err) << "\n";
        return -1.0f;
    }

    err = cudaMalloc(&d_histogram, 256 * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc d_histogram failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_image);
        return -1.0f;
    }

    cudaEventRecord(start);

    err = cudaMemcpy(d_image, image, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy H2D failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_image);
        cudaFree(d_histogram);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0f;
    }

    err = cudaMemset(d_histogram, 0, 256 * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "cudaMemset failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_image);
        cudaFree(d_histogram);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0f;
    }

    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    histogramKernelSingleThreadMerge<<<blocks, threadsPerBlock>>>(d_image, size, d_histogram);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Single-thread merge kernel launch failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_image);
        cudaFree(d_histogram);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0f;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Single-thread merge kernel execution failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_image);
        cudaFree(d_histogram);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0f;
    }

    err = cudaMemcpy(histogram, d_histogram, 256 * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy D2H failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_image);
        cudaFree(d_histogram);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0f;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_image);
    cudaFree(d_histogram);

    return ms;
}

// -------------------------------------------------
// Extra optimization 3: reduced-grid shared-memory histogram timed
// The number of blocks is limited to SMs * blocksPerSM.
// Each thread processes more pixels using the existing grid-stride loop.
// -------------------------------------------------
float computeHistogramGPUReducedGridTimedWithBlockSize(
    const unsigned char* image,
    int width,
    int height,
    int histogram[256],
    int threadsPerBlock,
    int blocksPerSM
) {
    const int size = width * height;

    if (threadsPerBlock <= 0 || threadsPerBlock > 1024) {
        std::cerr << "Invalid threadsPerBlock value: " << threadsPerBlock << "\n";
        return -1.0f;
    }

    if (blocksPerSM <= 0) {
        std::cerr << "Invalid blocksPerSM value: " << blocksPerSM << "\n";
        return -1.0f;
    }

    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDevice failed: " << cudaGetErrorString(err) << "\n";
        return -1.0f;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties failed: " << cudaGetErrorString(err) << "\n";
        return -1.0f;
    }

    int smCount = prop.multiProcessorCount;

    int normalBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    int reducedBlocks = smCount * blocksPerSM;

    int blocks = reducedBlocks;

    if (blocks > normalBlocks) {
        blocks = normalBlocks;
    }

    if (blocks < 1) {
        blocks = 1;
    }

    unsigned char* d_image = nullptr;
    int* d_histogram = nullptr;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    err = cudaMalloc(&d_image, size * sizeof(unsigned char));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc d_image failed: " << cudaGetErrorString(err) << "\n";
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0f;
    }

    err = cudaMalloc(&d_histogram, 256 * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc d_histogram failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_image);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0f;
    }

    cudaEventRecord(start);

    err = cudaMemcpy(d_image, image, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy H2D failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_image);
        cudaFree(d_histogram);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0f;
    }

    err = cudaMemset(d_histogram, 0, 256 * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "cudaMemset failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_image);
        cudaFree(d_histogram);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0f;
    }

    histogramKernelOptimized<<<blocks, threadsPerBlock>>>(d_image, size, d_histogram);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Reduced-grid kernel launch failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_image);
        cudaFree(d_histogram);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0f;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Reduced-grid kernel execution failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_image);
        cudaFree(d_histogram);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0f;
    }

    err = cudaMemcpy(histogram, d_histogram, 256 * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy D2H failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_image);
        cudaFree(d_histogram);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0f;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_image);
    cudaFree(d_histogram);

    return ms;
}

// -------------------------------------------------
// Extra optimization 4: bin-oriented reduction histogram timed
// One block computes one histogram bin.
// This version avoids atomicAdd, but scans the image once per bin.
// -------------------------------------------------
float computeHistogramGPUBinReductionTimedWithBlockSize(
    const unsigned char* image,
    int width,
    int height,
    int histogram[256],
    int threadsPerBlock
) {
    const int size = width * height;

    if (threadsPerBlock != 128 &&
        threadsPerBlock != 256 &&
        threadsPerBlock != 512 &&
        threadsPerBlock != 1024) {
        std::cerr << "Invalid threadsPerBlock value for bin reduction: "
                  << threadsPerBlock << "\n";
        std::cerr << "Use 128, 256, 512 or 1024.\n";
        return -1.0f;
    }

    unsigned char* d_image = nullptr;
    int* d_histogram = nullptr;
    cudaError_t err;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    err = cudaMalloc(&d_image, size * sizeof(unsigned char));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc d_image failed: " << cudaGetErrorString(err) << "\n";
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0f;
    }

    err = cudaMalloc(&d_histogram, 256 * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc d_histogram failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_image);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0f;
    }

    cudaEventRecord(start);

    err = cudaMemcpy(d_image, image, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy H2D failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_image);
        cudaFree(d_histogram);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0f;
    }

    err = cudaMemset(d_histogram, 0, 256 * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "cudaMemset failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_image);
        cudaFree(d_histogram);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0f;
    }

    // One block per histogram bin
    int blocks = 256;

    histogramKernelBinReduction<<<blocks, threadsPerBlock>>>(
        d_image,
        size,
        d_histogram
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Bin-reduction kernel launch failed: "
                  << cudaGetErrorString(err) << "\n";
        cudaFree(d_image);
        cudaFree(d_histogram);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0f;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Bin-reduction kernel execution failed: "
                  << cudaGetErrorString(err) << "\n";
        cudaFree(d_image);
        cudaFree(d_histogram);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0f;
    }

    err = cudaMemcpy(histogram, d_histogram, 256 * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy D2H failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_image);
        cudaFree(d_histogram);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0f;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_image);
    cudaFree(d_histogram);

    return ms;
}

// -------------------------------------------------
// Common GPU stage: LUT application timed
// -------------------------------------------------
float applyLUTGPUTimed(const unsigned char* input, unsigned char* output, int width, int height, const unsigned char lut[256]) {
    const int size = width * height;

    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;
    unsigned char* d_lut = nullptr;
    cudaError_t err;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    err = cudaMalloc(&d_input, size * sizeof(unsigned char));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc d_input failed: " << cudaGetErrorString(err) << "\n";
        return -1.0f;
    }

    err = cudaMalloc(&d_output, size * sizeof(unsigned char));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc d_output failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_input);
        return -1.0f;
    }

    err = cudaMalloc(&d_lut, 256 * sizeof(unsigned char));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc d_lut failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_input);
        cudaFree(d_output);
        return -1.0f;
    }

    cudaEventRecord(start);

    err = cudaMemcpy(d_input, input, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy input H2D failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_lut);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0f;
    }

    err = cudaMemcpy(d_lut, lut, 256 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy LUT H2D failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_lut);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0f;
    }

    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    applyLUTKernel<<<blocks, threadsPerBlock>>>(d_input, d_output, size, d_lut);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "applyLUT kernel launch failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_lut);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0f;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "applyLUT kernel execution failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_lut);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0f;
    }

    err = cudaMemcpy(output, d_output, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy output D2H failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_lut);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0f;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_lut);

    return ms;
}
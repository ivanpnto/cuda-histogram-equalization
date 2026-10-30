#include <iostream>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include "image_io.h"
#include "equalization_cpu.h"
#include "histogram_gpu.cuh"

struct Stats {
    double avg;
    double min;
    double max;
};

Stats computeStats(const std::vector<double>& values) {
    Stats s{};
    if (values.empty()) {
        s.avg = s.min = s.max = 0.0;
        return s;
    }

    s.avg = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    s.min = *std::min_element(values.begin(), values.end());
    s.max = *std::max_element(values.begin(), values.end());
    return s;
}

void buildLUTFromHistogram(const int histogram[256], int totalPixels, unsigned char lut[256]) {
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

    for (int i = 0; i < 256; i++) {
        float value = (cdf[i] - cdfMin) / (totalPixels - cdfMin);
        value *= 255.0f;

        if (value < 0.0f) value = 0.0f;
        if (value > 255.0f) value = 255.0f;

        lut[i] = static_cast<unsigned char>(std::round(value));
    }
}

int main(int argc, char* argv[]) {
    const char* inputPath = "../data/input/Large.png";
    const char* outputCPUPath = "../data/output/equalized_cpu.png";
    const char* outputGPUPath = "../data/output/equalized_gpu.png";
    const int repetitions = 10;
    int optimizedThreadsPerBlock = 1024;
    int reducedGridBlocksPerSM = 16;

    if (argc >= 2) {
        inputPath = argv[1];
    }
    if (argc >= 3) {
        optimizedThreadsPerBlock = std::atoi(argv[2]);
    }

    // Optional third parameter for the reduced-grid experiment.
    // It represents how many blocks are launched per GPU SM.
    // Example: with 36 SMs and blocksPerSM = 8, the kernel launches 288 blocks.
    if (argc >= 4) {
        reducedGridBlocksPerSM = std::atoi(argv[3]);
    }

    if (optimizedThreadsPerBlock != 128 &&
        optimizedThreadsPerBlock != 256 &&
        optimizedThreadsPerBlock != 512 &&
        optimizedThreadsPerBlock != 1024) {
        std::cerr << "Invalid optimized histogram block size: "
                << optimizedThreadsPerBlock << "\n";
        std::cerr << "Allowed values for this experiment are: 128, 256, 512, 1024\n";
        return 1;
    }

    if (reducedGridBlocksPerSM <= 0) {
        std::cerr << "Invalid reduced-grid blocks per SM: "
                  << reducedGridBlocksPerSM << "\n";
        std::cerr << "Use a positive integer, for example: 4, 8, 16, 32 or 64.\n";
        return 1;
    }

    unsigned char* image = nullptr;
    int width = 0, height = 0;

    if (!loadGrayscaleImage(inputPath, image, width, height)) {
        return 1;
    }

    std::cout << "Input image: " << inputPath << "\n";
    std::cout << "Image loaded successfully.\n";
    std::cout << "Resolution: " << width << "x" << height << "\n";
    std::cout << "Optimized histogram block size: " << optimizedThreadsPerBlock << "\n";
    std::cout << "Reduced-grid blocks per SM: " << reducedGridBlocksPerSM << "\n";

    int totalPixels = width * height;

    // -------------------------------------------------
    // Version 1: Sequential CPU implementation
    // -------------------------------------------------
    unsigned char* outputCPU = new unsigned char[totalPixels];
    equalizeHistogramCPU(image, outputCPU, width, height);

    if (!saveGrayscaleImage(outputCPUPath, outputCPU, width, height)) {
        freeImage(image);
        delete[] outputCPU;
        return 1;
    }

    std::cout << "Equalized CPU image saved successfully to: " << outputCPUPath << "\n";

    // -------------------------------------------------
    // Version 2: Baseline GPU histogram
    // -------------------------------------------------
    int histogramGPUBaseline[256] = {};
    computeHistogramGPUBaseline(image, width, height, histogramGPUBaseline);

    // -------------------------------------------------
    // Version 3: Optimized GPU histogram
    // -------------------------------------------------
    int histogramGPUOptimized[256] = {};
    computeHistogramGPUOptimizedTimedWithBlockSize(image, width, height, histogramGPUOptimized, optimizedThreadsPerBlock);

    // Validate baseline vs optimized histogram
    bool histogramsEqual = true;
    for (int i = 0; i < 256; i++) {
        if (histogramGPUBaseline[i] != histogramGPUOptimized[i]) {
            histogramsEqual = false;
            std::cout << "Histogram mismatch at bin " << i
                      << ": baseline=" << histogramGPUBaseline[i]
                      << ", optimized=" << histogramGPUOptimized[i] << "\n";
        }
    }

    if (histogramsEqual) {
        std::cout << "Baseline and optimized GPU histograms match.\n";
    } else {
        std::cout << "Baseline and optimized GPU histograms do NOT match.\n";
    }

    // Validate skip-zero variant against baseline histogram
    int histogramSkipZero[256] = {};
    float skipZeroCheckTime = computeHistogramGPUSharedSkipZeroTimedWithBlockSize(
        image,
        width,
        height,
        histogramSkipZero,
        optimizedThreadsPerBlock
    );

    if (skipZeroCheckTime < 0.0f) {
        std::cerr << "Skip-zero GPU histogram computation failed.\n";
        freeImage(image);
        delete[] outputCPU;
        return 1;
    }

    bool skipZeroEqual = true;
    for (int i = 0; i < 256; i++) {
        if (histogramSkipZero[i] != histogramGPUBaseline[i]) {
            skipZeroEqual = false;
            std::cout << "Skip-zero histogram mismatch at bin " << i
                      << ": baseline=" << histogramGPUBaseline[i]
                      << ", skipZero=" << histogramSkipZero[i] << "\n";
            break;
        }
    }

    if (skipZeroEqual) {
        std::cout << "Skip-zero GPU histogram matches baseline histogram.\n";
    } else {
        std::cout << "Skip-zero GPU histogram does NOT match baseline histogram.\n";
    }

    // Validate single-thread merge variant against baseline histogram
    int histogramSingleThreadMerge[256] = {};
    float singleThreadMergeCheckTime = computeHistogramGPUSingleThreadMergeTimedWithBlockSize(
        image,
        width,
        height,
        histogramSingleThreadMerge,
        optimizedThreadsPerBlock
    );

    if (singleThreadMergeCheckTime < 0.0f) {
        std::cerr << "Single-thread merge GPU histogram computation failed.\n";
        freeImage(image);
        delete[] outputCPU;
        return 1;
    }

    bool singleThreadMergeEqual = true;
    for (int i = 0; i < 256; i++) {
        if (histogramSingleThreadMerge[i] != histogramGPUBaseline[i]) {
            singleThreadMergeEqual = false;
            std::cout << "Single-thread merge histogram mismatch at bin " << i
                      << ": baseline=" << histogramGPUBaseline[i]
                      << ", singleThreadMerge=" << histogramSingleThreadMerge[i] << "\n";
            break;
        }
    }

    if (singleThreadMergeEqual) {
        std::cout << "Single-thread merge GPU histogram matches baseline histogram.\n";
    } else {
        std::cout << "Single-thread merge GPU histogram does NOT match baseline histogram.\n";
    }

    // Validate reduced-grid variant against baseline histogram
    int histogramReducedGrid[256] = {};

    float reducedGridCheckTime = computeHistogramGPUReducedGridTimedWithBlockSize(
        image,
        width,
        height,
        histogramReducedGrid,
        optimizedThreadsPerBlock,
        reducedGridBlocksPerSM
    );

    if (reducedGridCheckTime < 0.0f) {
        std::cerr << "Reduced-grid GPU histogram computation failed.\n";
        freeImage(image);
        delete[] outputCPU;
        return 1;
    }

    bool reducedGridEqual = true;

    for (int i = 0; i < 256; i++) {
        if (histogramReducedGrid[i] != histogramGPUBaseline[i]) {
            reducedGridEqual = false;
            std::cout << "Reduced-grid histogram mismatch at bin " << i
                    << ": baseline=" << histogramGPUBaseline[i]
                    << ", reducedGrid=" << histogramReducedGrid[i] << "\n";
            break;
        }
    }

    if (reducedGridEqual) {
        std::cout << "Reduced-grid GPU histogram matches baseline histogram.\n";
    } else {
        std::cout << "Reduced-grid GPU histogram does NOT match baseline histogram.\n";
    }

    // Validate bin-reduction variant against baseline histogram
    int histogramBinReduction[256] = {};

    float binReductionCheckTime = computeHistogramGPUBinReductionTimedWithBlockSize(
        image,
        width,
        height,
        histogramBinReduction,
        optimizedThreadsPerBlock
    );

    if (binReductionCheckTime < 0.0f) {
        std::cerr << "Bin-reduction GPU histogram computation failed.\n";
        freeImage(image);
        delete[] outputCPU;
        return 1;
    }

    bool binReductionEqual = true;

    for (int i = 0; i < 256; i++) {
        if (histogramBinReduction[i] != histogramGPUBaseline[i]) {
            binReductionEqual = false;
            std::cout << "Bin-reduction histogram mismatch at bin " << i
                    << ": baseline=" << histogramGPUBaseline[i]
                    << ", binReduction=" << histogramBinReduction[i] << "\n";
            break;
        }
    }

    if (binReductionEqual) {
        std::cout << "Bin-reduction GPU histogram matches baseline histogram.\n";
    } else {
        std::cout << "Bin-reduction GPU histogram does NOT match baseline histogram.\n";
    }

    // Build LUT from optimized histogram
    unsigned char lut[256] = {};
    buildLUTFromHistogram(histogramGPUOptimized, totalPixels, lut);

    // Apply LUT on GPU
    unsigned char* outputGPU = new unsigned char[totalPixels];
    applyLUTGPU(image, outputGPU, width, height, lut);

    if (!saveGrayscaleImage(outputGPUPath, outputGPU, width, height)) {
        freeImage(image);
        delete[] outputCPU;
        delete[] outputGPU;
        return 1;
    }

    std::cout << "Equalized GPU image saved successfully to: " << outputGPUPath << "\n";

    // Validate CPU vs GPU output
    int differentPixels = 0;
    int maxAbsDiff = 0;

    for (int i = 0; i < totalPixels; i++) {
        int diff = std::abs(static_cast<int>(outputCPU[i]) - static_cast<int>(outputGPU[i]));
        if (diff != 0) {
            differentPixels++;
            if (diff > maxAbsDiff) {
                maxAbsDiff = diff;
            }
        }
    }

    if (differentPixels == 0) {
        std::cout << "CPU and GPU outputs are exactly identical.\n";
    } else {
        std::cout << "CPU and GPU outputs are NOT identical.\n";
        std::cout << "Different pixels: " << differentPixels << "\n";
        std::cout << "Maximum absolute difference: " << maxAbsDiff << "\n";
    }

    // -------------------------------------------------
    // Benchmarking and experimental evaluation
    // -------------------------------------------------
    std::cout << "\nStarting benchmark with " << repetitions << " repetitions...\n";

    std::vector<double> cpuTimes;
    std::vector<double> gpuBaselineHistTimes;
    std::vector<double> gpuOptimizedHistTimes;
    std::vector<double> gpuApplyLUTTimes;
    std::vector<double> gpuTotalOptimizedTimes;
    std::vector<double> gpuSkipZeroHistTimes;
    std::vector<double> gpuSingleThreadMergeHistTimes;
    std::vector<double> gpuReducedGridHistTimes;
    std::vector<double> gpuBinReductionHistTimes;

    cpuTimes.reserve(repetitions);
    gpuBaselineHistTimes.reserve(repetitions);
    gpuOptimizedHistTimes.reserve(repetitions);
    gpuApplyLUTTimes.reserve(repetitions);
    gpuTotalOptimizedTimes.reserve(repetitions);
    gpuSkipZeroHistTimes.reserve(repetitions);
    gpuSingleThreadMergeHistTimes.reserve(repetitions);
    gpuReducedGridHistTimes.reserve(repetitions);
    gpuBinReductionHistTimes.reserve(repetitions);

    for (int r = 0; r < repetitions; r++) {
        // Version 1: CPU timing
        unsigned char* tempCPU = new unsigned char[totalPixels];
        auto cpuStart = std::chrono::high_resolution_clock::now();
        equalizeHistogramCPU(image, tempCPU, width, height);
        auto cpuEnd = std::chrono::high_resolution_clock::now();
        double cpuTimeMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();
        cpuTimes.push_back(cpuTimeMs);
        delete[] tempCPU;

        // Version 2: Baseline GPU histogram timing
        int tempHistBaseline[256] = {};
        double baselineHistMs = static_cast<double>(
            computeHistogramGPUBaselineTimed(image, width, height, tempHistBaseline)
        );
        gpuBaselineHistTimes.push_back(baselineHistMs);

        // Version 3: Optimized GPU histogram timing
        int tempHistOptimized[256] = {};
        double optimizedHistMs = static_cast<double>(
            computeHistogramGPUOptimizedTimedWithBlockSize(image, width, height, tempHistOptimized, optimizedThreadsPerBlock)
        );
        gpuOptimizedHistTimes.push_back(optimizedHistMs);

        // Extra optimization 1: shared-memory histogram with skip-zero merge
        int tempHistSkipZero[256] = {};

        double skipZeroHistMs = static_cast<double>(
            computeHistogramGPUSharedSkipZeroTimedWithBlockSize(
                image,
                width,
                height,
                tempHistSkipZero,
                optimizedThreadsPerBlock
            )
        );

        gpuSkipZeroHistTimes.push_back(skipZeroHistMs);

        // Extra optimization 2: shared-memory histogram with single-thread merge
        int tempHistSingleThreadMerge[256] = {};

        double singleThreadMergeHistMs = static_cast<double>(
            computeHistogramGPUSingleThreadMergeTimedWithBlockSize(
                image,
                width,
                height,
                tempHistSingleThreadMerge,
                optimizedThreadsPerBlock
            )
        );

        gpuSingleThreadMergeHistTimes.push_back(singleThreadMergeHistMs);

        // Extra optimization 3: shared-memory histogram with reduced grid
        int tempHistReducedGrid[256] = {};

        double reducedGridHistMs = static_cast<double>(
            computeHistogramGPUReducedGridTimedWithBlockSize(
                image,
                width,
                height,
                tempHistReducedGrid,
                optimizedThreadsPerBlock,
                reducedGridBlocksPerSM
            )
        );

        if (reducedGridHistMs < 0.0) {
            std::cerr << "Reduced-grid histogram benchmark failed.\n";
            freeImage(image);
            delete[] outputCPU;
            return 1;
        }

        gpuReducedGridHistTimes.push_back(reducedGridHistMs);


        // Extra optimization 4: bin-oriented reduction without atomicAdd
        int tempHistBinReduction[256] = {};

        double binReductionHistMs = static_cast<double>(
            computeHistogramGPUBinReductionTimedWithBlockSize(
                image,
                width,
                height,
                tempHistBinReduction,
                optimizedThreadsPerBlock
            )
        );

        if (binReductionHistMs < 0.0) {
            std::cerr << "Bin-reduction histogram benchmark failed.\n";
            freeImage(image);
            delete[] outputCPU;
            delete[] outputGPU;
            return 1;
        }

        gpuBinReductionHistTimes.push_back(binReductionHistMs);

        // Build LUT from optimized histogram
        unsigned char tempLUT[256] = {};
        buildLUTFromHistogram(tempHistOptimized, totalPixels, tempLUT);

        // Common GPU stage: LUT timing
        unsigned char* tempGPUOutput = new unsigned char[totalPixels];
        double applyMs = static_cast<double>(
            applyLUTGPUTimed(image, tempGPUOutput, width, height, tempLUT)
        );
        gpuApplyLUTTimes.push_back(applyMs);
        gpuTotalOptimizedTimes.push_back(optimizedHistMs + applyMs);
        delete[] tempGPUOutput;
    }

    Stats cpuStats = computeStats(cpuTimes);
    Stats gpuBaselineHistStats = computeStats(gpuBaselineHistTimes);
    Stats gpuOptimizedHistStats = computeStats(gpuOptimizedHistTimes);
    Stats gpuApplyStats = computeStats(gpuApplyLUTTimes);
    Stats gpuTotalOptimizedStats = computeStats(gpuTotalOptimizedTimes);
    Stats gpuSkipZeroHistStats = computeStats(gpuSkipZeroHistTimes);
    Stats gpuSingleThreadMergeHistStats = computeStats(gpuSingleThreadMergeHistTimes);
    Stats gpuReducedGridHistStats = computeStats(gpuReducedGridHistTimes);
    Stats gpuBinReductionHistStats = computeStats(gpuBinReductionHistTimes);

    std::cout << "\n--- Benchmark summary (" << repetitions << " repetitions) ---\n";

    std::cout << "Version 1 - CPU total time avg/min/max: "
              << cpuStats.avg << " / "
              << cpuStats.min << " / "
              << cpuStats.max << " ms\n";

    std::cout << "Version 2 - GPU baseline histogram avg/min/max: "
              << gpuBaselineHistStats.avg << " / "
              << gpuBaselineHistStats.min << " / "
              << gpuBaselineHistStats.max << " ms\n";

    std::cout << "Version 3 - GPU optimized histogram avg/min/max: "
              << gpuOptimizedHistStats.avg << " / "
              << gpuOptimizedHistStats.min << " / "
              << gpuOptimizedHistStats.max << " ms\n";

    std::cout << "Common GPU stage - apply LUT avg/min/max: "
              << gpuApplyStats.avg << " / "
              << gpuApplyStats.min << " / "
              << gpuApplyStats.max << " ms\n";

    std::cout << "Version 3 - GPU total optimized avg/min/max: "
              << gpuTotalOptimizedStats.avg << " / "
              << gpuTotalOptimizedStats.min << " / "
              << gpuTotalOptimizedStats.max << " ms\n";

    std::cout << "Extra 1 - GPU shared skip-zero histogram avg/min/max: "
              << gpuSkipZeroHistStats.avg << " / "
              << gpuSkipZeroHistStats.min << " / "
              << gpuSkipZeroHistStats.max << " ms\n";

    std::cout << "Extra 2 - GPU single-thread merge histogram avg/min/max: "
              << gpuSingleThreadMergeHistStats.avg << " / "
              << gpuSingleThreadMergeHistStats.min << " / "
              << gpuSingleThreadMergeHistStats.max << " ms\n";

    std::cout << "Extra 3 - GPU reduced-grid histogram (blocksPerSM="
              << reducedGridBlocksPerSM << ") avg/min/max: "
              << gpuReducedGridHistStats.avg << " / "
              << gpuReducedGridHistStats.min << " / "
              << gpuReducedGridHistStats.max << " ms\n";

    std::cout << "Extra 4 - GPU bin-reduction histogram avg/min/max: "
              << gpuBinReductionHistStats.avg << " / "
              << gpuBinReductionHistStats.min << " / "
              << gpuBinReductionHistStats.max << " ms\n";

    if (gpuTotalOptimizedStats.avg > 0.0) {
        std::cout << "Average speedup (CPU / optimized GPU): "
             << (cpuStats.avg / gpuTotalOptimizedStats.avg) << "x\n";
    }

    if (gpuOptimizedHistStats.avg > 0.0) {
        std::cout << "Average histogram improvement (baseline / optimized): "
            << (gpuBaselineHistStats.avg / gpuOptimizedHistStats.avg) << "x\n";
    }

    if (gpuSkipZeroHistStats.avg > 0.0) {
        std::cout << "Skip-zero ratio vs original optimized histogram: "
            << (gpuOptimizedHistStats.avg / gpuSkipZeroHistStats.avg) << "x\n";
    }

    if (gpuSingleThreadMergeHistStats.avg > 0.0) {
        std::cout << "Single-thread merge ratio vs original optimized histogram: "
            << (gpuOptimizedHistStats.avg / gpuSingleThreadMergeHistStats.avg) << "x\n";
    }

    if (gpuReducedGridHistStats.avg > 0.0) {
        std::cout << "Reduced-grid ratio vs original optimized histogram: "
            << (gpuOptimizedHistStats.avg / gpuReducedGridHistStats.avg) << "x\n";
    }

    if (gpuBinReductionHistStats.avg > 0.0) {
        std::cout << "Bin-reduction ratio vs original optimized histogram: "
            << (gpuOptimizedHistStats.avg / gpuBinReductionHistStats.avg) << "x\n";
    }

    freeImage(image);
    delete[] outputCPU;
    delete[] outputGPU;

    return 0;
}

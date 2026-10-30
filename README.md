# CUDA Histogram Equalization

Grayscale histogram equalization implemented in C++ on the CPU and in CUDA on the GPU. The project compares several GPU strategies for computing the histogram, the step most affected by memory contention, and measures them on an NVIDIA T4.

<!-- Before/after example: drag Small.png and the equalized output into the GitHub editor here -->

## Pipeline

1. **Load** the input image as 8-bit grayscale (`stb_image`).
2. **Histogram:** a sequential CPU reference and several GPU kernels (see below).
3. **CDF and LUT:** the cumulative distribution is turned into a 256-entry lookup table.
4. **Apply the LUT** to every pixel on the CPU and on the GPU (`applyLUTKernel`).
5. **Validate:** every GPU histogram is compared with the baseline, and the CPU and GPU output images are compared pixel by pixel. All variants produce identical results.
6. **Benchmark:** each measurement is repeated 10 times and the average, minimum and maximum are reported.

## GPU histogram strategies

| Variant | Kernel | Idea |
|---|---|---|
| Baseline | `histogramKernelBaseline` | Every thread updates a single global histogram with `atomicAdd`. Simple, but every pixel competes for the same 256 counters. |
| Shared memory | `histogramKernelOptimized` | Each block builds a private histogram in shared memory, then merges its 256 bins into the global histogram with one atomic per bin. |
| Skip-zero merge | `histogramKernelSharedSkipZero` | Same as above, but skips the global atomic for empty bins. |
| Single-thread merge | `histogramKernelSingleThreadMerge` | Only thread 0 of each block performs the 256 global atomics. |
| Reduced grid | `histogramKernelOptimized` | Launches only `blocksPerSM × #SMs` blocks (640 on a T4), so each thread processes several pixels and far fewer blocks need to merge. |
| Bin reduction | `histogramKernelBinReduction` | One block per bin and no atomics, but each of the 256 blocks scans the whole image. |

## Results

NVIDIA Tesla T4 (40 SMs) · Intel Xeon @ 2.00 GHz (2 vCPUs, Google Colab) · CUDA 13.0 · Release build · 10 repetitions per measurement.

**What is timed.** GPU times are measured with CUDA events and **include the host-to-device copy of the image** and the copy of the result back to the host. They exclude `cudaMalloc`. The CPU time covers the full single-threaded equalization: histogram, LUT and LUT application.

### End to end: CPU vs. GPU

GPU total = shared-memory histogram + LUT application, with 256 threads per block.

| Image | Resolution | CPU (ms) | GPU (ms) | Speedup |
|---|---|---|---|---|
| Small | 960 × 540 (0.5 MP) | 0.73 | 0.55 | 1.32× |
| Medium | 1920 × 1080 (2.1 MP) | 2.81 | 1.83 | 1.54× |
| Large | 3840 × 2160 (8.3 MP) | 10.94 | 6.06 | 1.81× |

The best configuration (`Large.png`, 512 threads per block) reaches **2.00×** (11.28 ms vs. 5.65 ms).

### Histogram strategies

`Large.png`, 256 threads per block, histogram stage only:

| Variant | Time (ms) | vs. baseline |
|---|---|---|
| Baseline (global atomics) | 5.37 | 1.00× |
| Shared memory | 2.36 | 2.27× |
| Skip-zero merge | 2.09 | 2.57× |
| Single-thread merge | 3.08 | 1.74× |
| **Reduced grid** | **1.89** | **2.84×** |
| Bin reduction | 25.81 | 0.21× |

### Block size sweep

`Large.png`, histogram stage in ms. The *merge atomics* column is the number of global atomics that the shared-memory kernel performs when the blocks merge their private histograms (256 per block).

| Threads per block | Blocks | Merge atomics | Shared memory | Skip-zero | Single-thread merge | Reduced grid |
|---|---|---|---|---|---|---|
| 128 | 64,800 | 16.6 M | 2.92 | 2.25 | 3.58 | 1.89 |
| 256 | 32,400 | 8.3 M | 2.36 | 2.09 | 3.08 | 1.89 |
| 512 | 16,200 | 4.1 M | 2.08 | 2.03 | 3.01 | 1.89 |
| 1024 | 8,100 | 2.1 M | 2.07 | 2.03 | 2.95 | 1.90 |

### Discussion

- **Privatizing the histogram in shared memory is the key optimization.** It moves almost all atomics to fast on-chip memory and cuts the histogram time by 2.3–2.8×.
- **The merge step explains the block-size sweep.** With small blocks, the 256 merge atomics per block add up: at 128 threads per block there are about 16.6 M merge atomics, twice as many as there are pixels. Larger blocks reduce this cost, and so does skipping empty bins (1.30× faster than the plain shared-memory kernel at 128 threads per block). The **reduced grid** removes the problem: with only 640 blocks, it needs about 164 K merge atomics regardless of block size, which is why its time stays at 1.89 ms.
- **The single-thread merge is slower**, because one thread performs 256 serialized atomics while the rest of the block waits.
- **Bin reduction avoids atomics but is the slowest variant:** it reads the whole image 256 times.
- **The end-to-end speedup is modest (1.3–2.0×) for three reasons.** First, the GPU times include PCIe transfers from pageable memory. Second, the image is uploaded twice, once for the histogram and once for the LUT stage. Third, the CPU baseline is a cheap, memory-bound loop of about 1.3 ns per pixel. The speedup grows with image size, as fixed overheads are amortized.
- The CPU time of `Large.png` varied between 10.85 and 11.45 ms across runs, so the end-to-end difference between 512 and 1024 threads per block is within noise.

**Next steps:** upload the image once and reuse it for both stages, use pinned host memory (`cudaMallocHost`), time kernels and transfers separately, overlap them with CUDA streams, and compare against a multithreaded CPU baseline.

## Build and run

**Requirements:** an NVIDIA GPU with the CUDA Toolkit, CMake 3.18 or newer and a C++17 compiler (GCC on Linux, MSVC on Windows). No local GPU? The benchmarks above were run on a free Google Colab GPU runtime.

```bash
git clone https://github.com/ivanpnto/cuda-histogram-equalization.git
cd cuda-histogram-equalization
cmake -S . -B build
cmake --build build --config Release -j
```

The build defaults to `Release` and to CUDA architecture `75` (Turing, e.g. T4). For another GPU, pass its architecture, for example `-DCMAKE_CUDA_ARCHITECTURES=86` for an RTX 30-series card.

Run it from the `build/` directory, because input and output paths are relative to it:

```bash
cd build
./histogram_equalization ../data/input/Large.png 512 16             # Linux
.\Release\histogram_equalization.exe ..\data\input\Large.png 512 16  # Windows (Visual Studio generator)
```

| Argument | Meaning | Default |
|---|---|---|
| 1 | Input image path | `../data/input/Large.png` |
| 2 | Threads per block for the shared-memory kernels (`128`, `256`, `512` or `1024`) | `1024` |
| 3 | Blocks per SM for the reduced-grid variant | `16` |

The program prints the validation results and the benchmark summary, and writes the equalized images to `data/output/`.

## Project structure

```
├── include/       Headers for image I/O and the CPU and GPU implementations
├── src/           main.cpp (pipeline, validation, benchmark) and implementations
├── external/      stb_image and stb_image_write
├── data/input/    Test images (Small, Medium, Large, Original)
├── data/output/   Equalized images (generated)
└── CMakeLists.txt
```

## Third-party code

[`stb_image` and `stb_image_write`](https://github.com/nothings/stb) by Sean Barrett, dual-licensed under the MIT License and the public domain.

## License

Released under the [MIT License](LICENSE).

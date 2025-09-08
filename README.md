# FastTTS: Accelerated Text-to-Speech

FastTTS is a high-performance, real-time text-to-speech (TTS) synthesis system built on PyTorch. This project leverages the power of `torch.compile` and NVIDIA's CUDA technology to significantly reduce inference latency and increase throughput, making it ideal for applications that require fast, responsive speech generation.

## Features

* **High Performance:** Achieves substantial speedups for TTS inference compared to standard eager-mode execution.
* **CUDA Acceleration:** Fully utilizes the parallel processing capabilities of NVIDIA GPUs for a massive performance boost.
* **Optimized with `torch.compile`:** Compiles the PyTorch model graph into a highly optimized, efficient kernel, reducing Python overhead and enabling advanced optimizations like kernel fusion.
* **Easy to Use:** Minimal code changes are required to integrate the `torch.compile` optimization.
* **Real-time Synthesis:** Designed for low-latency applications like conversational AI, virtual assistants, and real-time audio generation.

## Prerequisites

Before you begin, ensure you have the following installed:
* Python 3.10+
* NVIDIA GPU with a compatible CUDA-enabled driver.
* (Optional) CUDA Toolkit (Ensure the version matches your PyTorch installation).

## Installation

Clone the repository:
```bash
git clone https://github.com/ChenNingCong/FastTTS.git
cd FastTTS
```

Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

Install dependencies:
```bash
pip install .
```

## Usage

To get started with FastTTS, you simply run its main Gradio script from the command line.

```bash
python -m fasttts.demo.main --compiled_dir <dir>
```
This command will automatically compile your model, which can take a while. To save you time on future runs, FastTTS caches the compiled model. Just replace `dir` with the path where you want to store the cached model.

## How It Works

### `torch.compile`

Introduced in PyTorch 2.0, `torch.compile` is a just-in-time (JIT) compiler that transforms your PyTorch model into optimized kernels. It reduces Python overhead, fuses multiple operations into a single kernel, and minimizes memory transfers between the CPU and GPU. This results in a significant reduction in per-call latency, especially for small batch sizes, which is common in real-time TTS.

### CUDA Acceleration

By default, PyTorch uses an NCHW memory layout. However, modern GPUs are more efficient with an NHWC kernel. If you have the cudatoolkit installed, FastTTS can directly call cuDNN to use this more efficient layout.

### Fast Attention Kernel With Flashinfer
Fast attention kernels like those in FlashInfer make it possible to process text of varying lengths more efficiently. Naive padding, which makes all inputs the same length, wastes computation on tokens that are just placeholders. FastTTS addresses this by using FlashInfer to replace the text encoder from Kokoro-TTS, which speeds up the model's inference.

## License

This project is licensed under the Apache 2.0 License.
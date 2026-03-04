# TSTorch

TSTorch is a PyTorch-like Machine Learning framework in TypeScript + WebGPU, intended as a working library and educational resource.

Once completed, TSTorch will also allow runtime analysis, exposing the core execution mechanisms behind modern deep learning systems: autograd, graph capture, kernel fusion, and GPU memory planning.

## Project status

Active early development.

Features:
- Tensor operations
- Computation graph
- Backpropagation
- Autograd
- Multi-threaded accelerated operations
- GPU accelerated operations
- Matrix-multiplication using tensor-cores

Goals:

* predictable execution traces
* stable ONNX subset
* reproducible latency reporting across devices
* tooling for edge deployment decisions

## Steps to run demo
To run demo: `pnpm run demo`
To run tests: `pnpm run test-tstorch`

## Acknowledgements
- [MiniTorch diy teaching library](https://minitorch.github.io/)
- [Good blog on autograd](https://mathblog.vercel.app/blog/autograd/)
# TSTorch

TSTorch is a model execution observability runtime for developers who need to know whether a neural network will actually work on a real device.
Unlike PyTorch or ONNX Runtime, which focus on running models, TSTorch focuses on **revealing what actually happened while the model executed**. That is, latency sources, memory peaks, GPU dispatch behavior, and operator fusion.

## Typical workflow:

1. Train or fine-tune in PyTorch
2. Export to ONNX
3. Run with TSTorch on a real device
4. Inspect latency + memory behavior
5. Decide if the device can support it

After running a model you will see information similar to:

```
Model: resnet18.onnx

Total latency: 42.6 ms
Peak GPU memory: 312 MB
GPU dispatches: 184

Top latency contributors:
conv2d_13        9.8 ms
batchnorm_13     6.4 ms
relu_13          fused
matmul_1         5.1 ms

Operator fusion:
conv + batchnorm + relu -> fused (12 occurrences)

Conclusion:
Model is deployable on mid-range integrated GPUs.
```

Note: while TSTorch is capable and correct for training, serving APIs, and maximum throughput inference, they are not usecases why a user should adopt TSTorch today

## Project status

Active early development.

Goals:

* predictable execution traces
* stable ONNX subset
* reproducible latency reporting across devices
* tooling for edge deployment decisions

---

## Steps to run demo
To run demo: `pnpm run demo`
To run tests: `pnpm run test-tstorch`

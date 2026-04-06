# @mni-ml/framework

A minimal machine learning library written in TypeScript. Built for learning and experimentation. Implements core abstractions found in PyTorch -- autograd, tensors, modules, and training -- from scratch.

Inspired by [minitorch](https://minitorch.github.io/).

## Install

```bash
npm install @mni-ml/framework
```

## What's included

**Autodiff engine** -- both scalar-level and tensor-level automatic differentiation. Supports computation graph construction, topological sort, backpropagation, and chain rule.

**Tensors** -- n-dimensional arrays backed by `Float64Array`. Supports broadcasting, permutation, reshaping, contiguous conversion, and strided storage.

**Tensor operations** -- element-wise (map), pairwise (zip), reductions (sum, mean, max), matrix multiplication, 1D and 2D convolutions.

**Acceleration** -- parallel CPU ops via worker threads (`fast_ops`) for element-wise, zip, and reduce operations. WebGPU-based GPU ops are implemented but not yet wired into the autodiff graph.

**Module system** -- `Module` base class with automatic parameter registration via `Proxy`, recursive parameter collection, `train()`/`eval()` mode switching, and named parameter traversal.

**Layers** -- `Linear`, `Conv1d`, `Conv2d`, `Embedding` (with efficient scatter-add backward), `ReLU`, `Sigmoid`, `Tanh`.

**Loss functions** -- `mseLoss` (mean squared error), `crossEntropyLoss`.

**Functional ops** -- `softmax`, `logsoftmax`, `dropout`, `avgpool2d`, `maxpool2d`, `tile`.

**Optimizer** -- `SGD` with support for both `Scalar` and `Tensor` parameters.

**Datasets** -- built-in 2D classification datasets: simple, diagonal, split, xor, circle, spiral.

## Quick start

```typescript
import {
  Tensor, Linear, ReLU, SGD, mseLoss, Module, Parameter
} from "@mni-ml/framework";

// Define a model
class MLP extends Module {
  l1: Linear;
  l2: Linear;
  relu: ReLU;

  constructor() {
    super();
    this.l1 = new Linear(2, 10);
    this.relu = new ReLU();
    this.l2 = new Linear(10, 1);
  }

  forward(x: Tensor): Tensor {
    return this.l2.forward(this.relu.forward(this.l1.forward(x)));
  }
}

const model = new MLP();
const opt = new SGD(model.parameters(), 0.05);

// Training loop
for (let epoch = 0; epoch < 100; epoch++) {
  const x = Tensor.tensor([[0.1, 0.9], [0.8, 0.2]]);
  const target = Tensor.tensor([[1], [0]]);

  const pred = model.forward(x);
  const loss = mseLoss(pred, target);

  opt.zeroGrad();
  loss.backward();
  opt.step();
}
```

## API overview

### Tensor

```typescript
Tensor.tensor([[1, 2], [3, 4]])       // from nested arrays
Tensor.zeros([3, 3])                   // zeros
Tensor.ones([2, 4])                    // ones
Tensor.rand([2, 3])                    // uniform random

t.add(other)   t.sub(other)   t.mul(other)   // arithmetic
t.neg()        t.exp()        t.log()         // unary
t.sigmoid()    t.relu()                       // activations
t.matmul(other)                               // matrix multiply
t.conv1d(weight)  t.conv2d(weight)            // convolutions
t.sum(dim?)    t.mean(dim?)   t.max(dim)      // reductions
t.permute(...order)  t.view(...shape)         // reshaping
t.backward()                                  // backpropagation
```

### Modules

| Module | Description |
|--------|-------------|
| `Linear(in, out)` | Fully connected layer |
| `Conv1d(inCh, outCh, kernelW)` | 1D convolution |
| `Conv2d(inCh, outCh, [kH, kW])` | 2D convolution |
| `Embedding(numEmb, embDim)` | Lookup table with trainable weights |
| `ReLU` | Rectified linear unit |
| `Sigmoid` | Sigmoid activation |
| `Tanh` | Hyperbolic tangent activation |

### Loss functions

| Function | Use case |
|----------|----------|
| `mseLoss(pred, target)` | Regression |
| `crossEntropyLoss(pred, target)` | Classification |

### Functional

```typescript
softmax(input, dim)          // softmax probabilities
logsoftmax(input, dim)       // log-softmax (numerically stable)
dropout(input, rate, ignore) // inverted dropout
avgpool2d(input, [kH, kW])   // average pooling
maxpool2d(input, [kH, kW])   // max pooling
```

## Demos

The `packages/demo` folder contains working training scripts.

```bash
# 2D classification with scalar autograd
pnpm run demo scalar

# 2D classification with tensors
pnpm run demo

# Fast tensor backend with CLI args
pnpm run demo fast --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05 --EPOCHS 500
pnpm run demo fast --DATASET all

# MNIST digit classification (CNN)
pnpm run demo mnist

# SST-2 sentiment analysis (Conv1d + Embedding)
pnpm run demo sentiment
```

## Training results

### 2D classification (fast backend)

Configuration: `--BACKEND cpu --HIDDEN 100 --RATE 0.05`, 50 data points, 3-layer MLP (2 -> 100 -> 100 -> 1).

| Dataset | Epochs | Final Loss | Accuracy | Avg ms/epoch |
|---------|--------|-----------|----------|-------------|
| Simple  | 500    | 1.99      | 50/50    | 15.80       |
| Diag    | 500    | 5.04      | 49/50    | 15.76       |
| Split   | 500    | 20.96     | 48/50    | 16.00       |
| Xor     | 500    | 23.02     | 45/50    | 15.57       |
| Circle  | 1000   | 10.14     | 47/50    | 15.64       |
| Spiral  | 1000   | 33.63     | 29/50    | 15.53       |

Spiral is the hardest dataset (non-linearly separable with rotational structure). Simple and Diag converge to near-perfect accuracy within 500 epochs.

## Tests

```bash
pnpm run test-framework
```

11 test files covering autodiff, scalar operations, tensor data, tensor operations, tensor functions, modules, neural network layers, fast ops, and GPU ops.

## Project structure

```
packages/
  framework/        -- the library (published as @mni-ml/framework)
    src/
      autodiff.ts       -- backpropagation, topological sort, chain rule
      scalar.ts         -- scalar with computation history
      scalar_functions.ts -- scalar forward/backward ops
      tensor.ts         -- tensor class with autograd
      tensor_data.ts    -- strided n-dimensional storage
      tensor_functions.ts -- tensor forward/backward ops
      tensor_ops.ts     -- map, zip, reduce, conv implementations
      fast_ops.ts       -- parallel CPU ops via worker threads
      gpu_ops.ts        -- WebGPU compute shader ops
      module.ts         -- Module, Parameter base classes
      nn.ts             -- layers, loss functions, functional ops
      optimizer.ts      -- SGD optimizer
      operators.ts      -- scalar math operators
      datasets.ts       -- 2D classification datasets
      index.ts          -- public exports
  demo/             -- training scripts and examples
```

## License

MIT

## Acknowledgements

- [minitorch](https://minitorch.github.io/) -- the teaching library this project follows
- [Autograd blog post](https://mathblog.vercel.app/blog/autograd/)

import { _setNative } from "./tensor.js";
import { loadNative } from "./native-loader.js";

_setNative(loadNative());

export { Tensor, native, type TensorLike, type Shape } from "./tensor.js";
export { Module, Parameter } from "./module.js";
export {
    Linear, ReLU, Sigmoid, Tanh, Embedding,
    Conv1d, Conv2d,
    softmax, logsoftmax, gelu, dropout,
    crossEntropyLoss, crossEntropyLossGpu, mseLoss, layerNorm,
    flashAttention, residualLayerNorm, biasGelu,
    KvCache,
    randRange, tile, avgpool2d, maxpool2d,
} from "./nn.js";
export { Optimizer, SGD, Adam, GradScaler, type ParameterValue } from "./optimizer.js";

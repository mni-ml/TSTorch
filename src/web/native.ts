import {
    fromFloat32, zeros, ones, randTensor, randnTensor,
    tensorShape, toFloat32, getScalar, freeTensor,
    setRequiresGrad, getGrad, backward, noGradStart, noGradEnd,
    add, sub, mul, div,
    neg, mulScalar, expOp, logOp, powOp,
    relu, sigmoid, gelu,
    sumAll, sumOp, meanAll, meanOp, maxOp,
    lt, eqOp, gt, isClose,
    view, permute, contiguous,
    matmul,
    softmaxOp, embeddingForward, embeddingForwardGpu,
    dropoutOp, layernormOp, flashAttention,
    crossEntropyLoss, crossEntropyLossGpu,
    residualLayernorm, biasGelu,
    conv1DForward, conv2DForward,
    avgpool2D, maxpool2D,
    tile,
} from './ops.js';

import { clipAndStep, scaleGrads, zeroGrad } from './optimizer.js';

export const webNative: Record<string, any> = {
    // Creation
    fromFloat32,
    zeros,
    ones,
    randTensor,
    randnTensor,

    // Data access
    tensorShape,
    toFloat32,
    getScalar,
    freeTensor,

    // Gradient management
    setRequiresGrad,
    getGrad,
    backward,
    zeroGrad,
    noGradStart,
    noGradEnd,

    // Elementwise binary
    add,
    sub,
    mul,
    div,

    // Elementwise unary
    neg,
    mulScalar,
    expOp,
    logOp,
    powOp,

    // Activations
    relu,
    sigmoid,
    gelu,

    // Reductions
    sumAll,
    sumOp,
    meanAll,
    meanOp,
    maxOp,

    // Comparison
    lt,
    eqOp,
    gt,
    isClose,

    // Layout
    view,
    permute,
    contiguous,

    // Linear algebra
    matmul,

    // NN ops
    softmaxOp,
    embeddingForward,
    embeddingForwardGpu,
    dropoutOp,
    layernormOp,
    flashAttention,
    crossEntropyLoss,
    crossEntropyLossGpu,
    residualLayernorm,
    biasGelu,

    // Convolution
    conv1DForward,
    conv2DForward,

    // Pooling
    avgpool2D,
    maxpool2D,

    // Utility
    tile,

    // Optimizer
    clipAndStep,
    scaleGrads,
};

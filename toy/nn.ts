import { Module, Parameter } from './module.js';
import { Tensor } from './tensor.js';
import { TensorHistory } from './tensor_functions.js';

export class Linear extends Module {
    weight!: Parameter<Tensor>;
    bias!: Parameter<Tensor>;
    inFeatures: number;
    outFeatures: number;

    constructor(inFeatures: number, outFeatures: number) {
        super();
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        const bound = 1 / Math.sqrt(inFeatures);
        this.weight = new Parameter(Tensor.parameter([inFeatures, outFeatures], bound));
        this.bias = new Parameter(Tensor.parameter([outFeatures], bound));
    }

    forward(input: Tensor): Tensor {
        return input.matmul(this.weight.value).add(this.bias.value);
    }
}

export class RMSNorm extends Module {
    weight!: Parameter<Tensor>;
    eps: number;

    constructor(dim: number, eps: number = 1e-6) {
        super();
        this.eps = eps;
        const w = Tensor.ones([dim]);
        w.history = new TensorHistory();
        this.weight = new Parameter(w);
    }

    forward(input: Tensor): Tensor {
        const variance = input.mul(input).mean(-1);
        const rms = variance.add(this.eps).sqrt().inv();
        return input.mul(rms).mul(this.weight.value);
    }
}

export function softmax(x: Tensor, dim: number = -1): Tensor {
    if (dim < 0) dim = x.dims + dim;
    const expX = x.exp();
    const sumExp = expX.sum(dim);
    return expX.div(sumExp);
}

export function mseLoss(pred: Tensor, target: Tensor): Tensor {
    const diff = pred.sub(target);
    return diff.mul(diff).mean();
}

export function tanh(x: Tensor): Tensor {
    return x.mul(2).sigmoid().mul(2).sub(1);
}

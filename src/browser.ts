/**
 * Browser-compatible inference backend for @mni-ml/framework.
 *
 * Pure JavaScript implementation — no native bindings, no Node.js APIs.
 * Supports the subset of operations needed for transformer inference.
 */

// ============================================================
// Tensor Storage
// ============================================================

let _nextId = 0;
const _store = new Map<number, { data: Float32Array; shape: number[] }>();

function _alloc(data: Float32Array, shape: number[]): number {
    const id = _nextId++;
    _store.set(id, { data, shape });
    return id;
}

function _get(id: number): { data: Float32Array; shape: number[] } {
    const entry = _store.get(id);
    if (!entry) throw new Error(`Tensor id=${id} not found`);
    return entry;
}

function _sizeFromShape(shape: number[]): number {
    let s = 1;
    for (let i = 0; i < shape.length; i++) s *= shape[i];
    return s;
}

// ============================================================
// Core Operations (flat Float32Array math)
// ============================================================

function _broadcastAdd(aId: number, bId: number): number {
    const a = _get(aId);
    const b = _get(bId);
    const aData = a.data, bData = b.data;
    const aShape = a.shape, bShape = b.shape;

    if (aData.length === bData.length) {
        const out = new Float32Array(aData.length);
        for (let i = 0; i < aData.length; i++) out[i] = aData[i] + bData[i];
        return _alloc(out, [...aShape]);
    }

    // Broadcast b into a: a has more dims, b's shape is a suffix of a's shape
    const bSize = bData.length;
    const out = new Float32Array(aData.length);
    for (let i = 0; i < aData.length; i++) out[i] = aData[i] + bData[i % bSize];
    return _alloc(out, [...aShape]);
}

function _matmul(aId: number, bId: number): number {
    const a = _get(aId);
    const b = _get(bId);
    const aShape = a.shape, bShape = b.shape;
    const aDims = aShape.length, bDims = bShape.length;

    if (aDims === 2 && bDims === 2) {
        return _matmul2d(a.data, b.data, aShape[0], aShape[1], bShape[1]);
    }

    if (aDims === 3 && bDims === 2) {
        // [B, M, K] × [K, N] → [B, M, N]
        const B = aShape[0], M = aShape[1], K = aShape[2], N = bShape[1];
        const out = new Float32Array(B * M * N);
        const bData = b.data;
        for (let batch = 0; batch < B; batch++) {
            const aOff = batch * M * K;
            const oOff = batch * M * N;
            _matmul2dInto(a.data, aOff, bData, 0, M, K, N, out, oOff);
        }
        return _alloc(out, [B, M, N]);
    }

    if (aDims === 3 && bDims === 3) {
        // [B, M, K] × [B, K, N] → [B, M, N]
        const B = aShape[0], M = aShape[1], K = aShape[2], N = bShape[2];
        const out = new Float32Array(B * M * N);
        for (let batch = 0; batch < B; batch++) {
            const aOff = batch * M * K;
            const bOff = batch * K * N;
            const oOff = batch * M * N;
            _matmul2dInto(a.data, aOff, b.data, bOff, M, K, N, out, oOff);
        }
        return _alloc(out, [B, M, N]);
    }

    throw new Error(`matmul: unsupported shapes [${aShape}] × [${bShape}]`);
}

function _matmul2d(A: Float32Array, B: Float32Array, M: number, K: number, N: number): number {
    const out = new Float32Array(M * N);
    _matmul2dInto(A, 0, B, 0, M, K, N, out, 0);
    return _alloc(out, [M, N]);
}

function _matmul2dInto(
    A: Float32Array, aOff: number,
    B: Float32Array, bOff: number,
    M: number, K: number, N: number,
    out: Float32Array, oOff: number,
): void {
    // i-k-j loop order for cache-friendly B access
    for (let i = 0; i < M; i++) {
        const rowOff = oOff + i * N;
        for (let j = 0; j < N; j++) out[rowOff + j] = 0;
        const aRowOff = aOff + i * K;
        for (let k = 0; k < K; k++) {
            const aVal = A[aRowOff + k];
            const bRowOff = bOff + k * N;
            for (let j = 0; j < N; j++) {
                out[rowOff + j] += aVal * B[bRowOff + j];
            }
        }
    }
}

function _view(id: number, newShape: number[]): number {
    const t = _get(id);
    const newSize = _sizeFromShape(newShape);
    if (newSize !== t.data.length) {
        throw new Error(`view: size mismatch ${t.data.length} vs ${newSize} for shape [${newShape}]`);
    }
    return _alloc(t.data, newShape);
}

function _permute(id: number, dims: number[]): number {
    const t = _get(id);
    const shape = t.shape;
    const rank = shape.length;
    const data = t.data;

    const newShape = dims.map(d => shape[d]);
    const size = data.length;
    const out = new Float32Array(size);

    const oldStrides = new Array(rank);
    let stride = 1;
    for (let i = rank - 1; i >= 0; i--) { oldStrides[i] = stride; stride *= shape[i]; }

    const newStrides = new Array(rank);
    stride = 1;
    for (let i = rank - 1; i >= 0; i--) { newStrides[i] = stride; stride *= newShape[i]; }

    // Map from old flat index to new flat index
    const coords = new Array(rank);
    for (let flat = 0; flat < size; flat++) {
        let rem = flat;
        for (let d = 0; d < rank; d++) {
            coords[d] = (rem / oldStrides[d]) | 0;
            rem -= coords[d] * oldStrides[d];
        }
        let newFlat = 0;
        for (let d = 0; d < rank; d++) {
            newFlat += coords[dims[d]] * newStrides[d];
        }
        out[newFlat] = data[flat];
    }

    return _alloc(out, newShape);
}

function _embeddingForward(weightId: number, flatIndices: number[], batch: number, seqLen: number): number {
    const w = _get(weightId);
    const embedDim = w.shape[1];
    const wData = w.data;
    const out = new Float32Array(batch * seqLen * embedDim);
    for (let i = 0; i < flatIndices.length; i++) {
        const idx = flatIndices[i];
        const srcOff = idx * embedDim;
        const dstOff = i * embedDim;
        out.set(wData.subarray(srcOff, srcOff + embedDim), dstOff);
    }
    return _alloc(out, [batch, seqLen, embedDim]);
}

function _layerNorm(xId: number, gammaId: number, betaId: number, eps: number): number {
    const x = _get(xId);
    const gamma = _get(gammaId);
    const beta = _get(betaId);
    const shape = x.shape;
    const lastDim = shape[shape.length - 1];
    const outerSize = x.data.length / lastDim;
    const xData = x.data, gData = gamma.data, bData = beta.data;
    const out = new Float32Array(x.data.length);

    for (let i = 0; i < outerSize; i++) {
        const off = i * lastDim;
        let mean = 0;
        for (let j = 0; j < lastDim; j++) mean += xData[off + j];
        mean /= lastDim;

        let variance = 0;
        for (let j = 0; j < lastDim; j++) {
            const d = xData[off + j] - mean;
            variance += d * d;
        }
        variance /= lastDim;
        const invStd = 1 / Math.sqrt(variance + eps);

        for (let j = 0; j < lastDim; j++) {
            out[off + j] = (xData[off + j] - mean) * invStd * gData[j] + bData[j];
        }
    }

    return _alloc(out, [...shape]);
}

const GELU_COEFF = Math.sqrt(2 / Math.PI);
function _gelu(id: number): number {
    const t = _get(id);
    const data = t.data;
    const out = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) {
        const x = data[i];
        out[i] = 0.5 * x * (1 + Math.tanh(GELU_COEFF * (x + 0.044715 * x * x * x)));
    }
    return _alloc(out, [...t.shape]);
}

function _softmax(id: number, dim: number): number {
    const t = _get(id);
    const shape = t.shape;
    const rank = shape.length;
    const actualDim = dim < 0 ? rank + dim : dim;
    const dimSize = shape[actualDim];
    const data = t.data;
    const out = new Float32Array(data.length);

    // For the common case of last-dim softmax
    if (actualDim === rank - 1) {
        const outerSize = data.length / dimSize;
        for (let i = 0; i < outerSize; i++) {
            const off = i * dimSize;
            let max = -Infinity;
            for (let j = 0; j < dimSize; j++) {
                if (data[off + j] > max) max = data[off + j];
            }
            let sum = 0;
            for (let j = 0; j < dimSize; j++) {
                out[off + j] = Math.exp(data[off + j] - max);
                sum += out[off + j];
            }
            const invSum = 1 / sum;
            for (let j = 0; j < dimSize; j++) out[off + j] *= invSum;
        }
    } else {
        throw new Error(`softmax: only last-dim softmax supported, got dim=${dim}`);
    }

    return _alloc(out, [...shape]);
}

function _flashAttention(qId: number, kId: number, vId: number, scale: number, causal: boolean): number {
    const q = _get(qId), k = _get(kId), v = _get(vId);
    // Q, K, V: [B, S, D]
    const [B, Sq, D] = q.shape;
    const Sk = k.shape[1];
    const qData = q.data, kData = k.data, vData = v.data;

    // scores = Q @ K^T → [B, Sq, Sk], scaled
    const scores = new Float32Array(B * Sq * Sk);
    for (let b = 0; b < B; b++) {
        const qOff = b * Sq * D;
        const kOff = b * Sk * D;
        const sOff = b * Sq * Sk;
        for (let i = 0; i < Sq; i++) {
            const qRow = qOff + i * D;
            for (let j = 0; j < Sk; j++) {
                let dot = 0;
                const kRow = kOff + j * D;
                for (let d = 0; d < D; d++) dot += qData[qRow + d] * kData[kRow + d];
                scores[sOff + i * Sk + j] = dot * scale;
            }
        }
    }

    // Causal mask + softmax
    for (let b = 0; b < B; b++) {
        const sOff = b * Sq * Sk;
        for (let i = 0; i < Sq; i++) {
            const rowOff = sOff + i * Sk;
            if (causal) {
                for (let j = i + 1; j < Sk; j++) scores[rowOff + j] = -Infinity;
            }
            let max = -Infinity;
            for (let j = 0; j < Sk; j++) { if (scores[rowOff + j] > max) max = scores[rowOff + j]; }
            let sum = 0;
            for (let j = 0; j < Sk; j++) {
                scores[rowOff + j] = Math.exp(scores[rowOff + j] - max);
                sum += scores[rowOff + j];
            }
            const invSum = 1 / sum;
            for (let j = 0; j < Sk; j++) scores[rowOff + j] *= invSum;
        }
    }

    // output = scores @ V → [B, Sq, D]
    const out = new Float32Array(B * Sq * D);
    for (let b = 0; b < B; b++) {
        const sOff = b * Sq * Sk;
        const vOff = b * Sk * D;
        const oOff = b * Sq * D;
        for (let i = 0; i < Sq; i++) {
            const sRow = sOff + i * Sk;
            const oRow = oOff + i * D;
            for (let j = 0; j < Sk; j++) {
                const w = scores[sRow + j];
                const vRow = vOff + j * D;
                for (let d = 0; d < D; d++) out[oRow + d] += w * vData[vRow + d];
            }
        }
    }

    return _alloc(out, [B, Sq, D]);
}

// ============================================================
// Exported native-compatible interface (for internal use)
// ============================================================

export const native: Record<string, any> = {
    fromFloat32: (data: Float32Array, shape: number[]) => _alloc(new Float32Array(data), shape),
    zeros: (shape: number[]) => _alloc(new Float32Array(_sizeFromShape(shape)), shape),
    ones: (shape: number[]) => {
        const d = new Float32Array(_sizeFromShape(shape)); d.fill(1);
        return _alloc(d, shape);
    },
    randTensor: (shape: number[]) => {
        const size = _sizeFromShape(shape);
        const d = new Float32Array(size);
        for (let i = 0; i < size; i++) d[i] = Math.random();
        return _alloc(d, shape);
    },
    tensorShape: (id: number) => _get(id).shape,
    toFloat32: (id: number) => _get(id).data,
    getScalar: (id: number) => _get(id).data[0],
    freeTensor: (id: number) => { _store.delete(id); },
    add: _broadcastAdd,
    sub: (aId: number, bId: number) => {
        const a = _get(aId), b = _get(bId);
        const out = new Float32Array(a.data.length);
        const bLen = b.data.length;
        for (let i = 0; i < a.data.length; i++) out[i] = a.data[i] - b.data[i % bLen];
        return _alloc(out, [...a.shape]);
    },
    mul: (aId: number, bId: number) => {
        const a = _get(aId), b = _get(bId);
        const out = new Float32Array(a.data.length);
        const bLen = b.data.length;
        for (let i = 0; i < a.data.length; i++) out[i] = a.data[i] * b.data[i % bLen];
        return _alloc(out, [...a.shape]);
    },
    mulScalar: (id: number, s: number) => {
        const t = _get(id);
        const out = new Float32Array(t.data.length);
        for (let i = 0; i < t.data.length; i++) out[i] = t.data[i] * s;
        return _alloc(out, [...t.shape]);
    },
    neg: (id: number) => {
        const t = _get(id);
        const out = new Float32Array(t.data.length);
        for (let i = 0; i < t.data.length; i++) out[i] = -t.data[i];
        return _alloc(out, [...t.shape]);
    },
    expOp: (id: number) => {
        const t = _get(id);
        const out = new Float32Array(t.data.length);
        for (let i = 0; i < t.data.length; i++) out[i] = Math.exp(t.data[i]);
        return _alloc(out, [...t.shape]);
    },
    logOp: (id: number) => {
        const t = _get(id);
        const out = new Float32Array(t.data.length);
        for (let i = 0; i < t.data.length; i++) out[i] = Math.log(t.data[i]);
        return _alloc(out, [...t.shape]);
    },
    relu: (id: number) => {
        const t = _get(id);
        const out = new Float32Array(t.data.length);
        for (let i = 0; i < t.data.length; i++) out[i] = t.data[i] > 0 ? t.data[i] : 0;
        return _alloc(out, [...t.shape]);
    },
    sigmoid: (id: number) => {
        const t = _get(id);
        const out = new Float32Array(t.data.length);
        for (let i = 0; i < t.data.length; i++) out[i] = 1 / (1 + Math.exp(-t.data[i]));
        return _alloc(out, [...t.shape]);
    },
    matmul: _matmul,
    view: _view,
    permute: _permute,
    contiguous: (id: number) => {
        const t = _get(id);
        return _alloc(new Float32Array(t.data), [...t.shape]);
    },
    embeddingForward: _embeddingForward,
    softmaxOp: _softmax,
    gelu: _gelu,
    layernormOp: _layerNorm,
    flashAttention: _flashAttention,
    setRequiresGrad: () => {},
    getGrad: () => null,
    backward: () => { throw new Error('backward not supported in browser backend'); },
    zeroGrad: () => {},
    noGradStart: () => {},
    noGradEnd: () => {},
};

// ============================================================
// Tensor
// ============================================================

export type Shape = number[];

export class Tensor {
    readonly _id: number;
    private _shape: Shape;

    constructor(id: number, shape?: Shape) {
        this._id = id;
        this._shape = shape ?? native.tensorShape(id);
    }

    get shape(): Shape { return this._shape; }
    get size(): number { return _sizeFromShape(this._shape); }
    get dims(): number { return this._shape.length; }

    get grad(): Tensor | null { return null; }
    set grad(_v: Tensor | null) {}
    set history(_v: any) {}
    get history(): null { return null; }

    backward(): void { throw new Error('backward not supported in browser'); }

    toFloat32(): Float32Array { return native.toFloat32(this._id); }
    item(): number { return native.getScalar(this._id); }

    get(indices: number[]): number {
        const data = this.toFloat32();
        let flat = 0, stride = 1;
        for (let i = this._shape.length - 1; i >= 0; i--) {
            flat += indices[i] * stride;
            stride *= this._shape[i];
        }
        return data[flat];
    }

    free(): void { native.freeTensor(this._id); }

    static fromFloat32(data: Float32Array, shape: Shape): Tensor {
        const id = native.fromFloat32(data, shape);
        return new Tensor(id, shape);
    }

    static zeros(shape: Shape): Tensor {
        const id = native.zeros(shape);
        return new Tensor(id, shape);
    }

    static ones(shape: Shape): Tensor {
        const id = native.ones(shape);
        return new Tensor(id, shape);
    }

    static rand(shape: Shape): Tensor {
        const id = native.randTensor(shape);
        return new Tensor(id, shape);
    }

    add(other: Tensor | number): Tensor {
        if (typeof other === 'number') {
            const s = Tensor.fromFloat32(new Float32Array([other]), [1]);
            return new Tensor(native.add(this._id, s._id));
        }
        return new Tensor(native.add(this._id, other._id));
    }

    sub(other: Tensor | number): Tensor {
        if (typeof other === 'number') {
            const s = Tensor.fromFloat32(new Float32Array([other]), [1]);
            return new Tensor(native.sub(this._id, s._id));
        }
        return new Tensor(native.sub(this._id, other._id));
    }

    mul(other: Tensor | number): Tensor {
        if (typeof other === 'number') return new Tensor(native.mulScalar(this._id, other));
        return new Tensor(native.mul(this._id, other._id));
    }

    neg(): Tensor { return new Tensor(native.neg(this._id)); }
    exp(): Tensor { return new Tensor(native.expOp(this._id)); }
    log(): Tensor { return new Tensor(native.logOp(this._id)); }
    relu(): Tensor { return new Tensor(native.relu(this._id)); }
    sigmoid(): Tensor { return new Tensor(native.sigmoid(this._id)); }

    div(other: Tensor | number): Tensor {
        if (typeof other === 'number') return this.mul(1.0 / other);
        const a = _get(this._id), b = _get((other as Tensor)._id);
        const out = new Float32Array(a.data.length);
        const bLen = b.data.length;
        for (let i = 0; i < a.data.length; i++) out[i] = a.data[i] / b.data[i % bLen];
        return new Tensor(_alloc(out, [...a.shape]));
    }

    view(...shape: number[]): Tensor { return new Tensor(native.view(this._id, shape)); }
    permute(...dims: number[]): Tensor { return new Tensor(native.permute(this._id, dims)); }
    contiguous(): Tensor { return new Tensor(native.contiguous(this._id)); }
    matmul(other: Tensor): Tensor { return new Tensor(native.matmul(this._id, other._id)); }

    clone(): Tensor {
        const d = this.toFloat32();
        return Tensor.fromFloat32(new Float32Array(d), [...this._shape]);
    }

    detach(): Tensor { return this.clone(); }

    setRequiresGrad(_requires: boolean): Tensor { return this; }

    toString(): string {
        const data = this.toFloat32();
        const shapeStr = `[${this._shape.join(', ')}]`;
        if (data.length <= 10) {
            return `Tensor(${shapeStr}, [${Array.from(data).map(v => v.toFixed(4)).join(', ')}])`;
        }
        const first = Array.from(data.slice(0, 5)).map(v => v.toFixed(4)).join(', ');
        const last = Array.from(data.slice(-3)).map(v => v.toFixed(4)).join(', ');
        return `Tensor(${shapeStr}, [${first}, ..., ${last}])`;
    }
}

export type TensorLike = number | Tensor;

// ============================================================
// Module & Parameter
// ============================================================

abstract class BaseParameter { name?: string; }

export class Module<P extends BaseParameter = BaseParameter> {
    protected _modules: Record<string, Module<P>> = {};
    protected _parameters: Record<string, P> = {};
    training: boolean = true;

    constructor() {
        return new Proxy(this, {
            set: (target, key: string | symbol, value, receiver) => {
                if (value instanceof Module) target._modules[key as string] = value;
                else if (value instanceof BaseParameter) target._parameters[key as string] = value as P;
                return Reflect.set(target, key, value, receiver);
            }
        });
    }

    parameters(): P[] {
        const params: P[] = [];
        for (const p of Object.values(this._parameters)) params.push(p);
        for (const m of Object.values(this._modules) as Module<P>[]) params.push(...m.parameters());
        return params;
    }

    namedParameters(): Array<[string, P]> {
        const named: Array<[string, P]> = Object.entries(this._parameters);
        for (const [moduleName, mod] of Object.entries(this._modules)) {
            for (const [name, param] of mod.namedParameters()) {
                named.push([`${moduleName}.${name}`, param]);
            }
        }
        return named;
    }

    children(): Module<P>[] { return Object.values(this._modules); }

    train(): void {
        this.training = true;
        for (const c of this.children()) c.train();
    }

    eval(): void {
        this.training = false;
        for (const c of this.children()) c.eval();
    }
}

export class Parameter<T = Tensor> extends BaseParameter {
    value: T;
    constructor(value: T, name?: string) {
        super();
        this.value = value;
        if (name) this.name = name;
    }
    get grad() {
        if (this.value instanceof Tensor) return this.value.grad;
        return null;
    }
    update(v: T) { this.value = v; }
}

// ============================================================
// NN Modules
// ============================================================

function randRange(shape: number[], min: number, max: number): Tensor {
    const r = Tensor.rand(shape);
    const data = r.toFloat32();
    const range = max - min;
    const scaled = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) scaled[i] = data[i] * range + min;
    return Tensor.fromFloat32(scaled, shape);
}

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
        this.weight = new Parameter(randRange([inFeatures, outFeatures], -bound, bound));
        this.bias = new Parameter(randRange([outFeatures], -bound, bound));
    }

    forward(input: Tensor): Tensor {
        return input.matmul(this.weight.value).add(this.bias.value);
    }
}

export class Embedding extends Module {
    weight!: Parameter<Tensor>;
    vocabSize: number;
    embedDim: number;

    constructor(vocabSize: number, embedDim: number) {
        super();
        this.vocabSize = vocabSize;
        this.embedDim = embedDim;
        const bound = 1 / Math.sqrt(embedDim);
        this.weight = new Parameter(randRange([vocabSize, embedDim], -bound, bound));
    }

    forward(indices: number[][]): Tensor {
        const batch = indices.length;
        const seqLen = indices[0].length;
        const flat = indices.flat();
        const id = native.embeddingForward(this.weight.value._id, flat, batch, seqLen);
        return new Tensor(id);
    }
}

// ============================================================
// Functional NN ops
// ============================================================

export function softmax(x: Tensor, dim: number = -1): Tensor {
    return new Tensor(native.softmaxOp(x._id, dim));
}

export function gelu(x: Tensor): Tensor {
    return new Tensor(native.gelu(x._id));
}

export function dropout(x: Tensor, _rate: number = 0.0, inference: boolean = false): Tensor {
    // Browser backend is inference-only; always returns identity
    return x;
}

export function layerNorm(x: Tensor, gamma: Tensor, beta: Tensor, eps: number = 1e-5): Tensor {
    return new Tensor(native.layernormOp(x._id, gamma._id, beta._id, eps));
}

export function flashAttention(q: Tensor, k: Tensor, v: Tensor, scale: number, causal: boolean = true): Tensor {
    return new Tensor(native.flashAttention(q._id, k._id, v._id, scale, causal));
}

/**
 * Free all tensors from the store. Useful for resetting memory between runs.
 */
export function freeAll(): void {
    _store.clear();
    _nextId = 0;
}

/**
 * Return the number of live tensors in the store (for debugging).
 */
export function liveTensorCount(): number {
    return _store.size;
}

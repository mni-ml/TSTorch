import {
    store, shapeSize, computeStrides, isContiguous,
    shapeEqual, normalizeDim, broadcastShapes,
} from './store.js';
import { tape } from './tape.js';

// ===================================================================
// Helpers
// ===================================================================

function needsGrad(...ids: number[]): boolean {
    for (const id of ids) if (store.get(id).requiresGrad) return true;
    return false;
}

function allocResult(data: Float32Array, shape: number[], ...inputIds: number[]): number {
    const id = store.alloc(data, shape);
    if (tape.enabled && needsGrad(...inputIds)) {
        store.get(id).requiresGrad = true;
    }
    return id;
}

/** Reduce gradient from broadcastShape back to targetShape by summing. */
function rawUnbroadcast(gradData: Float32Array, gradShape: number[], targetShape: number[]): Float32Array {
    if (shapeEqual(gradShape, targetShape)) return new Float32Array(gradData);
    const targetSize = shapeSize(targetShape);
    if (targetSize === 0) return new Float32Array(0);
    const result = new Float32Array(targetSize);
    const gradSize = gradData.length;
    const gradNdim = gradShape.length;
    const idx = new Array(gradNdim).fill(0);
    const dimOffset = gradNdim - targetShape.length;

    for (let i = 0; i < gradSize; i++) {
        let targetFlat = 0;
        let targetStride = 1;
        for (let d = targetShape.length - 1; d >= 0; d--) {
            const gIdx = idx[d + dimOffset];
            const tIdx = targetShape[d] === 1 ? 0 : gIdx;
            targetFlat += tIdx * targetStride;
            targetStride *= targetShape[d];
        }
        result[targetFlat] += gradData[i];

        for (let d = gradNdim - 1; d >= 0; d--) {
            if (++idx[d] < gradShape[d]) break;
            idx[d] = 0;
        }
    }
    return result;
}

function rawBroadcastBinary(
    aData: Float32Array, aShape: number[],
    bData: Float32Array, bShape: number[],
    fn: (a: number, b: number) => number,
): { data: Float32Array; shape: number[] } {
    const outShape = broadcastShapes(aShape, bShape);
    const outSize = shapeSize(outShape);
    const out = new Float32Array(outSize);
    const ndim = outShape.length;
    const aOff = ndim - aShape.length;
    const bOff = ndim - bShape.length;
    const aStrides = computeStrides(aShape);
    const bStrides = computeStrides(bShape);
    const idx = new Array(ndim).fill(0);

    for (let i = 0; i < outSize; i++) {
        let aFlat = 0, bFlat = 0;
        for (let d = 0; d < aShape.length; d++) {
            aFlat += (aShape[d] === 1 ? 0 : idx[d + aOff]) * aStrides[d];
        }
        for (let d = 0; d < bShape.length; d++) {
            bFlat += (bShape[d] === 1 ? 0 : idx[d + bOff]) * bStrides[d];
        }
        out[i] = fn(aData[aFlat], bData[bFlat]);

        for (let d = ndim - 1; d >= 0; d--) {
            if (++idx[d] < outShape[d]) break;
            idx[d] = 0;
        }
    }
    return { data: out, shape: outShape };
}

function rawTransposeLastTwo(data: Float32Array, shape: number[]): { data: Float32Array; shape: number[] } {
    const ndim = shape.length;
    const rows = shape[ndim - 2];
    const cols = shape[ndim - 1];
    const batchDims = shape.slice(0, -2);
    const batchSize = batchDims.reduce((a, b) => a * b, 1) || 1;
    const outShape = [...batchDims, cols, rows];
    const out = new Float32Array(data.length);
    const matSize = rows * cols;
    for (let b = 0; b < batchSize; b++) {
        const off = b * matSize;
        for (let r = 0; r < rows; r++)
            for (let c = 0; c < cols; c++)
                out[off + c * rows + r] = data[off + r * cols + c];
    }
    return { data: out, shape: outShape };
}

function getBatchIndex(flatIdx: number, outBatch: number[], srcBatch: number[]): number {
    if (srcBatch.length === 0) return 0;
    const ndim = outBatch.length;
    const idx = new Array(ndim);
    let rem = flatIdx;
    for (let i = ndim - 1; i >= 0; i--) {
        idx[i] = rem % outBatch[i];
        rem = (rem / outBatch[i]) | 0;
    }
    const srcNdim = srcBatch.length;
    const offset = ndim - srcNdim;
    let srcFlat = 0;
    let srcStride = 1;
    for (let i = srcNdim - 1; i >= 0; i--) {
        const di = srcBatch[i] === 1 ? 0 : idx[i + offset];
        srcFlat += di * srcStride;
        srcStride *= srcBatch[i];
    }
    return srcFlat;
}

function rawMatmul(aData: Float32Array, aShape: number[], bData: Float32Array, bShape: number[]): { data: Float32Array; shape: number[] } {
    const aNdim = aShape.length;
    const bNdim = bShape.length;
    const M = aShape[aNdim - 2];
    const K = aShape[aNdim - 1];
    const N = bShape[bNdim - 1];
    const aBatch = aNdim > 2 ? aShape.slice(0, -2) : [];
    const bBatch = bNdim > 2 ? bShape.slice(0, -2) : [];
    const outBatch = broadcastShapes(aBatch, bBatch);
    const outShape = [...outBatch, M, N];
    const batchSize = outBatch.length > 0 ? outBatch.reduce((a, b) => a * b, 1) : 1;
    const outData = new Float32Array(batchSize * M * N);

    for (let batch = 0; batch < batchSize; batch++) {
        const aIdx = getBatchIndex(batch, outBatch, aBatch);
        const bIdx = getBatchIndex(batch, outBatch, bBatch);
        const aOff = aIdx * M * K;
        const bOff = bIdx * K * N;
        const oOff = batch * M * N;
        for (let m = 0; m < M; m++) {
            const rowOff = aOff + m * K;
            for (let k = 0; k < K; k++) {
                const a_mk = aData[rowOff + k];
                const bRowOff = bOff + k * N;
                const oRowOff = oOff + m * N;
                for (let n = 0; n < N; n++) {
                    outData[oRowOff + n] += a_mk * bData[bRowOff + n];
                }
            }
        }
    }
    return { data: outData, shape: outShape };
}

function rawSumAlongDim(data: Float32Array, shape: number[], dim: number): { data: Float32Array; shape: number[] } {
    const ndim = shape.length;
    dim = normalizeDim(dim, ndim);
    const outShape = shape.map((s, i) => (i === dim ? 1 : s));
    const outSize = shapeSize(outShape);
    const out = new Float32Array(outSize);
    const size = data.length;
    const idx = new Array(ndim).fill(0);
    const outStrides = computeStrides(outShape);

    for (let i = 0; i < size; i++) {
        let oFlat = 0;
        for (let d = 0; d < ndim; d++) {
            oFlat += (d === dim ? 0 : idx[d]) * outStrides[d];
        }
        out[oFlat] += data[i];
        for (let d = ndim - 1; d >= 0; d--) {
            if (++idx[d] < shape[d]) break;
            idx[d] = 0;
        }
    }
    return { data: out, shape: outShape };
}

// ===================================================================
// Creation
// ===================================================================

export function fromFloat32(data: Float32Array, shape: number[]): number {
    return store.alloc(new Float32Array(data), shape);
}

export function zeros(shape: number[]): number {
    return store.alloc(new Float32Array(shapeSize(shape)), shape);
}

export function ones(shape: number[]): number {
    return store.alloc(new Float32Array(shapeSize(shape)).fill(1), shape);
}

export function randTensor(shape: number[]): number {
    const size = shapeSize(shape);
    const d = new Float32Array(size);
    for (let i = 0; i < size; i++) d[i] = Math.random();
    return store.alloc(d, shape);
}

export function randnTensor(shape: number[]): number {
    const size = shapeSize(shape);
    const d = new Float32Array(size);
    for (let i = 0; i < size; i++) {
        const u1 = Math.random() || 1e-10;
        const u2 = Math.random();
        d[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    return store.alloc(d, shape);
}

// ===================================================================
// Data access
// ===================================================================

export function tensorShape(id: number): number[] {
    return [...store.get(id).shape];
}

export function toFloat32(id: number): Float32Array {
    return new Float32Array(store.getContiguousData(id));
}

export function getScalar(id: number): number {
    return store.getContiguousData(id)[0];
}

export function freeTensor(id: number): void {
    store.free(id);
}

// ===================================================================
// Gradient management
// ===================================================================

export function setRequiresGrad(id: number, requires: boolean): void {
    store.get(id).requiresGrad = requires;
}

export function getGrad(id: number): number | null {
    return store.get(id).gradId;
}

export function backward(id: number): void {
    tape.backward(id);
}

export function noGradStart(): void {
    tape.enabled = false;
}

export function noGradEnd(): void {
    tape.enabled = true;
}

// ===================================================================
// Elementwise binary ops
// ===================================================================

export function add(aId: number, bId: number): number {
    const aData = store.getContiguousData(aId);
    const aShape = [...store.get(aId).shape];
    const bData = store.getContiguousData(bId);
    const bShape = [...store.get(bId).shape];
    const { data, shape } = rawBroadcastBinary(aData, aShape, bData, bShape, (a, b) => a + b);
    const outId = allocResult(data, shape, aId, bId);

    if (tape.enabled && needsGrad(aId, bId)) {
        tape.record({
            outputId: outId,
            inputIds: [aId, bId],
            backward: (gId) => {
                const gData = store.getContiguousData(gId);
                const gShape = [...store.get(gId).shape];
                const ga = rawUnbroadcast(gData, gShape, aShape);
                const gb = rawUnbroadcast(gData, gShape, bShape);
                return [store.alloc(ga, aShape), store.alloc(gb, bShape)];
            },
        });
    }
    return outId;
}

export function sub(aId: number, bId: number): number {
    const aData = store.getContiguousData(aId);
    const aShape = [...store.get(aId).shape];
    const bData = store.getContiguousData(bId);
    const bShape = [...store.get(bId).shape];
    const { data, shape } = rawBroadcastBinary(aData, aShape, bData, bShape, (a, b) => a - b);
    const outId = allocResult(data, shape, aId, bId);

    if (tape.enabled && needsGrad(aId, bId)) {
        tape.record({
            outputId: outId,
            inputIds: [aId, bId],
            backward: (gId) => {
                const gData = store.getContiguousData(gId);
                const gShape = [...store.get(gId).shape];
                const ga = rawUnbroadcast(gData, gShape, aShape);
                const gbFull = new Float32Array(gData.length);
                for (let i = 0; i < gData.length; i++) gbFull[i] = -gData[i];
                const gb = rawUnbroadcast(gbFull, gShape, bShape);
                return [store.alloc(ga, aShape), store.alloc(gb, bShape)];
            },
        });
    }
    return outId;
}

export function mul(aId: number, bId: number): number {
    const aData = store.getContiguousData(aId);
    const aShape = [...store.get(aId).shape];
    const bData = store.getContiguousData(bId);
    const bShape = [...store.get(bId).shape];
    const { data, shape } = rawBroadcastBinary(aData, aShape, bData, bShape, (a, b) => a * b);
    const outId = allocResult(data, shape, aId, bId);

    if (tape.enabled && needsGrad(aId, bId)) {
        const savedA = new Float32Array(aData);
        const savedB = new Float32Array(bData);
        tape.record({
            outputId: outId,
            inputIds: [aId, bId],
            backward: (gId) => {
                const gData = store.getContiguousData(gId);
                const gShape = [...store.get(gId).shape];
                const { data: gaFull } = rawBroadcastBinary(gData, gShape, savedB, bShape, (g, b) => g * b);
                const { data: gbFull } = rawBroadcastBinary(gData, gShape, savedA, aShape, (g, a) => g * a);
                const ga = rawUnbroadcast(gaFull, gShape, aShape);
                const gb = rawUnbroadcast(gbFull, gShape, bShape);
                return [store.alloc(ga, aShape), store.alloc(gb, bShape)];
            },
        });
    }
    return outId;
}

export function div(aId: number, bId: number): number {
    const aData = store.getContiguousData(aId);
    const aShape = [...store.get(aId).shape];
    const bData = store.getContiguousData(bId);
    const bShape = [...store.get(bId).shape];
    const { data, shape } = rawBroadcastBinary(aData, aShape, bData, bShape, (a, b) => a / b);
    const outId = allocResult(data, shape, aId, bId);

    if (tape.enabled && needsGrad(aId, bId)) {
        const savedA = new Float32Array(aData);
        const savedB = new Float32Array(bData);
        tape.record({
            outputId: outId,
            inputIds: [aId, bId],
            backward: (gId) => {
                const gData = store.getContiguousData(gId);
                const gShape = [...store.get(gId).shape];
                const { data: gaFull } = rawBroadcastBinary(gData, gShape, savedB, bShape, (g, b) => g / b);
                const { data: gbFull } = rawBroadcastBinary(
                    gData, gShape, savedA, aShape,
                    (g, _a) => g,
                );
                // grad_b = -grad * a / b^2
                const gbFull2 = new Float32Array(gShape.reduce((x, y) => x * y, 1));
                const outSize = gbFull2.length;
                const outNdim = gShape.length;
                const aOff = outNdim - aShape.length;
                const bOff = outNdim - bShape.length;
                const aStrides = computeStrides(aShape);
                const bStrides = computeStrides(bShape);
                const idx = new Array(outNdim).fill(0);
                for (let i = 0; i < outSize; i++) {
                    let aFlat = 0, bFlat = 0;
                    for (let d = 0; d < aShape.length; d++) aFlat += (aShape[d] === 1 ? 0 : idx[d + aOff]) * aStrides[d];
                    for (let d = 0; d < bShape.length; d++) bFlat += (bShape[d] === 1 ? 0 : idx[d + bOff]) * bStrides[d];
                    gbFull2[i] = -gData[i] * savedA[aFlat] / (savedB[bFlat] * savedB[bFlat]);
                    for (let d = outNdim - 1; d >= 0; d--) { if (++idx[d] < gShape[d]) break; idx[d] = 0; }
                }
                const ga = rawUnbroadcast(gaFull, gShape, aShape);
                const gb = rawUnbroadcast(gbFull2, gShape, bShape);
                return [store.alloc(ga, aShape), store.alloc(gb, bShape)];
            },
        });
    }
    return outId;
}

// ===================================================================
// Elementwise unary ops
// ===================================================================

export function neg(id: number): number {
    const d = store.getContiguousData(id);
    const s = [...store.get(id).shape];
    const out = new Float32Array(d.length);
    for (let i = 0; i < d.length; i++) out[i] = -d[i];
    const outId = allocResult(out, s, id);

    if (tape.enabled && needsGrad(id)) {
        tape.record({
            outputId: outId, inputIds: [id],
            backward: (gId) => {
                const g = store.getContiguousData(gId);
                const r = new Float32Array(g.length);
                for (let i = 0; i < g.length; i++) r[i] = -g[i];
                return [store.alloc(r, s)];
            },
        });
    }
    return outId;
}

export function mulScalar(id: number, scalar: number): number {
    const d = store.getContiguousData(id);
    const s = [...store.get(id).shape];
    const out = new Float32Array(d.length);
    for (let i = 0; i < d.length; i++) out[i] = d[i] * scalar;
    const outId = allocResult(out, s, id);

    if (tape.enabled && needsGrad(id)) {
        tape.record({
            outputId: outId, inputIds: [id],
            backward: (gId) => {
                const g = store.getContiguousData(gId);
                const r = new Float32Array(g.length);
                for (let i = 0; i < g.length; i++) r[i] = g[i] * scalar;
                return [store.alloc(r, s)];
            },
        });
    }
    return outId;
}

export function expOp(id: number): number {
    const d = store.getContiguousData(id);
    const s = [...store.get(id).shape];
    const out = new Float32Array(d.length);
    for (let i = 0; i < d.length; i++) out[i] = Math.exp(d[i]);
    const outId = allocResult(out, s, id);

    if (tape.enabled && needsGrad(id)) {
        const savedOut = new Float32Array(out);
        tape.record({
            outputId: outId, inputIds: [id],
            backward: (gId) => {
                const g = store.getContiguousData(gId);
                const r = new Float32Array(g.length);
                for (let i = 0; i < g.length; i++) r[i] = g[i] * savedOut[i];
                return [store.alloc(r, s)];
            },
        });
    }
    return outId;
}

export function logOp(id: number): number {
    const d = store.getContiguousData(id);
    const s = [...store.get(id).shape];
    const out = new Float32Array(d.length);
    for (let i = 0; i < d.length; i++) out[i] = Math.log(d[i]);
    const outId = allocResult(out, s, id);

    if (tape.enabled && needsGrad(id)) {
        const savedInput = new Float32Array(d);
        tape.record({
            outputId: outId, inputIds: [id],
            backward: (gId) => {
                const g = store.getContiguousData(gId);
                const r = new Float32Array(g.length);
                for (let i = 0; i < g.length; i++) r[i] = g[i] / savedInput[i];
                return [store.alloc(r, s)];
            },
        });
    }
    return outId;
}

export function powOp(id: number, exponent: number): number {
    const d = store.getContiguousData(id);
    const s = [...store.get(id).shape];
    const out = new Float32Array(d.length);
    for (let i = 0; i < d.length; i++) out[i] = Math.pow(d[i], exponent);
    const outId = allocResult(out, s, id);

    if (tape.enabled && needsGrad(id)) {
        const savedInput = new Float32Array(d);
        tape.record({
            outputId: outId, inputIds: [id],
            backward: (gId) => {
                const g = store.getContiguousData(gId);
                const r = new Float32Array(g.length);
                for (let i = 0; i < g.length; i++)
                    r[i] = g[i] * exponent * Math.pow(savedInput[i], exponent - 1);
                return [store.alloc(r, s)];
            },
        });
    }
    return outId;
}

// ===================================================================
// Activations
// ===================================================================

export function relu(id: number): number {
    const d = store.getContiguousData(id);
    const s = [...store.get(id).shape];
    const out = new Float32Array(d.length);
    for (let i = 0; i < d.length; i++) out[i] = d[i] > 0 ? d[i] : 0;
    const outId = allocResult(out, s, id);

    if (tape.enabled && needsGrad(id)) {
        const savedInput = new Float32Array(d);
        tape.record({
            outputId: outId, inputIds: [id],
            backward: (gId) => {
                const g = store.getContiguousData(gId);
                const r = new Float32Array(g.length);
                for (let i = 0; i < g.length; i++) r[i] = savedInput[i] > 0 ? g[i] : 0;
                return [store.alloc(r, s)];
            },
        });
    }
    return outId;
}

export function sigmoid(id: number): number {
    const d = store.getContiguousData(id);
    const s = [...store.get(id).shape];
    const out = new Float32Array(d.length);
    for (let i = 0; i < d.length; i++) {
        out[i] = d[i] >= 0 ? 1 / (1 + Math.exp(-d[i])) : Math.exp(d[i]) / (1 + Math.exp(d[i]));
    }
    const outId = allocResult(out, s, id);

    if (tape.enabled && needsGrad(id)) {
        const savedOut = new Float32Array(out);
        tape.record({
            outputId: outId, inputIds: [id],
            backward: (gId) => {
                const g = store.getContiguousData(gId);
                const r = new Float32Array(g.length);
                for (let i = 0; i < g.length; i++) r[i] = g[i] * savedOut[i] * (1 - savedOut[i]);
                return [store.alloc(r, s)];
            },
        });
    }
    return outId;
}

export function gelu(id: number): number {
    const d = store.getContiguousData(id);
    const s = [...store.get(id).shape];
    const SQRT_2_PI = Math.sqrt(2 / Math.PI);
    const out = new Float32Array(d.length);
    for (let i = 0; i < d.length; i++) {
        const x = d[i];
        const cdf = 0.5 * (1 + Math.tanh(SQRT_2_PI * (x + 0.044715 * x * x * x)));
        out[i] = x * cdf;
    }
    const outId = allocResult(out, s, id);

    if (tape.enabled && needsGrad(id)) {
        const savedInput = new Float32Array(d);
        tape.record({
            outputId: outId, inputIds: [id],
            backward: (gId) => {
                const g = store.getContiguousData(gId);
                const r = new Float32Array(g.length);
                for (let i = 0; i < g.length; i++) {
                    const x = savedInput[i];
                    const inner = SQRT_2_PI * (x + 0.044715 * x * x * x);
                    const tanhVal = Math.tanh(inner);
                    const cdf = 0.5 * (1 + tanhVal);
                    const sech2 = 1 - tanhVal * tanhVal;
                    const dInner = SQRT_2_PI * (1 + 3 * 0.044715 * x * x);
                    r[i] = g[i] * (cdf + x * 0.5 * sech2 * dInner);
                }
                return [store.alloc(r, s)];
            },
        });
    }
    return outId;
}

// ===================================================================
// Reductions
// ===================================================================

export function sumAll(id: number): number {
    const d = store.getContiguousData(id);
    const s = [...store.get(id).shape];
    let total = 0;
    for (let i = 0; i < d.length; i++) total += d[i];
    const outId = allocResult(new Float32Array([total]), [1], id);

    if (tape.enabled && needsGrad(id)) {
        const inputSize = d.length;
        tape.record({
            outputId: outId, inputIds: [id],
            backward: (gId) => {
                const gVal = store.getContiguousData(gId)[0];
                const r = new Float32Array(inputSize).fill(gVal);
                return [store.alloc(r, s)];
            },
        });
    }
    return outId;
}

export function sumOp(id: number, dim: number): number {
    const d = store.getContiguousData(id);
    const s = [...store.get(id).shape];
    const { data, shape } = rawSumAlongDim(d, s, dim);
    const outId = allocResult(data, shape, id);

    if (tape.enabled && needsGrad(id)) {
        tape.record({
            outputId: outId, inputIds: [id],
            backward: (gId) => {
                const gData = store.getContiguousData(gId);
                const gShape = [...store.get(gId).shape];
                // Broadcast grad back to input shape
                const r = new Float32Array(shapeSize(s));
                const ndim = s.length;
                const realDim = normalizeDim(dim, ndim);
                const idx = new Array(ndim).fill(0);
                const gStrides = computeStrides(gShape);
                for (let i = 0; i < r.length; i++) {
                    let gFlat = 0;
                    for (let dd = 0; dd < ndim; dd++) {
                        gFlat += (dd === realDim ? 0 : idx[dd]) * gStrides[dd];
                    }
                    r[i] = gData[gFlat];
                    for (let dd = ndim - 1; dd >= 0; dd--) { if (++idx[dd] < s[dd]) break; idx[dd] = 0; }
                }
                return [store.alloc(r, s)];
            },
        });
    }
    return outId;
}

export function meanAll(id: number): number {
    const d = store.getContiguousData(id);
    const s = [...store.get(id).shape];
    const n = d.length;
    let total = 0;
    for (let i = 0; i < n; i++) total += d[i];
    const outId = allocResult(new Float32Array([total / n]), [1], id);

    if (tape.enabled && needsGrad(id)) {
        tape.record({
            outputId: outId, inputIds: [id],
            backward: (gId) => {
                const gVal = store.getContiguousData(gId)[0];
                const r = new Float32Array(n).fill(gVal / n);
                return [store.alloc(r, s)];
            },
        });
    }
    return outId;
}

export function meanOp(id: number, dim: number): number {
    const d = store.getContiguousData(id);
    const s = [...store.get(id).shape];
    const ndim = s.length;
    const realDim = normalizeDim(dim, ndim);
    const dimSize = s[realDim];
    const { data, shape } = rawSumAlongDim(d, s, dim);
    for (let i = 0; i < data.length; i++) data[i] /= dimSize;
    const outId = allocResult(data, shape, id);

    if (tape.enabled && needsGrad(id)) {
        tape.record({
            outputId: outId, inputIds: [id],
            backward: (gId) => {
                const gData = store.getContiguousData(gId);
                const gShape = [...store.get(gId).shape];
                const r = new Float32Array(shapeSize(s));
                const idx = new Array(ndim).fill(0);
                const gStrides = computeStrides(gShape);
                for (let i = 0; i < r.length; i++) {
                    let gFlat = 0;
                    for (let dd = 0; dd < ndim; dd++) {
                        gFlat += (dd === realDim ? 0 : idx[dd]) * gStrides[dd];
                    }
                    r[i] = gData[gFlat] / dimSize;
                    for (let dd = ndim - 1; dd >= 0; dd--) { if (++idx[dd] < s[dd]) break; idx[dd] = 0; }
                }
                return [store.alloc(r, s)];
            },
        });
    }
    return outId;
}

export function maxOp(id: number, dim: number): number {
    const d = store.getContiguousData(id);
    const s = [...store.get(id).shape];
    const ndim = s.length;
    const realDim = normalizeDim(dim, ndim);
    const outShape = s.map((v, i) => (i === realDim ? 1 : v));
    const outSize = shapeSize(outShape);
    const out = new Float32Array(outSize).fill(-Infinity);
    const argmax = new Int32Array(outSize).fill(0);

    const idx = new Array(ndim).fill(0);
    const outStrides = computeStrides(outShape);
    const inSize = d.length;

    for (let i = 0; i < inSize; i++) {
        let oFlat = 0;
        for (let dd = 0; dd < ndim; dd++) oFlat += (dd === realDim ? 0 : idx[dd]) * outStrides[dd];
        if (d[i] > out[oFlat]) {
            out[oFlat] = d[i];
            argmax[oFlat] = idx[realDim];
        }
        for (let dd = ndim - 1; dd >= 0; dd--) { if (++idx[dd] < s[dd]) break; idx[dd] = 0; }
    }
    const outId = allocResult(out, outShape, id);

    if (tape.enabled && needsGrad(id)) {
        const savedArgmax = new Int32Array(argmax);
        tape.record({
            outputId: outId, inputIds: [id],
            backward: (gId) => {
                const gData = store.getContiguousData(gId);
                const r = new Float32Array(shapeSize(s));
                const idx2 = new Array(ndim).fill(0);
                for (let i = 0; i < r.length; i++) {
                    let oFlat = 0;
                    for (let dd = 0; dd < ndim; dd++) oFlat += (dd === realDim ? 0 : idx2[dd]) * outStrides[dd];
                    r[i] = idx2[realDim] === savedArgmax[oFlat] ? gData[oFlat] : 0;
                    for (let dd = ndim - 1; dd >= 0; dd--) { if (++idx2[dd] < s[dd]) break; idx2[dd] = 0; }
                }
                return [store.alloc(r, s)];
            },
        });
    }
    return outId;
}

// ===================================================================
// Comparison (no gradient)
// ===================================================================

export function lt(aId: number, bId: number): number {
    const aData = store.getContiguousData(aId);
    const aShape = [...store.get(aId).shape];
    const bData = store.getContiguousData(bId);
    const bShape = [...store.get(bId).shape];
    const { data, shape } = rawBroadcastBinary(aData, aShape, bData, bShape, (a, b) => (a < b ? 1 : 0));
    return store.alloc(data, shape);
}

export function eqOp(aId: number, bId: number): number {
    const aData = store.getContiguousData(aId);
    const aShape = [...store.get(aId).shape];
    const bData = store.getContiguousData(bId);
    const bShape = [...store.get(bId).shape];
    const { data, shape } = rawBroadcastBinary(aData, aShape, bData, bShape, (a, b) => (a === b ? 1 : 0));
    return store.alloc(data, shape);
}

export function gt(aId: number, bId: number): number {
    const aData = store.getContiguousData(aId);
    const aShape = [...store.get(aId).shape];
    const bData = store.getContiguousData(bId);
    const bShape = [...store.get(bId).shape];
    const { data, shape } = rawBroadcastBinary(aData, aShape, bData, bShape, (a, b) => (a > b ? 1 : 0));
    return store.alloc(data, shape);
}

export function isClose(aId: number, bId: number, tol: number): number {
    const aData = store.getContiguousData(aId);
    const aShape = [...store.get(aId).shape];
    const bData = store.getContiguousData(bId);
    const bShape = [...store.get(bId).shape];
    const { data, shape } = rawBroadcastBinary(aData, aShape, bData, bShape, (a, b) => (Math.abs(a - b) < tol ? 1 : 0));
    return store.alloc(data, shape);
}

// ===================================================================
// Layout
// ===================================================================

export function view(id: number, newShape: number[]): number {
    const entry = store.get(id);
    const oldSize = shapeSize(entry.shape);
    let inferIdx = -1;
    let knownSize = 1;
    for (let i = 0; i < newShape.length; i++) {
        if (newShape[i] === -1) { inferIdx = i; }
        else { knownSize *= newShape[i]; }
    }
    if (inferIdx >= 0) {
        newShape = [...newShape];
        newShape[inferIdx] = oldSize / knownSize;
    }

    const data = store.getContiguousData(id);
    const outId = store.alloc(data, newShape);
    const se = store.get(id);
    const oe = store.get(outId);
    oe.requiresGrad = se.requiresGrad;
    // Share data reference for zero-copy view
    oe.data = data;

    if (tape.enabled && needsGrad(id)) {
        const origShape = [...entry.shape];
        tape.record({
            outputId: outId, inputIds: [id],
            backward: (gId) => {
                const gData = store.getContiguousData(gId);
                return [store.alloc(new Float32Array(gData), origShape)];
            },
        });
    }
    return outId;
}

export function permute(id: number, dims: number[]): number {
    const entry = store.get(id);
    const ndim = entry.shape.length;
    const newShape = new Array(ndim);
    const newStrides = new Array(ndim);
    for (let i = 0; i < ndim; i++) {
        newShape[i] = entry.shape[dims[i]];
        newStrides[i] = entry.strides[dims[i]];
    }
    const outId = store.alloc(entry.data, newShape, newStrides, entry.offset);
    const oe = store.get(outId);
    oe.requiresGrad = entry.requiresGrad;

    if (tape.enabled && needsGrad(id)) {
        const inverseDims = new Array(ndim);
        for (let i = 0; i < ndim; i++) inverseDims[dims[i]] = i;
        tape.record({
            outputId: outId, inputIds: [id],
            backward: (gId) => {
                const gEntry = store.get(gId);
                const gNdim = gEntry.shape.length;
                const invShape = new Array(gNdim);
                const invStrides = new Array(gNdim);
                for (let i = 0; i < gNdim; i++) {
                    invShape[i] = gEntry.shape[inverseDims[i]];
                    invStrides[i] = gEntry.strides[inverseDims[i]];
                }
                const gData = store.getContiguousData(gId);
                // Make the inverse-permuted data contiguous
                const size = shapeSize(invShape);
                const out = new Float32Array(size);
                const gContStrides = computeStrides(gEntry.shape);
                const idx = new Array(gNdim).fill(0);
                for (let i = 0; i < gData.length; i++) {
                    // idx is in the grad's layout; map to inverse layout
                    const invIdx = new Array(gNdim);
                    for (let d = 0; d < gNdim; d++) invIdx[inverseDims[d]] = idx[d];
                    let flat = 0;
                    const invContStrides = computeStrides(invShape);
                    for (let d = 0; d < gNdim; d++) flat += invIdx[d] * invContStrides[d];
                    out[flat] = gData[i];
                    for (let d = gNdim - 1; d >= 0; d--) { if (++idx[d] < gEntry.shape[d]) break; idx[d] = 0; }
                }
                return [store.alloc(out, invShape)];
            },
        });
    }
    return outId;
}

export function contiguous(id: number): number {
    const entry = store.get(id);
    if (entry.offset === 0 && isContiguous(entry.shape, entry.strides)) return id;
    const data = store.getContiguousData(id);
    const outId = store.alloc(new Float32Array(data), [...entry.shape]);
    store.get(outId).requiresGrad = entry.requiresGrad;

    if (tape.enabled && needsGrad(id)) {
        tape.record({
            outputId: outId, inputIds: [id],
            backward: (gId) => [gId],
        });
    }
    return outId;
}

// ===================================================================
// Matmul
// ===================================================================

export function matmul(aId: number, bId: number): number {
    const aData = store.getContiguousData(aId);
    const aShape = [...store.get(aId).shape];
    const bData = store.getContiguousData(bId);
    const bShape = [...store.get(bId).shape];
    const { data, shape } = rawMatmul(aData, aShape, bData, bShape);
    const outId = allocResult(data, shape, aId, bId);

    if (tape.enabled && needsGrad(aId, bId)) {
        const savedA = new Float32Array(aData);
        const savedB = new Float32Array(bData);
        tape.record({
            outputId: outId,
            inputIds: [aId, bId],
            backward: (gId) => {
                const gData = store.getContiguousData(gId);
                const gShape = [...store.get(gId).shape];
                // grad_A = gradOut @ B^T
                const { data: bT, shape: bTShape } = rawTransposeLastTwo(savedB, bShape);
                const { data: gaFull, shape: gaFullShape } = rawMatmul(gData, gShape, bT, bTShape);
                const gaData = rawUnbroadcast(gaFull, gaFullShape, aShape);
                // grad_B = A^T @ gradOut
                const { data: aT, shape: aTShape } = rawTransposeLastTwo(savedA, aShape);
                const { data: gbFull, shape: gbFullShape } = rawMatmul(aT, aTShape, gData, gShape);
                const gbData = rawUnbroadcast(gbFull, gbFullShape, bShape);
                return [store.alloc(gaData, aShape), store.alloc(gbData, bShape)];
            },
        });
    }
    return outId;
}

// ===================================================================
// NN ops
// ===================================================================

export function softmaxOp(id: number, dim: number): number {
    const d = store.getContiguousData(id);
    const s = [...store.get(id).shape];
    const ndim = s.length;
    const realDim = normalizeDim(dim, ndim);
    const dimSize = s[realDim];
    const outerSize = shapeSize(s) / dimSize;

    const out = new Float32Array(d.length);
    const outerStride = computeStrides(s);

    // Compute sizes for iterating along the reduction dimension
    let innerSize = 1;
    for (let i = realDim + 1; i < ndim; i++) innerSize *= s[i];

    for (let outer = 0; outer < outerSize; outer++) {
        const outerHigh = Math.floor(outer / innerSize);
        const outerLow = outer % innerSize;

        // Find max for numerical stability
        let max = -Infinity;
        for (let j = 0; j < dimSize; j++) {
            const idx = outerHigh * dimSize * innerSize + j * innerSize + outerLow;
            if (d[idx] > max) max = d[idx];
        }
        let sum = 0;
        for (let j = 0; j < dimSize; j++) {
            const idx = outerHigh * dimSize * innerSize + j * innerSize + outerLow;
            const e = Math.exp(d[idx] - max);
            out[idx] = e;
            sum += e;
        }
        for (let j = 0; j < dimSize; j++) {
            const idx = outerHigh * dimSize * innerSize + j * innerSize + outerLow;
            out[idx] /= sum;
        }
    }
    const outId = allocResult(out, s, id);

    if (tape.enabled && needsGrad(id)) {
        const savedOut = new Float32Array(out);
        tape.record({
            outputId: outId, inputIds: [id],
            backward: (gId) => {
                const g = store.getContiguousData(gId);
                const r = new Float32Array(g.length);
                for (let outer = 0; outer < outerSize; outer++) {
                    const outerHigh = Math.floor(outer / innerSize);
                    const outerLow = outer % innerSize;
                    let dot = 0;
                    for (let j = 0; j < dimSize; j++) {
                        const idx = outerHigh * dimSize * innerSize + j * innerSize + outerLow;
                        dot += g[idx] * savedOut[idx];
                    }
                    for (let j = 0; j < dimSize; j++) {
                        const idx = outerHigh * dimSize * innerSize + j * innerSize + outerLow;
                        r[idx] = savedOut[idx] * (g[idx] - dot);
                    }
                }
                return [store.alloc(r, s)];
            },
        });
    }
    return outId;
}

export function embeddingForward(weightId: number, flatIndices: number[], batch: number, seqLen: number): number {
    const wData = store.getContiguousData(weightId);
    const wShape = [...store.get(weightId).shape];
    const embedDim = wShape[1];
    const outShape = [batch, seqLen, embedDim];
    const out = new Float32Array(batch * seqLen * embedDim);

    for (let i = 0; i < flatIndices.length; i++) {
        const tokenIdx = flatIndices[i];
        const srcOff = tokenIdx * embedDim;
        const dstOff = i * embedDim;
        for (let j = 0; j < embedDim; j++) out[dstOff + j] = wData[srcOff + j];
    }
    const outId = allocResult(out, outShape, weightId);

    if (tape.enabled && needsGrad(weightId)) {
        const savedIndices = [...flatIndices];
        tape.record({
            outputId: outId, inputIds: [weightId],
            backward: (gId) => {
                const gData = store.getContiguousData(gId);
                const gradW = new Float32Array(shapeSize(wShape));
                for (let i = 0; i < savedIndices.length; i++) {
                    const tokenIdx = savedIndices[i];
                    const srcOff = i * embedDim;
                    const dstOff = tokenIdx * embedDim;
                    for (let j = 0; j < embedDim; j++) gradW[dstOff + j] += gData[srcOff + j];
                }
                return [store.alloc(gradW, wShape)];
            },
        });
    }
    return outId;
}

export function embeddingForwardGpu(): number {
    throw new Error('embeddingForwardGpu is not available in the web backend');
}

export function dropoutOp(id: number, rate: number, training: boolean): number {
    const d = store.getContiguousData(id);
    const s = [...store.get(id).shape];

    if (!training || rate === 0) return id;

    const out = new Float32Array(d.length);
    const mask = new Float32Array(d.length);
    const scale = 1 / (1 - rate);
    for (let i = 0; i < d.length; i++) {
        const keep = Math.random() >= rate ? 1 : 0;
        mask[i] = keep * scale;
        out[i] = d[i] * mask[i];
    }
    const outId = allocResult(out, s, id);

    if (tape.enabled && needsGrad(id)) {
        const savedMask = new Float32Array(mask);
        tape.record({
            outputId: outId, inputIds: [id],
            backward: (gId) => {
                const g = store.getContiguousData(gId);
                const r = new Float32Array(g.length);
                for (let i = 0; i < g.length; i++) r[i] = g[i] * savedMask[i];
                return [store.alloc(r, s)];
            },
        });
    }
    return outId;
}

export function layernormOp(xId: number, gammaId: number, betaId: number, eps: number): number {
    const xData = store.getContiguousData(xId);
    const xShape = [...store.get(xId).shape];
    const gammaData = store.getContiguousData(gammaId);
    const betaData = store.getContiguousData(betaId);
    const ndim = xShape.length;
    const D = xShape[ndim - 1];
    const N = xData.length / D;

    const out = new Float32Array(xData.length);
    const xHat = new Float32Array(xData.length);
    const invStd = new Float32Array(N);

    for (let n = 0; n < N; n++) {
        const off = n * D;
        let mean = 0;
        for (let j = 0; j < D; j++) mean += xData[off + j];
        mean /= D;
        let variance = 0;
        for (let j = 0; j < D; j++) {
            const diff = xData[off + j] - mean;
            variance += diff * diff;
        }
        variance /= D;
        const std = Math.sqrt(variance + eps);
        invStd[n] = 1 / std;
        for (let j = 0; j < D; j++) {
            xHat[off + j] = (xData[off + j] - mean) * invStd[n];
            out[off + j] = gammaData[j] * xHat[off + j] + betaData[j];
        }
    }
    const outId = allocResult(out, xShape, xId, gammaId, betaId);

    if (tape.enabled && needsGrad(xId, gammaId, betaId)) {
        const savedXHat = new Float32Array(xHat);
        const savedInvStd = new Float32Array(invStd);
        const savedGamma = new Float32Array(gammaData);
        const gammaShape = [...store.get(gammaId).shape];
        const betaShape = [...store.get(betaId).shape];
        tape.record({
            outputId: outId, inputIds: [xId, gammaId, betaId],
            backward: (gId) => {
                const gData = store.getContiguousData(gId);
                const gradX = new Float32Array(xData.length);
                const gradGamma = new Float32Array(D);
                const gradBeta = new Float32Array(D);

                for (let n = 0; n < N; n++) {
                    const off = n * D;
                    const iStd = savedInvStd[n];
                    // Accumulate gamma and beta grads
                    let meanG = 0, meanGXhat = 0;
                    for (let j = 0; j < D; j++) {
                        gradGamma[j] += gData[off + j] * savedXHat[off + j];
                        gradBeta[j] += gData[off + j];
                        const g = gData[off + j] * savedGamma[j];
                        meanG += g;
                        meanGXhat += g * savedXHat[off + j];
                    }
                    meanG /= D;
                    meanGXhat /= D;
                    for (let j = 0; j < D; j++) {
                        const g = gData[off + j] * savedGamma[j];
                        gradX[off + j] = iStd * (g - meanG - savedXHat[off + j] * meanGXhat);
                    }
                }
                return [
                    store.alloc(gradX, xShape),
                    store.alloc(gradGamma, gammaShape),
                    store.alloc(gradBeta, betaShape),
                ];
            },
        });
    }
    return outId;
}

export function flashAttention(qId: number, kId: number, vId: number, scale: number, causal: boolean): number {
    const qData = store.getContiguousData(qId);
    const qShape = [...store.get(qId).shape]; // [B, H, T, D]
    const kData = store.getContiguousData(kId);
    const kShape = [...store.get(kId).shape];
    const vData = store.getContiguousData(vId);
    const vShape = [...store.get(vId).shape];

    const B = qShape[0], H = qShape[1], T = qShape[2], D = qShape[3];
    const outShape = [B, H, T, D];
    const out = new Float32Array(B * H * T * D);
    const attnWeights = new Float32Array(B * H * T * T);

    for (let b = 0; b < B; b++) {
        for (let h = 0; h < H; h++) {
            const bhOff = (b * H + h);
            const qOff = bhOff * T * D;
            const kOff = bhOff * T * D;
            const vOff = bhOff * T * D;
            const attnOff = bhOff * T * T;
            const outOff = bhOff * T * D;

            // scores = Q @ K^T * scale
            for (let i = 0; i < T; i++) {
                for (let j = 0; j < T; j++) {
                    if (causal && j > i) {
                        attnWeights[attnOff + i * T + j] = -Infinity;
                        continue;
                    }
                    let dot = 0;
                    for (let d = 0; d < D; d++)
                        dot += qData[qOff + i * D + d] * kData[kOff + j * D + d];
                    attnWeights[attnOff + i * T + j] = dot * scale;
                }
                // softmax over j
                let max = -Infinity;
                for (let j = 0; j < T; j++) {
                    const v = attnWeights[attnOff + i * T + j];
                    if (v > max) max = v;
                }
                let sum = 0;
                for (let j = 0; j < T; j++) {
                    const e = Math.exp(attnWeights[attnOff + i * T + j] - max);
                    attnWeights[attnOff + i * T + j] = e;
                    sum += e;
                }
                for (let j = 0; j < T; j++) attnWeights[attnOff + i * T + j] /= sum;
            }

            // out = attn @ V
            for (let i = 0; i < T; i++) {
                for (let d = 0; d < D; d++) {
                    let val = 0;
                    for (let j = 0; j < T; j++)
                        val += attnWeights[attnOff + i * T + j] * vData[vOff + j * D + d];
                    out[outOff + i * D + d] = val;
                }
            }
        }
    }
    const outId = allocResult(out, outShape, qId, kId, vId);

    if (tape.enabled && needsGrad(qId, kId, vId)) {
        const savedAttn = new Float32Array(attnWeights);
        const savedQ = new Float32Array(qData);
        const savedK = new Float32Array(kData);
        const savedV = new Float32Array(vData);
        tape.record({
            outputId: outId,
            inputIds: [qId, kId, vId],
            backward: (gId) => {
                const gData = store.getContiguousData(gId);
                const gradQ = new Float32Array(B * H * T * D);
                const gradK = new Float32Array(B * H * T * D);
                const gradV = new Float32Array(B * H * T * D);
                const gradAttn = new Float32Array(T * T);

                for (let b = 0; b < B; b++) {
                    for (let h = 0; h < H; h++) {
                        const bhOff = (b * H + h);
                        const qOff = bhOff * T * D;
                        const kOff = bhOff * T * D;
                        const vOff = bhOff * T * D;
                        const attnOff = bhOff * T * T;
                        const outOff = bhOff * T * D;

                        // grad_V = attn^T @ grad_out
                        for (let j = 0; j < T; j++) {
                            for (let d = 0; d < D; d++) {
                                let val = 0;
                                for (let i = 0; i < T; i++)
                                    val += savedAttn[attnOff + i * T + j] * gData[outOff + i * D + d];
                                gradV[vOff + j * D + d] += val;
                            }
                        }

                        // grad_attn = grad_out @ V^T
                        gradAttn.fill(0);
                        for (let i = 0; i < T; i++) {
                            for (let j = 0; j < T; j++) {
                                let val = 0;
                                for (let d = 0; d < D; d++)
                                    val += gData[outOff + i * D + d] * savedV[vOff + j * D + d];
                                gradAttn[i * T + j] = val;
                            }
                        }

                        // softmax backward: grad_scores = attn * (grad_attn - sum(grad_attn * attn, dim=-1))
                        const gradScores = new Float32Array(T * T);
                        for (let i = 0; i < T; i++) {
                            let dot = 0;
                            for (let j = 0; j < T; j++)
                                dot += gradAttn[i * T + j] * savedAttn[attnOff + i * T + j];
                            for (let j = 0; j < T; j++) {
                                let gs = savedAttn[attnOff + i * T + j] * (gradAttn[i * T + j] - dot);
                                if (causal && j > i) gs = 0;
                                gradScores[i * T + j] = gs * scale;
                            }
                        }

                        // grad_Q = grad_scores @ K
                        for (let i = 0; i < T; i++) {
                            for (let d = 0; d < D; d++) {
                                let val = 0;
                                for (let j = 0; j < T; j++)
                                    val += gradScores[i * T + j] * savedK[kOff + j * D + d];
                                gradQ[qOff + i * D + d] += val;
                            }
                        }

                        // grad_K = grad_scores^T @ Q
                        for (let j = 0; j < T; j++) {
                            for (let d = 0; d < D; d++) {
                                let val = 0;
                                for (let i = 0; i < T; i++)
                                    val += gradScores[i * T + j] * savedQ[qOff + i * D + d];
                                gradK[kOff + j * D + d] += val;
                            }
                        }
                    }
                }
                return [
                    store.alloc(gradQ, qShape),
                    store.alloc(gradK, kShape),
                    store.alloc(gradV, vShape),
                ];
            },
        });
    }
    return outId;
}

export function crossEntropyLoss(logitsId: number, flatTargets: number[]): number {
    const logits = store.getContiguousData(logitsId);
    const logitsShape = [...store.get(logitsId).shape]; // [BT, V]
    const BT = logitsShape[0];
    const V = logitsShape[1];

    // Stable log-softmax + NLL
    let totalLoss = 0;
    const probs = new Float32Array(BT * V);
    for (let i = 0; i < BT; i++) {
        const off = i * V;
        let max = -Infinity;
        for (let j = 0; j < V; j++) if (logits[off + j] > max) max = logits[off + j];
        let sum = 0;
        for (let j = 0; j < V; j++) {
            probs[off + j] = Math.exp(logits[off + j] - max);
            sum += probs[off + j];
        }
        for (let j = 0; j < V; j++) probs[off + j] /= sum;
        totalLoss -= Math.log(probs[off + flatTargets[i]] + 1e-10);
    }
    const loss = totalLoss / BT;
    const outId = allocResult(new Float32Array([loss]), [1], logitsId);

    if (tape.enabled && needsGrad(logitsId)) {
        const savedProbs = new Float32Array(probs);
        const savedTargets = [...flatTargets];
        tape.record({
            outputId: outId, inputIds: [logitsId],
            backward: (gId) => {
                const gVal = store.getContiguousData(gId)[0];
                const grad = new Float32Array(BT * V);
                for (let i = 0; i < BT; i++) {
                    const off = i * V;
                    for (let j = 0; j < V; j++) {
                        grad[off + j] = savedProbs[off + j];
                    }
                    grad[off + savedTargets[i]] -= 1;
                    for (let j = 0; j < V; j++) grad[off + j] *= gVal / BT;
                }
                return [store.alloc(grad, logitsShape)];
            },
        });
    }
    return outId;
}

export function crossEntropyLossGpu(): number {
    throw new Error('crossEntropyLossGpu is not available in the web backend');
}

export function residualLayernorm(
    xId: number, residualId: number, gammaId: number, betaId: number, eps: number,
): number {
    // Fused: layerNorm(x + residual, gamma, beta, eps)
    const xData = store.getContiguousData(xId);
    const xShape = [...store.get(xId).shape];
    const resData = store.getContiguousData(residualId);
    const combined = new Float32Array(xData.length);
    for (let i = 0; i < xData.length; i++) combined[i] = xData[i] + resData[i];
    const combinedId = store.alloc(combined, xShape);
    store.get(combinedId).requiresGrad = store.get(xId).requiresGrad || store.get(residualId).requiresGrad;

    // Record add for backward
    if (tape.enabled && needsGrad(xId, residualId)) {
        tape.record({
            outputId: combinedId,
            inputIds: [xId, residualId],
            backward: (gId) => [gId, gId],
        });
    }

    return layernormOp(combinedId, gammaId, betaId, eps);
}

export function biasGelu(xId: number, biasId: number): number {
    // Fused: gelu(x + bias)
    const addedId = add(xId, biasId);
    return gelu(addedId);
}

// ===================================================================
// Convolution
// ===================================================================

export function conv1DForward(inputId: number, weightId: number, stride: number, padding: number): number {
    const inp = store.getContiguousData(inputId);
    const inpShape = [...store.get(inputId).shape]; // [N, Cin, L]
    const w = store.getContiguousData(weightId);
    const wShape = [...store.get(weightId).shape]; // [Cout, Cin, K]

    const N = inpShape[0], Cin = inpShape[1], L = inpShape[2];
    const Cout = wShape[0], K = wShape[2];
    const Lout = Math.floor((L + 2 * padding - K) / stride) + 1;
    const outShape = [N, Cout, Lout];
    const out = new Float32Array(N * Cout * Lout);

    for (let n = 0; n < N; n++) {
        for (let co = 0; co < Cout; co++) {
            for (let lo = 0; lo < Lout; lo++) {
                let val = 0;
                for (let ci = 0; ci < Cin; ci++) {
                    for (let k = 0; k < K; k++) {
                        const li = lo * stride - padding + k;
                        if (li >= 0 && li < L) {
                            val += inp[n * Cin * L + ci * L + li] * w[co * Cin * K + ci * K + k];
                        }
                    }
                }
                out[n * Cout * Lout + co * Lout + lo] = val;
            }
        }
    }
    const outId = allocResult(out, outShape, inputId, weightId);

    if (tape.enabled && needsGrad(inputId, weightId)) {
        const savedInp = new Float32Array(inp);
        const savedW = new Float32Array(w);
        tape.record({
            outputId: outId, inputIds: [inputId, weightId],
            backward: (gId) => {
                const gData = store.getContiguousData(gId);
                const gradInp = new Float32Array(N * Cin * L);
                const gradW = new Float32Array(Cout * Cin * K);

                for (let n = 0; n < N; n++) {
                    for (let co = 0; co < Cout; co++) {
                        for (let lo = 0; lo < Lout; lo++) {
                            const gVal = gData[n * Cout * Lout + co * Lout + lo];
                            for (let ci = 0; ci < Cin; ci++) {
                                for (let k = 0; k < K; k++) {
                                    const li = lo * stride - padding + k;
                                    if (li >= 0 && li < L) {
                                        gradInp[n * Cin * L + ci * L + li] += gVal * savedW[co * Cin * K + ci * K + k];
                                        gradW[co * Cin * K + ci * K + k] += gVal * savedInp[n * Cin * L + ci * L + li];
                                    }
                                }
                            }
                        }
                    }
                }
                return [store.alloc(gradInp, inpShape), store.alloc(gradW, wShape)];
            },
        });
    }
    return outId;
}

export function conv2DForward(inputId: number, weightId: number, stride: number, padding: number): number {
    const inp = store.getContiguousData(inputId);
    const inpShape = [...store.get(inputId).shape]; // [N, Cin, H, W]
    const w = store.getContiguousData(weightId);
    const wShape = [...store.get(weightId).shape]; // [Cout, Cin, KH, KW]

    const N = inpShape[0], Cin = inpShape[1], H = inpShape[2], W = inpShape[3];
    const Cout = wShape[0], KH = wShape[2], KW = wShape[3];
    const Hout = Math.floor((H + 2 * padding - KH) / stride) + 1;
    const Wout = Math.floor((W + 2 * padding - KW) / stride) + 1;
    const outShape = [N, Cout, Hout, Wout];
    const out = new Float32Array(N * Cout * Hout * Wout);

    for (let n = 0; n < N; n++) {
        for (let co = 0; co < Cout; co++) {
            for (let ho = 0; ho < Hout; ho++) {
                for (let wo = 0; wo < Wout; wo++) {
                    let val = 0;
                    for (let ci = 0; ci < Cin; ci++) {
                        for (let kh = 0; kh < KH; kh++) {
                            for (let kw = 0; kw < KW; kw++) {
                                const hi = ho * stride - padding + kh;
                                const wi = wo * stride - padding + kw;
                                if (hi >= 0 && hi < H && wi >= 0 && wi < W) {
                                    val += inp[n * Cin * H * W + ci * H * W + hi * W + wi]
                                         * w[co * Cin * KH * KW + ci * KH * KW + kh * KW + kw];
                                }
                            }
                        }
                    }
                    out[n * Cout * Hout * Wout + co * Hout * Wout + ho * Wout + wo] = val;
                }
            }
        }
    }
    const outId = allocResult(out, outShape, inputId, weightId);

    if (tape.enabled && needsGrad(inputId, weightId)) {
        const savedInp = new Float32Array(inp);
        const savedW = new Float32Array(w);
        tape.record({
            outputId: outId, inputIds: [inputId, weightId],
            backward: (gId) => {
                const gData = store.getContiguousData(gId);
                const gradInp = new Float32Array(N * Cin * H * W);
                const gradW = new Float32Array(Cout * Cin * KH * KW);

                for (let n = 0; n < N; n++) {
                    for (let co = 0; co < Cout; co++) {
                        for (let ho = 0; ho < Hout; ho++) {
                            for (let wo = 0; wo < Wout; wo++) {
                                const gVal = gData[n * Cout * Hout * Wout + co * Hout * Wout + ho * Wout + wo];
                                for (let ci = 0; ci < Cin; ci++) {
                                    for (let kh = 0; kh < KH; kh++) {
                                        for (let kw = 0; kw < KW; kw++) {
                                            const hi = ho * stride - padding + kh;
                                            const wi = wo * stride - padding + kw;
                                            if (hi >= 0 && hi < H && wi >= 0 && wi < W) {
                                                gradInp[n * Cin * H * W + ci * H * W + hi * W + wi] +=
                                                    gVal * savedW[co * Cin * KH * KW + ci * KH * KW + kh * KW + kw];
                                                gradW[co * Cin * KH * KW + ci * KH * KW + kh * KW + kw] +=
                                                    gVal * savedInp[n * Cin * H * W + ci * H * W + hi * W + wi];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                return [store.alloc(gradInp, inpShape), store.alloc(gradW, wShape)];
            },
        });
    }
    return outId;
}

// ===================================================================
// Pooling
// ===================================================================

export function avgpool2D(id: number, kh: number, kw: number): number {
    const d = store.getContiguousData(id);
    const s = [...store.get(id).shape]; // [N, C, H, W]
    const N = s[0], C = s[1], H = s[2], W = s[3];
    const Hout = Math.floor(H / kh);
    const Wout = Math.floor(W / kw);
    const outShape = [N, C, Hout, Wout];
    const out = new Float32Array(N * C * Hout * Wout);

    for (let n = 0; n < N; n++) {
        for (let c = 0; c < C; c++) {
            for (let ho = 0; ho < Hout; ho++) {
                for (let wo = 0; wo < Wout; wo++) {
                    let sum = 0;
                    for (let i = 0; i < kh; i++) {
                        for (let j = 0; j < kw; j++) {
                            sum += d[n * C * H * W + c * H * W + (ho * kh + i) * W + (wo * kw + j)];
                        }
                    }
                    out[n * C * Hout * Wout + c * Hout * Wout + ho * Wout + wo] = sum / (kh * kw);
                }
            }
        }
    }
    const outId = allocResult(out, outShape, id);

    if (tape.enabled && needsGrad(id)) {
        tape.record({
            outputId: outId, inputIds: [id],
            backward: (gId) => {
                const gData = store.getContiguousData(gId);
                const gradInp = new Float32Array(shapeSize(s));
                const scale = 1 / (kh * kw);
                for (let n = 0; n < N; n++) {
                    for (let c = 0; c < C; c++) {
                        for (let ho = 0; ho < Hout; ho++) {
                            for (let wo = 0; wo < Wout; wo++) {
                                const gVal = gData[n * C * Hout * Wout + c * Hout * Wout + ho * Wout + wo] * scale;
                                for (let i = 0; i < kh; i++) {
                                    for (let j = 0; j < kw; j++) {
                                        gradInp[n * C * H * W + c * H * W + (ho * kh + i) * W + (wo * kw + j)] += gVal;
                                    }
                                }
                            }
                        }
                    }
                }
                return [store.alloc(gradInp, s)];
            },
        });
    }
    return outId;
}

export function maxpool2D(id: number, kh: number, kw: number): number {
    const d = store.getContiguousData(id);
    const s = [...store.get(id).shape]; // [N, C, H, W]
    const N = s[0], C = s[1], H = s[2], W = s[3];
    const Hout = Math.floor(H / kh);
    const Wout = Math.floor(W / kw);
    const outShape = [N, C, Hout, Wout];
    const outSize = N * C * Hout * Wout;
    const out = new Float32Array(outSize);
    const argmaxH = new Int32Array(outSize);
    const argmaxW = new Int32Array(outSize);

    for (let n = 0; n < N; n++) {
        for (let c = 0; c < C; c++) {
            for (let ho = 0; ho < Hout; ho++) {
                for (let wo = 0; wo < Wout; wo++) {
                    let maxVal = -Infinity;
                    let maxI = 0, maxJ = 0;
                    for (let i = 0; i < kh; i++) {
                        for (let j = 0; j < kw; j++) {
                            const v = d[n * C * H * W + c * H * W + (ho * kh + i) * W + (wo * kw + j)];
                            if (v > maxVal) { maxVal = v; maxI = ho * kh + i; maxJ = wo * kw + j; }
                        }
                    }
                    const oIdx = n * C * Hout * Wout + c * Hout * Wout + ho * Wout + wo;
                    out[oIdx] = maxVal;
                    argmaxH[oIdx] = maxI;
                    argmaxW[oIdx] = maxJ;
                }
            }
        }
    }
    const outId = allocResult(out, outShape, id);

    if (tape.enabled && needsGrad(id)) {
        const savedArgH = new Int32Array(argmaxH);
        const savedArgW = new Int32Array(argmaxW);
        tape.record({
            outputId: outId, inputIds: [id],
            backward: (gId) => {
                const gData = store.getContiguousData(gId);
                const gradInp = new Float32Array(shapeSize(s));
                for (let n = 0; n < N; n++) {
                    for (let c = 0; c < C; c++) {
                        for (let ho = 0; ho < Hout; ho++) {
                            for (let wo = 0; wo < Wout; wo++) {
                                const oIdx = n * C * Hout * Wout + c * Hout * Wout + ho * Wout + wo;
                                const mh = savedArgH[oIdx];
                                const mw = savedArgW[oIdx];
                                gradInp[n * C * H * W + c * H * W + mh * W + mw] += gData[oIdx];
                            }
                        }
                    }
                }
                return [store.alloc(gradInp, s)];
            },
        });
    }
    return outId;
}

// ===================================================================
// Utility
// ===================================================================

export function tile(id: number, reps: number[]): number {
    const d = store.getContiguousData(id);
    const s = [...store.get(id).shape];
    const ndim = s.length;
    const outShape = s.map((v, i) => v * (reps[i] ?? 1));
    const outSize = shapeSize(outShape);
    const out = new Float32Array(outSize);

    const inStrides = computeStrides(s);
    const outStrides = computeStrides(outShape);
    const idx = new Array(ndim).fill(0);

    for (let i = 0; i < outSize; i++) {
        let inFlat = 0;
        for (let dd = 0; dd < ndim; dd++) {
            inFlat += (idx[dd] % s[dd]) * inStrides[dd];
        }
        out[i] = d[inFlat];
        for (let dd = ndim - 1; dd >= 0; dd--) { if (++idx[dd] < outShape[dd]) break; idx[dd] = 0; }
    }
    const outId = allocResult(out, outShape, id);

    if (tape.enabled && needsGrad(id)) {
        tape.record({
            outputId: outId, inputIds: [id],
            backward: (gId) => {
                const gData = store.getContiguousData(gId);
                const grad = new Float32Array(shapeSize(s));
                const idx2 = new Array(ndim).fill(0);
                for (let i = 0; i < outSize; i++) {
                    let inFlat = 0;
                    for (let dd = 0; dd < ndim; dd++) inFlat += (idx2[dd] % s[dd]) * inStrides[dd];
                    grad[inFlat] += gData[i];
                    for (let dd = ndim - 1; dd >= 0; dd--) { if (++idx2[dd] < outShape[dd]) break; idx2[dd] = 0; }
                }
                return [store.alloc(grad, s)];
            },
        });
    }
    return outId;
}

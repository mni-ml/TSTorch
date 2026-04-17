export interface TensorEntry {
    data: Float32Array;
    shape: number[];
    strides: number[];
    offset: number;
    requiresGrad: boolean;
    gradId: number | null;
}

export function computeStrides(shape: number[]): number[] {
    const ndim = shape.length;
    const s = new Array(ndim);
    let stride = 1;
    for (let i = ndim - 1; i >= 0; i--) {
        s[i] = stride;
        stride *= shape[i];
    }
    return s;
}

export function shapeSize(shape: number[]): number {
    let s = 1;
    for (let i = 0; i < shape.length; i++) s *= shape[i];
    return s;
}

export function isContiguous(shape: number[], strides: number[]): boolean {
    let expected = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
        if (shape[i] > 1 && strides[i] !== expected) return false;
        expected *= shape[i];
    }
    return true;
}

export function shapeEqual(a: number[], b: number[]): boolean {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
}

export function normalizeDim(dim: number, ndim: number): number {
    return dim < 0 ? dim + ndim : dim;
}

export function broadcastShapes(a: number[], b: number[]): number[] {
    if (a.length === 0) return [...b];
    if (b.length === 0) return [...a];
    const maxDims = Math.max(a.length, b.length);
    const result = new Array(maxDims);
    for (let i = 0; i < maxDims; i++) {
        const da = i < a.length ? a[a.length - 1 - i] : 1;
        const db = i < b.length ? b[b.length - 1 - i] : 1;
        if (da === db) result[maxDims - 1 - i] = da;
        else if (da === 1) result[maxDims - 1 - i] = db;
        else if (db === 1) result[maxDims - 1 - i] = da;
        else throw new Error(`Cannot broadcast shapes [${a}] and [${b}]`);
    }
    return result;
}

export class TensorStore {
    private entries = new Map<number, TensorEntry>();
    private nextId = 1;

    alloc(data: Float32Array, shape: number[], strides?: number[], offset?: number): number {
        const id = this.nextId++;
        this.entries.set(id, {
            data,
            shape: [...shape],
            strides: strides ? [...strides] : computeStrides(shape),
            offset: offset ?? 0,
            requiresGrad: false,
            gradId: null,
        });
        return id;
    }

    get(id: number): TensorEntry {
        const entry = this.entries.get(id);
        if (!entry) throw new Error(`Tensor ${id} not found`);
        return entry;
    }

    has(id: number): boolean {
        return this.entries.has(id);
    }

    free(id: number): void {
        this.entries.delete(id);
    }

    getContiguousData(id: number): Float32Array {
        const entry = this.get(id);
        if (entry.offset === 0 && isContiguous(entry.shape, entry.strides)) {
            return entry.data;
        }
        return this._makeContiguous(entry);
    }

    private _makeContiguous(entry: TensorEntry): Float32Array {
        const size = shapeSize(entry.shape);
        const out = new Float32Array(size);
        const ndim = entry.shape.length;
        if (ndim === 0) {
            out[0] = entry.data[entry.offset];
            return out;
        }
        const idx = new Array(ndim).fill(0);
        for (let i = 0; i < size; i++) {
            let srcIdx = entry.offset;
            for (let d = 0; d < ndim; d++) srcIdx += idx[d] * entry.strides[d];
            out[i] = entry.data[srcIdx];
            for (let d = ndim - 1; d >= 0; d--) {
                if (++idx[d] < entry.shape[d]) break;
                idx[d] = 0;
            }
        }
        return out;
    }

    ensureGrad(id: number): number {
        const entry = this.get(id);
        if (entry.gradId !== null) return entry.gradId;
        const size = shapeSize(entry.shape);
        const gradId = this.alloc(new Float32Array(size), entry.shape);
        entry.gradId = gradId;
        return gradId;
    }

    accumulateGrad(id: number, gradData: Float32Array): void {
        const gradId = this.ensureGrad(id);
        const ge = this.get(gradId);
        for (let i = 0; i < gradData.length; i++) ge.data[i] += gradData[i];
    }
}

export const store = new TensorStore();

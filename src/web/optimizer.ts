import { store, shapeSize } from './store.js';

const optimState = new Map<number, { m: Float32Array; v: Float32Array }>();

export function clipAndStep(
    ids: number[],
    lr: number,
    beta1: number,
    beta2: number,
    eps: number,
    weightDecay: number,
    t: number,
    maxGradNorm: number,
): number {
    let normSq = 0;
    for (const id of ids) {
        const entry = store.get(id);
        if (entry.gradId === null) continue;
        const gd = store.getContiguousData(entry.gradId);
        for (let i = 0; i < gd.length; i++) normSq += gd[i] * gd[i];
    }
    const gradNorm = Math.sqrt(normSq);
    const clipCoeff = maxGradNorm > 0 && gradNorm > maxGradNorm
        ? maxGradNorm / (gradNorm + 1e-6)
        : 1.0;

    const bc1 = 1 - Math.pow(beta1, t);
    const bc2 = 1 - Math.pow(beta2, t);

    for (const id of ids) {
        const entry = store.get(id);
        if (entry.gradId === null) continue;
        const paramData = store.getContiguousData(id);
        const gradData = store.getContiguousData(entry.gradId);
        const size = paramData.length;

        let state = optimState.get(id);
        if (!state) {
            state = { m: new Float32Array(size), v: new Float32Array(size) };
            optimState.set(id, state);
        }

        for (let i = 0; i < size; i++) {
            const g = gradData[i] * clipCoeff;
            state.m[i] = beta1 * state.m[i] + (1 - beta1) * g;
            state.v[i] = beta2 * state.v[i] + (1 - beta2) * g * g;
            const mHat = state.m[i] / bc1;
            const vHat = state.v[i] / bc2;
            paramData[i] = paramData[i] * (1 - lr * weightDecay) - lr * mHat / (Math.sqrt(vHat) + eps);
        }
    }
    return gradNorm;
}

export function scaleGrads(ids: number[], invScale: number): boolean {
    let foundInf = false;
    for (const id of ids) {
        const entry = store.get(id);
        if (entry.gradId === null) continue;
        const gd = store.getContiguousData(entry.gradId);
        for (let i = 0; i < gd.length; i++) {
            gd[i] *= invScale;
            if (!isFinite(gd[i])) foundInf = true;
        }
    }
    return foundInf;
}

export function zeroGrad(ids: number[]): void {
    for (const id of ids) {
        const entry = store.get(id);
        if (entry.gradId !== null) {
            store.get(entry.gradId).data.fill(0);
        }
    }
}

import { store, shapeSize } from './store.js';

export interface TapeEntry {
    outputId: number;
    inputIds: number[];
    backward: (gradOutputId: number) => (number | null)[];
}

class Tape {
    entries: TapeEntry[] = [];
    enabled: boolean = true;

    record(entry: TapeEntry): void {
        if (this.enabled) this.entries.push(entry);
    }

    clear(): void {
        this.entries = [];
    }

    backward(lossId: number): void {
        const prev = this.enabled;
        this.enabled = false;
        try {
            const lossEntry = store.get(lossId);
            const size = shapeSize(lossEntry.shape);
            const seedId = store.alloc(new Float32Array(size).fill(1.0), [...lossEntry.shape]);

            const gradMap = new Map<number, number>();
            gradMap.set(lossId, seedId);

            for (let i = this.entries.length - 1; i >= 0; i--) {
                const entry = this.entries[i];
                const gradOutId = gradMap.get(entry.outputId);
                if (gradOutId === undefined) continue;

                const inputGradIds = entry.backward(gradOutId);

                for (let j = 0; j < entry.inputIds.length; j++) {
                    const inputId = entry.inputIds[j];
                    const gId = inputGradIds[j];
                    if (gId == null) continue;

                    const existing = gradMap.get(inputId);
                    if (existing !== undefined) {
                        const eData = store.get(existing).data;
                        const nData = store.getContiguousData(gId);
                        for (let k = 0; k < eData.length; k++) eData[k] += nData[k];
                    } else {
                        gradMap.set(inputId, gId);
                    }
                }
            }

            for (const [tensorId, gradId] of gradMap) {
                if (!store.has(tensorId)) continue;
                const entry = store.get(tensorId);
                if (entry.requiresGrad) {
                    store.accumulateGrad(tensorId, store.getContiguousData(gradId));
                }
            }

            this.entries = [];
        } finally {
            this.enabled = prev;
        }
    }
}

export const tape = new Tape();

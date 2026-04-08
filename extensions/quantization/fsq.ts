import { Module } from '../../toy/module.js';
import { Tensor } from '../../toy/tensor.js';
import { TensorData, createSharedStorage } from '../../toy/tensor_data.js';
import { tanh } from '../../toy/nn.js';

export class FiniteScalarQuantize extends Module {
    levels: number[];
    implicitCodebookSize: number;

    constructor(levels: number[]) {
        super();
        this.levels = levels;
        this.implicitCodebookSize = levels.reduce((a, b) => a * b, 1);
    }

    forward(x: Tensor): { quantized: Tensor; indices: number[] } {
        // x: [N, C] where C === levels.length
        const N = x.shape[0]!;
        const C = this.levels.length;

        // 1. Bound to [-1, 1] via tanh
        const bounded = tanh(x);

        // 2. Scale to [0, L-1], round (straight-through), scale back to [-1, 1]
        // For each channel c with L_c levels:
        //   scaled = (bounded + 1) / 2 * (L_c - 1)
        //   rounded = round(scaled)  -- straight-through
        //   quantized = rounded / (L_c - 1) * 2 - 1
        const boundedData = bounded.data.storage;
        const quantizedStorage = createSharedStorage(N * C);
        const indicesArr: number[] = [];

        for (let i = 0; i < N; i++) {
            let index = 0;
            let multiplier = 1;
            for (let c = C - 1; c >= 0; c--) {
                const L = this.levels[c]!;
                const val = boundedData[i * C + c]!;
                const scaled = (val + 1) / 2 * (L - 1);
                const rounded = Math.round(Math.max(0, Math.min(L - 1, scaled)));
                quantizedStorage[i * C + c] = rounded / (L - 1) * 2 - 1;
                index += rounded * multiplier;
                multiplier *= L;
            }
            indicesArr.push(index);
        }

        const quantizedRaw = new Tensor(new TensorData(quantizedStorage, [N, C]));

        // Straight-through: forward uses quantized, backward flows through bounded (via x)
        const quantized = bounded.add(quantizedRaw.sub(bounded).detach());

        return { quantized, indices: indicesArr };
    }
}

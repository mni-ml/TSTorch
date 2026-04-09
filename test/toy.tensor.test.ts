import { Tensor } from '../toy/tensor.js';
import { assert, assertClose, section, summarize } from './helpers.js';

section('sin/cos/sqrt preserve shape');

{
    const x = Tensor.tensor([[1, 2], [3, 4]]);
    const s = x.sin();
    assert(s.shape[0] === 2 && s.shape[1] === 2, 'sin preserves 2D shape');
    assertClose(s.get([0, 0]), Math.sin(1), 1e-6, 'sin 2D [0,0]');
    assertClose(s.get([1, 1]), Math.sin(4), 1e-6, 'sin 2D [1,1]');

    const c = x.cos();
    assertClose(c.get([0, 1]), Math.cos(2), 1e-6, 'cos 2D [0,1]');

    const q = Tensor.tensor([[1, 4], [9, 16]]).sqrt();
    assertClose(q.get([0, 0]), 1, 1e-6, 'sqrt 2D [0,0]');
    assertClose(q.get([1, 1]), 4, 1e-6, 'sqrt 2D [1,1]');
}

section('scalar tensor');

{
    const x = Tensor.tensor(3.14);
    assert(x.dims === 0, 'scalar tensor has 0 dims');
    assert(x.size === 1, 'scalar tensor has size 1');
    assertClose(x.item(), 3.14, 1e-10, 'scalar tensor item()');

    const y = x.sin();
    assert(y.dims === 0, 'sin of scalar is scalar');
    assertClose(y.item(), Math.sin(3.14), 1e-6, 'sin of scalar value');
}

section('1D tensor');

{
    const x = Tensor.tensor([1, 2, 3, 4, 5]);
    assert(x.shape[0] === 5, '1D shape');
    assert(x.dims === 1, '1D dims');

    const s = x.sum();
    assertClose(s.item(), 15, 1e-6, 'sum of 1D');

    const m = x.mean();
    assertClose(m.item(), 3, 1e-6, 'mean of 1D');
}

summarize();

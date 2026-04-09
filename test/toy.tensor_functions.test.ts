import { Tensor } from '../toy/tensor.js';
import { TensorHistory } from '../toy/tensor_functions.js';
import { assert, assertClose, section, summarize } from './helpers.js';

section('Sin forward');

{
    const x = Tensor.tensor([0, Math.PI / 6, Math.PI / 4, Math.PI / 2, Math.PI]);
    const y = x.sin();
    assertClose(y.get([0]), Math.sin(0), 1e-6, 'sin(0)');
    assertClose(y.get([1]), Math.sin(Math.PI / 6), 1e-6, 'sin(pi/6)');
    assertClose(y.get([2]), Math.sin(Math.PI / 4), 1e-6, 'sin(pi/4)');
    assertClose(y.get([3]), Math.sin(Math.PI / 2), 1e-6, 'sin(pi/2)');
    assertClose(y.get([4]), Math.sin(Math.PI), 1e-6, 'sin(pi)');
}

section('Cos forward');

{
    const x = Tensor.tensor([0, Math.PI / 6, Math.PI / 4, Math.PI / 2, Math.PI]);
    const y = x.cos();
    assertClose(y.get([0]), Math.cos(0), 1e-6, 'cos(0)');
    assertClose(y.get([1]), Math.cos(Math.PI / 6), 1e-6, 'cos(pi/6)');
    assertClose(y.get([2]), Math.cos(Math.PI / 4), 1e-6, 'cos(pi/4)');
    assertClose(y.get([3]), Math.cos(Math.PI / 2), 1e-6, 'cos(pi/2)');
    assertClose(y.get([4]), Math.cos(Math.PI), 1e-6, 'cos(pi)');
}

section('Sqrt forward');

{
    const x = Tensor.tensor([0, 1, 4, 9, 16]);
    const y = x.sqrt();
    assertClose(y.get([0]), 0, 1e-4, 'sqrt(0)');
    assertClose(y.get([1]), 1, 1e-6, 'sqrt(1)');
    assertClose(y.get([2]), 2, 1e-6, 'sqrt(4)');
    assertClose(y.get([3]), 3, 1e-6, 'sqrt(9)');
    assertClose(y.get([4]), 4, 1e-6, 'sqrt(16)');
}

section('Sin backward');

{
    // d/dx sin(x) = cos(x)
    const x = Tensor.tensor([0.5, 1.0, 2.0]);
    x.history = new TensorHistory();
    x.sin().sum().backward();
    assertClose(x.grad!.get([0]), Math.cos(0.5), 1e-5, 'dsin/dx at 0.5');
    assertClose(x.grad!.get([1]), Math.cos(1.0), 1e-5, 'dsin/dx at 1.0');
    assertClose(x.grad!.get([2]), Math.cos(2.0), 1e-5, 'dsin/dx at 2.0');
}

section('Cos backward');

{
    // d/dx cos(x) = -sin(x)
    const x = Tensor.tensor([0.5, 1.0, 2.0]);
    x.history = new TensorHistory();
    x.cos().sum().backward();
    assertClose(x.grad!.get([0]), -Math.sin(0.5), 1e-5, 'dcos/dx at 0.5');
    assertClose(x.grad!.get([1]), -Math.sin(1.0), 1e-5, 'dcos/dx at 1.0');
    assertClose(x.grad!.get([2]), -Math.sin(2.0), 1e-5, 'dcos/dx at 2.0');
}

section('Sqrt backward');

{
    // d/dx sqrt(x) = 1 / (2 * sqrt(x))
    const x = Tensor.tensor([1.0, 4.0, 9.0]);
    x.history = new TensorHistory();
    x.sqrt().sum().backward();
    assertClose(x.grad!.get([0]), 1 / (2 * Math.sqrt(1.0)), 1e-5, 'dsqrt/dx at 1');
    assertClose(x.grad!.get([1]), 1 / (2 * Math.sqrt(4.0)), 1e-5, 'dsqrt/dx at 4');
    assertClose(x.grad!.get([2]), 1 / (2 * Math.sqrt(9.0)), 1e-5, 'dsqrt/dx at 9');
}

section('gradient chain: sin(cos(x))');

{
    // d/dx sin(cos(x)) = cos(cos(x)) * (-sin(x))
    const x = Tensor.tensor([0.7]);
    x.history = new TensorHistory();
    x.cos().sin().sum().backward();
    const expected = Math.cos(Math.cos(0.7)) * (-Math.sin(0.7));
    assertClose(x.grad!.get([0]), expected, 1e-5, 'chain rule sin(cos(x))');
}

section('autograd vs finite differences');

{
    const eps = 1e-5;
    const x = Tensor.tensor([2.0]);
    x.history = new TensorHistory();
    x.sqrt().sum().backward();
    const sqrtGradNumeric = (Math.sqrt(2.0 + eps) - Math.sqrt(2.0 - eps)) / (2 * eps);
    assertClose(x.grad!.get([0]), sqrtGradNumeric, 1e-4, 'sqrt autograd vs finite diff');
}

summarize();

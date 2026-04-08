import { Tensor } from '../toy/tensor.ts';
import { TensorHistory } from '../toy/tensor_functions.ts';
import { SGD } from '../toy/optimizer.ts';
import { Parameter } from '../toy/module.ts';
import { Linear, RMSNorm, softmax, mseLoss, tanh } from '../toy/nn.ts';
import { VectorQuantize } from '../extensions/quantization/vq.ts';

let passed = 0;
let failed = 0;
const failures = [];

function assert(cond, msg) {
    if (!cond) {
        failed++;
        failures.push(msg);
        console.error(`  FAIL: ${msg}`);
    } else {
        passed++;
    }
}

function assertClose(a, b, tol = 1e-4, msg = '') {
    if (Math.abs(a - b) > tol) {
        failed++;
        const detail = `${msg}: ${a} != ${b} (tol=${tol})`;
        failures.push(detail);
        console.error(`  FAIL: ${detail}`);
    } else {
        passed++;
    }
}

function section(name) {
    console.log(`\n--- ${name} ---`);
}

// ============================================================
// sin forward
// ============================================================

section('sin');

{
    const x = Tensor.tensor([0, Math.PI / 6, Math.PI / 4, Math.PI / 2, Math.PI]);
    const y = x.sin();
    assertClose(y.get([0]), Math.sin(0), 1e-6, 'sin(0)');
    assertClose(y.get([1]), Math.sin(Math.PI / 6), 1e-6, 'sin(pi/6)');
    assertClose(y.get([2]), Math.sin(Math.PI / 4), 1e-6, 'sin(pi/4)');
    assertClose(y.get([3]), Math.sin(Math.PI / 2), 1e-6, 'sin(pi/2)');
    assertClose(y.get([4]), Math.sin(Math.PI), 1e-6, 'sin(pi)');
}

// ============================================================
// cos forward
// ============================================================

section('cos');

{
    const x = Tensor.tensor([0, Math.PI / 6, Math.PI / 4, Math.PI / 2, Math.PI]);
    const y = x.cos();
    assertClose(y.get([0]), Math.cos(0), 1e-6, 'cos(0)');
    assertClose(y.get([1]), Math.cos(Math.PI / 6), 1e-6, 'cos(pi/6)');
    assertClose(y.get([2]), Math.cos(Math.PI / 4), 1e-6, 'cos(pi/4)');
    assertClose(y.get([3]), Math.cos(Math.PI / 2), 1e-6, 'cos(pi/2)');
    assertClose(y.get([4]), Math.cos(Math.PI), 1e-6, 'cos(pi)');
}

// ============================================================
// sqrt forward
// ============================================================

section('sqrt');

{
    const x = Tensor.tensor([0, 1, 4, 9, 16]);
    const y = x.sqrt();
    assertClose(y.get([0]), 0, 1e-4, 'sqrt(0)');
    assertClose(y.get([1]), 1, 1e-6, 'sqrt(1)');
    assertClose(y.get([2]), 2, 1e-6, 'sqrt(4)');
    assertClose(y.get([3]), 3, 1e-6, 'sqrt(9)');
    assertClose(y.get([4]), 4, 1e-6, 'sqrt(16)');
}

// ============================================================
// sin backward
// ============================================================

section('sin backward');

{
    // d/dx sin(x) = cos(x)
    const x = Tensor.tensor([0.5, 1.0, 2.0]);
    x.history = new TensorHistory();
    const y = x.sin().sum();
    y.backward();
    assertClose(x.grad.get([0]), Math.cos(0.5), 1e-5, 'dsin/dx at 0.5');
    assertClose(x.grad.get([1]), Math.cos(1.0), 1e-5, 'dsin/dx at 1.0');
    assertClose(x.grad.get([2]), Math.cos(2.0), 1e-5, 'dsin/dx at 2.0');
}

// ============================================================
// cos backward
// ============================================================

section('cos backward');

{
    // d/dx cos(x) = -sin(x)
    const x = Tensor.tensor([0.5, 1.0, 2.0]);
    x.history = new TensorHistory();
    const y = x.cos().sum();
    y.backward();
    assertClose(x.grad.get([0]), -Math.sin(0.5), 1e-5, 'dcos/dx at 0.5');
    assertClose(x.grad.get([1]), -Math.sin(1.0), 1e-5, 'dcos/dx at 1.0');
    assertClose(x.grad.get([2]), -Math.sin(2.0), 1e-5, 'dcos/dx at 2.0');
}

// ============================================================
// sqrt backward
// ============================================================

section('sqrt backward');

{
    // d/dx sqrt(x) = 1 / (2 * sqrt(x))
    const x = Tensor.tensor([1.0, 4.0, 9.0]);
    x.history = new TensorHistory();
    const y = x.sqrt().sum();
    y.backward();
    assertClose(x.grad.get([0]), 1 / (2 * Math.sqrt(1.0)), 1e-5, 'dsqrt/dx at 1');
    assertClose(x.grad.get([1]), 1 / (2 * Math.sqrt(4.0)), 1e-5, 'dsqrt/dx at 4');
    assertClose(x.grad.get([2]), 1 / (2 * Math.sqrt(9.0)), 1e-5, 'dsqrt/dx at 9');
}

// ============================================================
// gradient chain: sin(cos(x))
// ============================================================

section('gradient chain: sin(cos(x))');

{
    // d/dx sin(cos(x)) = cos(cos(x)) * (-sin(x))
    const x = Tensor.tensor([0.7]);
    x.history = new TensorHistory();
    const y = x.cos().sin().sum();
    y.backward();
    const expected = Math.cos(Math.cos(0.7)) * (-Math.sin(0.7));
    assertClose(x.grad.get([0]), expected, 1e-5, 'chain rule sin(cos(x))');
}

// ============================================================
// finite difference gradient check
// ============================================================

section('finite difference gradient check');

{
    const eps = 1e-5;

    // Check sin gradient via finite differences
    const xVal = 1.3;
    const sinGradAnalytic = Math.cos(xVal);
    const sinGradNumeric = (Math.sin(xVal + eps) - Math.sin(xVal - eps)) / (2 * eps);
    assertClose(sinGradAnalytic, sinGradNumeric, 1e-5, 'sin finite diff');

    // Check cos gradient via finite differences
    const cosGradAnalytic = -Math.sin(xVal);
    const cosGradNumeric = (Math.cos(xVal + eps) - Math.cos(xVal - eps)) / (2 * eps);
    assertClose(cosGradAnalytic, cosGradNumeric, 1e-5, 'cos finite diff');

    // Check sqrt gradient via finite differences on autograd
    const x = Tensor.tensor([2.0]);
    x.history = new TensorHistory();
    x.sqrt().sum().backward();
    const sqrtGradNumeric = (Math.sqrt(2.0 + eps) - Math.sqrt(2.0 - eps)) / (2 * eps);
    assertClose(x.grad.get([0]), sqrtGradNumeric, 1e-4, 'sqrt autograd vs finite diff');
}

// ============================================================
// 2D tensor operations
// ============================================================

section('2D sin/cos/sqrt');

{
    const x = Tensor.tensor([[1, 2], [3, 4]]);
    const s = x.sin();
    assert(s.shape[0] === 2 && s.shape[1] === 2, 'sin preserves shape');
    assertClose(s.get([0, 0]), Math.sin(1), 1e-6, 'sin 2D element');
    assertClose(s.get([1, 1]), Math.sin(4), 1e-6, 'sin 2D element [1,1]');

    const c = x.cos();
    assertClose(c.get([0, 1]), Math.cos(2), 1e-6, 'cos 2D element');

    const q = Tensor.tensor([[1, 4], [9, 16]]).sqrt();
    assertClose(q.get([0, 0]), 1, 1e-6, 'sqrt 2D element [0,0]');
    assertClose(q.get([1, 1]), 4, 1e-6, 'sqrt 2D element [1,1]');
}

// ============================================================
// matmul forward
// ============================================================

section('matmul');

{
    // [2,3] x [3,2] = [2,2]
    const a = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);
    const b = Tensor.tensor([[7, 8], [9, 10], [11, 12]]);
    const c = a.matmul(b);
    assert(c.shape[0] === 2 && c.shape[1] === 2, 'matmul output shape');
    // Row 0: 1*7+2*9+3*11=58, 1*8+2*10+3*12=64
    assertClose(c.get([0, 0]), 58, 1e-6, 'matmul [0,0]');
    assertClose(c.get([0, 1]), 64, 1e-6, 'matmul [0,1]');
    // Row 1: 4*7+5*9+6*11=139, 4*8+5*10+6*12=154
    assertClose(c.get([1, 0]), 139, 1e-6, 'matmul [1,0]');
    assertClose(c.get([1, 1]), 154, 1e-6, 'matmul [1,1]');
}

// ============================================================
// matmul backward
// ============================================================

section('matmul backward');

{
    const a = Tensor.tensor([[1, 2], [3, 4]]);
    a.history = new TensorHistory();
    const b = Tensor.tensor([[5, 6], [7, 8]]);
    b.history = new TensorHistory();
    const c = a.matmul(b).sum();
    c.backward();

    // dL/dA = ones(2,2) @ B^T
    // B^T = [[5,7],[6,8]], ones @ B^T = [[11,14],[11,14]]
    assertClose(a.grad.get([0, 0]), 11, 1e-5, 'matmul grad A [0,0]');
    assertClose(a.grad.get([0, 1]), 15, 1e-5, 'matmul grad A [0,1]');
    // dL/dB = A^T @ ones(2,2)
    // A^T = [[1,3],[2,4]], A^T @ ones = [[4,4],[6,6]]
    assertClose(b.grad.get([0, 0]), 4, 1e-5, 'matmul grad B [0,0]');
    assertClose(b.grad.get([1, 1]), 6, 1e-5, 'matmul grad B [1,1]');
}

// ============================================================
// transpose
// ============================================================

section('transpose');

{
    const a = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);
    const t = a.transpose();
    assert(t.shape[0] === 3 && t.shape[1] === 2, 'transpose shape');
    assertClose(t.get([0, 0]), 1, 1e-6, 'transpose [0,0]');
    assertClose(t.get([0, 1]), 4, 1e-6, 'transpose [0,1]');
    assertClose(t.get([2, 0]), 3, 1e-6, 'transpose [2,0]');
}

// ============================================================
// div
// ============================================================

section('div');

{
    const a = Tensor.tensor([6, 8, 10]);
    const b = Tensor.tensor([2, 4, 5]);
    const c = a.div(b);
    assertClose(c.get([0]), 3, 1e-6, 'div [0]');
    assertClose(c.get([1]), 2, 1e-6, 'div [1]');
    assertClose(c.get([2]), 2, 1e-6, 'div [2]');

    // div by scalar
    const d = a.div(2);
    assertClose(d.get([0]), 3, 1e-6, 'div scalar [0]');
}

// ============================================================
// div backward
// ============================================================

section('div backward');

{
    const x = Tensor.tensor([2.0, 3.0]);
    x.history = new TensorHistory();
    // f(x) = sum(x / 2) = sum(x) / 2, df/dx = [0.5, 0.5]
    x.div(2).sum().backward();
    assertClose(x.grad.get([0]), 0.5, 1e-5, 'div backward [0]');
    assertClose(x.grad.get([1]), 0.5, 1e-5, 'div backward [1]');
}

// ============================================================
// SGD with Tensor parameters
// ============================================================

section('SGD with Tensor parameters');

{
    // Create a tensor parameter
    const w = Tensor.tensor([[1.0, 2.0], [3.0, 4.0]]);
    w.history = new TensorHistory(); // mark as leaf
    const param = new Parameter(w);
    const opt = new SGD([param], 0.1);

    // Simulate a gradient: all ones
    w.grad = Tensor.ones([2, 2]);
    opt.step();

    // w_new = w - 0.1 * grad = [[0.9, 1.9], [2.9, 3.9]]
    assertClose(param.value.get([0, 0]), 0.9, 1e-6, 'SGD tensor step [0,0]');
    assertClose(param.value.get([1, 1]), 3.9, 1e-6, 'SGD tensor step [1,1]');

    // Updated param should still be a leaf (requiresGrad)
    assert(param.value.requiresGrad(), 'SGD preserves grad tracking');

    // zeroGrad should clear
    opt.zeroGrad();
    assert(param.value.grad === null, 'SGD zeroGrad clears tensor grad');
}

// ============================================================
// negative dim in sum/mean
// ============================================================

section('negative dim');

{
    const x = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);
    const s = x.sum(-1); // sum along last dim
    assert(s.shape[0] === 2 && s.shape[1] === 1, 'sum(-1) shape');
    assertClose(s.get([0, 0]), 6, 1e-6, 'sum(-1) row 0');
    assertClose(s.get([1, 0]), 15, 1e-6, 'sum(-1) row 1');

    const m = x.mean(-1);
    assertClose(m.get([0, 0]), 2, 1e-6, 'mean(-1) row 0');
    assertClose(m.get([1, 0]), 5, 1e-6, 'mean(-1) row 1');
}

// ============================================================
// Tensor.randn
// ============================================================

section('Tensor.randn');

{
    const r = Tensor.randn([100]);
    assert(r.shape[0] === 100, 'randn shape');
    // Check rough statistics: mean should be near 0
    const mean = r.sum().item() / 100;
    assert(Math.abs(mean) < 1.0, 'randn mean roughly 0');
}

// ============================================================
// Tensor.parameter
// ============================================================

section('Tensor.parameter');

{
    const p = Tensor.parameter([3, 4]);
    assert(p.shape[0] === 3 && p.shape[1] === 4, 'parameter shape');
    assert(p.requiresGrad(), 'parameter requires grad');
    assert(p.isLeaf(), 'parameter is leaf');
}

// ============================================================
// Linear
// ============================================================

section('Linear');

{
    const linear = new Linear(3, 2);
    const x = Tensor.tensor([[1, 0, 0], [0, 1, 0]]);
    const out = linear.forward(x);
    assert(out.shape[0] === 2 && out.shape[1] === 2, 'Linear output shape');

    // Test that parameters exist and have correct shapes
    const params = linear.parameters();
    assert(params.length === 2, 'Linear has 2 parameters');
    assert(linear.weight.value.shape[0] === 3, 'Linear weight shape[0]');
    assert(linear.weight.value.shape[1] === 2, 'Linear weight shape[1]');
    assert(linear.bias.value.shape[0] === 2, 'Linear bias shape');
}

// ============================================================
// Linear backward
// ============================================================

section('Linear backward');

{
    const linear = new Linear(2, 1);
    const x = Tensor.tensor([[1, 2]]);
    x.history = new TensorHistory();
    const out = linear.forward(x);
    const loss = out.sum();
    loss.backward();
    // Weight and bias should have gradients
    assert(linear.weight.value.grad !== null, 'Linear weight has grad');
    assert(linear.bias.value.grad !== null, 'Linear bias has grad');
    assert(x.grad !== null, 'Linear input has grad');
}

// ============================================================
// RMSNorm
// ============================================================

section('RMSNorm');

{
    const norm = new RMSNorm(3);
    const x = Tensor.tensor([[1, 2, 3]]);
    x.history = new TensorHistory();
    const out = norm.forward(x);
    assert(out.shape[0] === 1 && out.shape[1] === 3, 'RMSNorm output shape');

    // RMS of [1,2,3] = sqrt(mean([1,4,9])) = sqrt(14/3)
    // normalized = [1,2,3] / sqrt(14/3)
    const rms = Math.sqrt((1 + 4 + 9) / 3);
    assertClose(out.get([0, 0]), 1 / rms, 1e-4, 'RMSNorm value [0]');
    assertClose(out.get([0, 1]), 2 / rms, 1e-4, 'RMSNorm value [1]');
    assertClose(out.get([0, 2]), 3 / rms, 1e-4, 'RMSNorm value [2]');

    // Should support backward
    out.sum().backward();
    assert(x.grad !== null, 'RMSNorm backward produces grad');
}

// ============================================================
// softmax
// ============================================================

section('softmax');

{
    const x = Tensor.tensor([[1, 2, 3]]);
    const s = softmax(x, -1);
    assert(s.shape[0] === 1 && s.shape[1] === 3, 'softmax shape');

    // All positive
    assert(s.get([0, 0]) > 0, 'softmax positive [0]');
    assert(s.get([0, 1]) > 0, 'softmax positive [1]');
    assert(s.get([0, 2]) > 0, 'softmax positive [2]');

    // Sum to 1
    const total = s.get([0, 0]) + s.get([0, 1]) + s.get([0, 2]);
    assertClose(total, 1.0, 1e-5, 'softmax sums to 1');

    // Monotonic
    assert(s.get([0, 2]) > s.get([0, 1]), 'softmax monotonic');
    assert(s.get([0, 1]) > s.get([0, 0]), 'softmax monotonic 2');
}

// ============================================================
// mseLoss
// ============================================================

section('mseLoss');

{
    const pred = Tensor.tensor([1, 2, 3]);
    const target = Tensor.tensor([1, 2, 3]);
    const loss = mseLoss(pred, target);
    assertClose(loss.item(), 0, 1e-6, 'mseLoss perfect prediction');

    const pred2 = Tensor.tensor([1, 2, 3]);
    pred2.history = new TensorHistory();
    const target2 = Tensor.tensor([2, 3, 4]);
    const loss2 = mseLoss(pred2, target2);
    // MSE = mean([(1-2)^2, (2-3)^2, (3-4)^2]) = mean([1,1,1]) = 1
    assertClose(loss2.item(), 1, 1e-6, 'mseLoss value');

    loss2.backward();
    // d/dpred MSE = 2*(pred-target)/n = 2*(-1)/3 = -2/3
    assertClose(pred2.grad.get([0]), -2/3, 1e-5, 'mseLoss grad [0]');
}

// ============================================================
// tanh
// ============================================================

section('tanh');

{
    const x = Tensor.tensor([0, 1, -1]);
    const y = tanh(x);
    assertClose(y.get([0]), Math.tanh(0), 1e-4, 'tanh(0)');
    assertClose(y.get([1]), Math.tanh(1), 1e-3, 'tanh(1)');
    assertClose(y.get([2]), Math.tanh(-1), 1e-3, 'tanh(-1)');
}

// ============================================================
// VectorQuantize
// ============================================================

section('VectorQuantize');

{
    const vq = new VectorQuantize(4, 8);
    const x = Tensor.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]);
    x.history = new TensorHistory();
    const { quantized, indices, commitmentLoss } = vq.forward(x);

    // Output shape matches input
    assert(quantized.shape[0] === 3 && quantized.shape[1] === 4, 'VQ output shape');

    // Indices are valid codebook entries
    assert(indices.length === 3, 'VQ indices length');
    assert(indices.every(i => i >= 0 && i < 8), 'VQ indices in range');

    // Commitment loss is non-negative
    assert(commitmentLoss.item() >= 0, 'VQ commitment loss non-negative');

    // Straight-through: gradient should flow through x
    quantized.sum().backward();
    assert(x.grad !== null, 'VQ straight-through gradient exists');
    // Gradient should be all ones (from sum) since straight-through passes through
    assertClose(x.grad.get([0, 0]), 1, 1e-5, 'VQ straight-through grad value');
}

// ============================================================
// VectorQuantize - nearest neighbor correctness
// ============================================================

section('VectorQuantize nearest neighbor');

{
    const vq = new VectorQuantize(2, 4);
    // Set codebook manually to known values
    const cb = Tensor.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]]);
    cb.history = new TensorHistory();
    vq.codebook = new Parameter(cb);

    const x = Tensor.tensor([[0.9, 0.1], [-0.1, 0.8]]);
    const { indices } = vq.forward(x);

    // [0.9, 0.1] should be closest to [1, 0] (index 0)
    assert(indices[0] === 0, 'VQ nearest to [1,0]');
    // [-0.1, 0.8] should be closest to [0, 1] (index 1)
    assert(indices[1] === 1, 'VQ nearest to [0,1]');
}

// ============================================================
// Summary
// ============================================================

console.log(`\n${'='.repeat(50)}`);
if (failed === 0) {
    console.log(`All ${passed} tests passed!`);
} else {
    console.log(`${passed} passed, ${failed} FAILED:`);
    for (const f of failures) console.log(`  - ${f}`);
    process.exit(1);
}

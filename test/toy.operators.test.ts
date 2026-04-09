import * as operators from '../toy/operators.js';
import { assert, assertClose, section, summarize } from './helpers.js';

section('sin operator');

assertClose(operators.sin(0), 0, 1e-10, 'sin(0)');
assertClose(operators.sin(Math.PI / 2), 1, 1e-10, 'sin(pi/2)');
assertClose(operators.sin(Math.PI), 0, 1e-10, 'sin(pi)');
assertClose(operators.sin(-Math.PI / 2), -1, 1e-10, 'sin(-pi/2)');

section('cos operator');

assertClose(operators.cos(0), 1, 1e-10, 'cos(0)');
assertClose(operators.cos(Math.PI / 2), 0, 1e-10, 'cos(pi/2)');
assertClose(operators.cos(Math.PI), -1, 1e-10, 'cos(pi)');

section('sqrt operator');

assertClose(operators.sqrt(0), 0, 1e-6, 'sqrt(0)');
assertClose(operators.sqrt(1), 1, 1e-10, 'sqrt(1)');
assertClose(operators.sqrt(4), 2, 1e-10, 'sqrt(4)');
assertClose(operators.sqrt(9), 3, 1e-10, 'sqrt(9)');
// EPS guard: negative values clamp to EPS
assert(operators.sqrt(-1) >= 0, 'sqrt(-1) non-negative via EPS guard');

section('finite difference checks');

{
    const eps = 1e-5;
    const x = 1.3;

    const sinGrad = operators.cos(x);
    const sinGradNumeric = (operators.sin(x + eps) - operators.sin(x - eps)) / (2 * eps);
    assertClose(sinGrad, sinGradNumeric, 1e-5, 'sin derivative matches cos');

    const cosGrad = -operators.sin(x);
    const cosGradNumeric = (operators.cos(x + eps) - operators.cos(x - eps)) / (2 * eps);
    assertClose(cosGrad, cosGradNumeric, 1e-5, 'cos derivative matches -sin');

    const sqrtGradNumeric = (operators.sqrt(x + eps) - operators.sqrt(x - eps)) / (2 * eps);
    const sqrtGradAnalytic = 1 / (2 * Math.sqrt(x));
    assertClose(sqrtGradAnalytic, sqrtGradNumeric, 1e-4, 'sqrt derivative matches 1/(2sqrt(x))');
}

summarize();

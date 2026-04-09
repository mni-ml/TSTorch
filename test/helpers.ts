let passed = 0;
let failed = 0;
let skipped = 0;
const failures: string[] = [];

export function assert(cond: boolean, msg: string): void {
    if (!cond) {
        failed++;
        failures.push(msg);
        console.error(`  FAIL: ${msg}`);
    } else {
        passed++;
        console.log(`  pass  ${msg}`);
    }
}

export function assertClose(a: number, b: number, tol: number = 1e-4, msg: string = ''): void {
    if (Math.abs(a - b) > tol) {
        failed++;
        const detail = `${msg}: ${a} != ${b} (tol=${tol})`;
        failures.push(detail);
        console.error(`  FAIL: ${detail}`);
    } else {
        passed++;
        console.log(`  pass  ${msg}`);
    }
}

export function skip(msg: string): void {
    skipped++;
    console.log(`  (skipped — ${msg})`);
}

export function section(name: string): void {
    console.log(`\n--- ${name} ---`);
}

export function summarize(): void {
    console.log(`\n${'='.repeat(50)}`);
    if (failed === 0) {
        const parts = [`${passed} passed`];
        if (skipped > 0) parts.push(`${skipped} skipped`);
        console.log(parts.join(', '));
    } else {
        console.log(`${passed} passed, ${failed} FAILED:`);
        for (const f of failures) console.log(`  - ${f}`);
        process.exit(1);
    }
}

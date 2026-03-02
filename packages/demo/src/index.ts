import runScalar from "./runScalar.js";
import runTensor from "./runTensor.js";

const mode = process.argv[2] ?? "tensor";

if (mode === "scalar") {
    runScalar();
} else {
    runTensor();
}

import runScalar from "./runScalar.js";
import runTensor from "./runTensor.js";
import runFastTensor from "./runFastTensor.js";

const mode = process.argv[2] ?? "tensor";

if (mode === "scalar") {
    runScalar();
} else if (mode === "fast") {
    runFastTensor();
} else if (mode === "mnist") {
    const { default: runMnist } = await import("./runMnist.js");
    await runMnist();
} else if (mode === "sentiment") {
    const { default: runSentiment } = await import("./runSentiment.js");
    runSentiment();
} else {
    runTensor();
}

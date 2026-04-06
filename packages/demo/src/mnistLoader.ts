import { createWriteStream, existsSync, readFileSync, mkdirSync } from "node:fs";
import { get } from "node:https";
import { join } from "node:path";
import { createGunzip } from "node:zlib";
import { pipeline } from "node:stream/promises";

const MNIST_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/";

const FILES = {
    trainImages: "train-images-idx3-ubyte.gz",
    trainLabels: "train-labels-idx1-ubyte.gz",
    testImages: "t10k-images-idx3-ubyte.gz",
    testLabels: "t10k-labels-idx1-ubyte.gz",
};

export interface MnistData {
    trainImages: Float64Array;
    trainLabels: Uint8Array;
    testImages: Float64Array;
    testLabels: Uint8Array;
    numTrain: number;
    numTest: number;
    rows: number;
    cols: number;
}

function download(url: string, dest: string): Promise<void> {
    return new Promise((resolve, reject) => {
        const file = createWriteStream(dest);
        get(url, (response) => {
            if (response.statusCode === 301 || response.statusCode === 302) {
                const redirectUrl = response.headers.location!;
                get(redirectUrl, (r2) => {
                    pipeline(r2, createGunzip(), file).then(resolve).catch(reject);
                }).on("error", reject);
            } else {
                pipeline(response, createGunzip(), file).then(resolve).catch(reject);
            }
        }).on("error", reject);
    });
}

function parseImages(buf: Buffer): { images: Float64Array; count: number; rows: number; cols: number } {
    const magic = buf.readUInt32BE(0);
    if (magic !== 2051) throw new Error(`Bad image magic: ${magic}`);
    const count = buf.readUInt32BE(4);
    const rows = buf.readUInt32BE(8);
    const cols = buf.readUInt32BE(12);
    const pixels = count * rows * cols;
    const images = new Float64Array(pixels);
    for (let i = 0; i < pixels; i++) {
        images[i] = buf[16 + i]! / 255.0;
    }
    return { images, count, rows, cols };
}

function parseLabels(buf: Buffer): Uint8Array {
    const magic = buf.readUInt32BE(0);
    if (magic !== 2049) throw new Error(`Bad label magic: ${magic}`);
    const count = buf.readUInt32BE(4);
    return new Uint8Array(buf.buffer, buf.byteOffset + 8, count);
}

export async function loadMnist(
    cacheDir: string,
    maxTrain: number = 5000,
    maxTest: number = 500,
): Promise<MnistData> {
    if (!existsSync(cacheDir)) {
        mkdirSync(cacheDir, { recursive: true });
    }

    for (const [key, filename] of Object.entries(FILES)) {
        const decompressed = filename.replace(".gz", "");
        const dest = join(cacheDir, decompressed);
        if (!existsSync(dest)) {
            const url = MNIST_BASE + filename;
            console.log(`Downloading ${url} ...`);
            await download(url, dest);
        }
    }

    const trainImgBuf = readFileSync(join(cacheDir, FILES.trainImages.replace(".gz", "")));
    const trainLblBuf = readFileSync(join(cacheDir, FILES.trainLabels.replace(".gz", "")));
    const testImgBuf = readFileSync(join(cacheDir, FILES.testImages.replace(".gz", "")));
    const testLblBuf = readFileSync(join(cacheDir, FILES.testLabels.replace(".gz", "")));

    const trainParsed = parseImages(trainImgBuf);
    const testParsed = parseImages(testImgBuf);
    const trainLabelsAll = parseLabels(trainLblBuf);
    const testLabelsAll = parseLabels(testLblBuf);

    const { rows, cols } = trainParsed;
    const imgSize = rows * cols;

    const numTrain = Math.min(maxTrain, trainParsed.count);
    const numTest = Math.min(maxTest, testParsed.count);

    const trainImages = trainParsed.images.slice(0, numTrain * imgSize);
    const trainLabels = trainLabelsAll.slice(0, numTrain);
    const testImages = testParsed.images.slice(0, numTest * imgSize);
    const testLabels = testLabelsAll.slice(0, numTest);

    return { trainImages, trainLabels, testImages, testLabels, numTrain, numTest, rows, cols };
}

/**
 * Downloads SST-2 from the GLUE benchmark, tokenizes it, and saves
 * a pre-processed JSON for the sentiment demo.
 *
 * Run: node scripts/prepare_sst2.mjs
 */

import { writeFileSync, existsSync, mkdirSync } from "node:fs";
import { get } from "node:https";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { createGunzip } from "node:zlib";
import { createWriteStream, readFileSync } from "node:fs";
import { pipeline } from "node:stream/promises";
import { createUnzip } from "node:zlib";

const __dirname = dirname(fileURLToPath(import.meta.url));
const DATA_DIR = join(__dirname, "..", "data");
const OUTPUT = join(DATA_DIR, "sst2.json");

const MAX_TRAIN = 3000;
const MAX_VALID = 500;
const MAX_SEQ_LEN = 20;
const VOCAB_SIZE = 2000;
const EMBED_DIM = 25;

function fetch(url) {
    return new Promise((resolve, reject) => {
        get(url, (res) => {
            if (res.statusCode === 301 || res.statusCode === 302) {
                fetch(res.headers.location).then(resolve).catch(reject);
                return;
            }
            const chunks = [];
            res.on("data", (c) => chunks.push(c));
            res.on("end", () => resolve(Buffer.concat(chunks)));
            res.on("error", reject);
        }).on("error", reject);
    });
}

function tokenize(text) {
    return text.toLowerCase().match(/\w+/g) || [];
}

async function downloadSST2() {
    // Download from GLUE benchmark
    const url = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip";
    const zipPath = join(DATA_DIR, "SST-2.zip");

    if (!existsSync(DATA_DIR)) mkdirSync(DATA_DIR, { recursive: true });

    if (!existsSync(zipPath)) {
        console.log("Downloading SST-2...");
        const data = await fetch(url);
        writeFileSync(zipPath, data);
    }

    // Use Node's built-in unzip via child_process
    const { execSync } = await import("node:child_process");
    const sst2Dir = join(DATA_DIR, "SST-2");
    if (!existsSync(sst2Dir)) {
        execSync(`unzip -o "${zipPath}" -d "${DATA_DIR}"`, { stdio: "pipe" });
    }

    // Parse TSV files
    function parseTsv(filePath) {
        const content = readFileSync(filePath, "utf-8");
        const lines = content.trim().split("\n");
        const header = lines[0];
        const rows = [];
        for (let i = 1; i < lines.length; i++) {
            const parts = lines[i].split("\t");
            if (parts.length >= 2) {
                rows.push({ sentence: parts[0], label: parseInt(parts[1]) });
            }
        }
        return rows;
    }

    const train = parseTsv(join(sst2Dir, "train.tsv"));
    // SST-2 dev set has labels, test set doesn't
    const valid = parseTsv(join(sst2Dir, "dev.tsv"));

    return { train, valid };
}

async function main() {
    console.log("Loading SST-2 data...");
    const { train: trainRaw, valid: validRaw } = await downloadSST2();

    // Tokenize
    const trainTexts = trainRaw.slice(0, MAX_TRAIN).map((r) => tokenize(r.sentence));
    const trainLabels = trainRaw.slice(0, MAX_TRAIN).map((r) => r.label);
    const validTexts = validRaw.slice(0, MAX_VALID).map((r) => tokenize(r.sentence));
    const validLabels = validRaw.slice(0, MAX_VALID).map((r) => r.label);

    // Build vocab
    const counter = {};
    for (const tokens of trainTexts) {
        for (const t of tokens) {
            counter[t] = (counter[t] || 0) + 1;
        }
    }
    const sorted = Object.entries(counter).sort((a, b) => b[1] - a[1]);
    const vocabWords = ["<pad>", "<unk>", ...sorted.slice(0, VOCAB_SIZE - 2).map(([w]) => w)];
    const word2idx = {};
    vocabWords.forEach((w, i) => (word2idx[w] = i));

    // Random initial embeddings (will be trained)
    const embeddings = [];
    for (let i = 0; i < vocabWords.length; i++) {
        const vec = [];
        for (let d = 0; d < EMBED_DIM; d++) {
            vec.push((Math.random() - 0.5) * 0.2);
        }
        embeddings.push(vec);
    }
    // Keep <pad> as zeros
    embeddings[0] = new Array(EMBED_DIM).fill(0);

    // Encode
    function encode(tokensList) {
        return tokensList.map((tokens) => {
            const indices = tokens.map((t) => word2idx[t] ?? 1); // 1 = <unk>
            if (indices.length < MAX_SEQ_LEN) {
                return [...indices, ...new Array(MAX_SEQ_LEN - indices.length).fill(0)];
            }
            return indices.slice(0, MAX_SEQ_LEN);
        });
    }

    const trainEncoded = encode(trainTexts);
    const validEncoded = encode(validTexts);

    const output = {
        vocab_size: vocabWords.length,
        embed_dim: EMBED_DIM,
        seq_len: MAX_SEQ_LEN,
        embeddings,
        train: trainEncoded.map((t, i) => ({ tokens: t, label: trainLabels[i] })),
        valid: validEncoded.map((t, i) => ({ tokens: t, label: validLabels[i] })),
    };

    writeFileSync(OUTPUT, JSON.stringify(output));
    const sizeMb = Buffer.byteLength(JSON.stringify(output)) / (1024 * 1024);
    console.log(`Saved ${OUTPUT} (${sizeMb.toFixed(1)} MB)`);
    console.log(`  Vocab: ${vocabWords.length}, Embed dim: ${EMBED_DIM}`);
    console.log(`  Train: ${trainEncoded.length}, Valid: ${validEncoded.length}`);
}

main().catch(console.error);

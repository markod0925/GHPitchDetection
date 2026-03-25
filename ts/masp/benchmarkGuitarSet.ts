import fs from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';

import {
  MASP_TUNED_PARAMS,
  computeCentError,
  computeHar,
  computeHarmonicityH,
  computeMbw,
  computeValidationDecision,
  scoreMaspMidiFrame,
} from './maspCore.mjs';

const NOTE_CONTEXT_MARGIN_SEC = 0.005;
const TARGET_RMS = 0.1;
const REPORT_EPS = 1e-9;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..', '..');
const outputDir = path.join(repoRoot, 'output', 'debug');

function parseArgs(argv) {
  let limit = null;
  let split = 'all';
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--limit') {
      limit = Number(argv[i + 1]);
      i += 1;
    } else if (arg === '--split') {
      split = String(argv[i + 1] ?? 'all').toLowerCase();
      i += 1;
    }
  }
  if (limit !== null && (!Number.isFinite(limit) || limit <= 0)) {
    throw new Error(`invalid --limit value: ${limit}`);
  }
  if (!['all', 'mono', 'comp'].includes(split)) {
    throw new Error(`invalid --split value: ${split}`);
  }
  return { limit, split };
}

function assertNodeV22() {
  if (!process.versions.node.startsWith('22.')) {
    throw new Error(`Node v22 is required, got ${process.version}`);
  }
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function readJsonl(filePath) {
  return fs
    .readFileSync(filePath, 'utf8')
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function writeJson(filePath, value) {
  fs.writeFileSync(filePath, JSON.stringify(value, null, 2));
}

function writeJsonl(filePath, rows) {
  const data = rows.map((row) => JSON.stringify(row)).join('\n') + '\n';
  fs.writeFileSync(filePath, data);
}

function writeText(filePath, text) {
  fs.writeFileSync(filePath, text);
}

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function ratio(numerator, denominator) {
  return denominator > 0 ? numerator / denominator : 0.0;
}

function percentile(sortedValues, q) {
  if (!sortedValues.length) {
    return 0.0;
  }
  const idx = Math.min(
    sortedValues.length - 1,
    Math.max(0, Math.ceil(q * sortedValues.length) - 1)
  );
  return sortedValues[idx];
}

function summarizeTimings(results) {
  const values = results.map((item) => item.execution_ms).sort((a, b) => a - b);
  if (!values.length) {
    return { mean: 0, median: 0, p95: 0, p99: 0, max: 0 };
  }
  const sum = values.reduce((acc, x) => acc + x, 0.0);
  return {
    mean: sum / values.length,
    median: percentile(values, 0.5),
    p95: percentile(values, 0.95),
    p99: percentile(values, 0.99),
    max: values[values.length - 1],
  };
}

function summarizeClassification(results) {
  let tp = 0;
  let fp = 0;
  let tn = 0;
  let fn = 0;
  for (const item of results) {
    const positive = item.expected_valid === true;
    if (item.pass && positive) tp += 1;
    else if (item.pass && !positive) fp += 1;
    else if (!item.pass && positive) fn += 1;
    else tn += 1;
  }
  const precision = ratio(tp, tp + fp);
  const recall = ratio(tp, tp + fn);
  const accuracy = ratio(tp + tn, results.length);
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0.0;
  return { tp, fp, tn, fn, precision, recall, f1, accuracy };
}

function formatMetricBlock(label, summary, timings) {
  return [
    `${label}`,
    '',
    `Precision: ${summary.precision.toFixed(6)}`,
    `Recall: ${summary.recall.toFixed(6)}`,
    `F1: ${summary.f1.toFixed(6)}`,
    `Accuracy: ${summary.accuracy.toFixed(6)}`,
    `TP / FP / TN / FN: ${summary.tp} / ${summary.fp} / ${summary.tn} / ${summary.fn}`,
    `execution_ms mean / median / p95 / p99 / max: ${timings.mean.toFixed(6)} / ${timings.median.toFixed(6)} / ${timings.p95.toFixed(6)} / ${timings.p99.toFixed(6)} / ${timings.max.toFixed(6)}`,
  ].join('\n');
}

function parseWavMono(filePath) {
  const buf = fs.readFileSync(filePath);
  if (buf.toString('ascii', 0, 4) !== 'RIFF' || buf.toString('ascii', 8, 12) !== 'WAVE') {
    throw new Error(`unsupported WAV container: ${filePath}`);
  }

  let offset = 12;
  let fmt = null;
  let dataOffset = -1;
  let dataSize = 0;

  while (offset + 8 <= buf.length) {
    const id = buf.toString('ascii', offset, offset + 4);
    const size = buf.readUInt32LE(offset + 4);
    const start = offset + 8;
    if (id === 'fmt ') {
      fmt = {
        audioFormat: buf.readUInt16LE(start),
        channels: buf.readUInt16LE(start + 2),
        sampleRate: buf.readUInt32LE(start + 4),
        blockAlign: buf.readUInt16LE(start + 12),
        bitsPerSample: buf.readUInt16LE(start + 14),
      };
    } else if (id === 'data') {
      dataOffset = start;
      dataSize = size;
      break;
    }
    offset = start + size + (size % 2);
  }

  if (!fmt || dataOffset < 0) {
    throw new Error(`incomplete WAV file: ${filePath}`);
  }

  const frameCount = Math.floor(dataSize / fmt.blockAlign);
  const out = new Float32Array(frameCount);
  const bytesPerSample = Math.floor(fmt.bitsPerSample / 8);

  for (let frame = 0; frame < frameCount; frame += 1) {
    let sum = 0.0;
    const frameBase = dataOffset + frame * fmt.blockAlign;
    for (let ch = 0; ch < fmt.channels; ch += 1) {
      const sampleOffset = frameBase + ch * bytesPerSample;
      let value = 0.0;
      if (fmt.audioFormat === 3) {
        if (fmt.bitsPerSample === 32) value = buf.readFloatLE(sampleOffset);
        else if (fmt.bitsPerSample === 64) value = buf.readDoubleLE(sampleOffset);
        else throw new Error(`unsupported float WAV bit depth: ${fmt.bitsPerSample}`);
      } else if (fmt.audioFormat === 1) {
        if (fmt.bitsPerSample === 8) {
          value = (buf.readUInt8(sampleOffset) - 128) / 127;
        } else if (fmt.bitsPerSample === 16) {
          value = buf.readInt16LE(sampleOffset) / 32767;
        } else if (fmt.bitsPerSample === 24) {
          const b0 = buf[sampleOffset];
          const b1 = buf[sampleOffset + 1];
          const b2 = buf[sampleOffset + 2];
          let sample = b0 | (b1 << 8) | (b2 << 16);
          if (sample & 0x800000) sample |= 0xff000000;
          value = sample / 8388607;
        } else if (fmt.bitsPerSample === 32) {
          value = buf.readInt32LE(sampleOffset) / 2147483647;
        } else {
          throw new Error(`unsupported PCM WAV bit depth: ${fmt.bitsPerSample}`);
        }
      } else {
        throw new Error(`unsupported WAV format code ${fmt.audioFormat} in ${filePath}`);
      }
      sum += value;
    }
    out[frame] = sum / fmt.channels;
  }

  return { sampleRate: fmt.sampleRate, samples: out };
}

function rms(samples) {
  if (!samples.length) return 0.0;
  let sumSq = 0.0;
  for (let i = 0; i < samples.length; i += 1) {
    const x = samples[i];
    sumSq += x * x;
  }
  return Math.sqrt(sumSq / samples.length);
}

function normalizeRms(samples, targetRms) {
  const current = rms(samples);
  if (!(current > 0.0) || !(targetRms > 0.0)) return current;
  const gain = targetRms / current;
  for (let i = 0; i < samples.length; i += 1) {
    samples[i] *= gain;
  }
  return current;
}

function resampleLinear(samples, srcRate, dstRate) {
  if (!samples.length || !srcRate || !dstRate) return new Float32Array(0);
  if (srcRate === dstRate) return Float32Array.from(samples);
  const ratio = dstRate / srcRate;
  const outLen = Math.max(1, Math.round(samples.length * ratio));
  const out = new Float32Array(outLen);
  const invRatio = srcRate / dstRate;
  for (let i = 0; i < outLen; i += 1) {
    const srcPos = i * invRatio;
    const srcI = Math.floor(srcPos);
    const frac = srcPos - srcI;
    const x0 = samples[Math.min(srcI, samples.length - 1)];
    const x1 = samples[Math.min(srcI + 1, samples.length - 1)];
    out[i] = x0 + (x1 - x0) * frac;
  }
  return out;
}

function trimAudioRegion(samples, sampleRate, startSec, endSec) {
  if (!samples.length) return new Float32Array(0);
  const startIdx = Math.min(samples.length, Math.floor(Math.max(0.0, startSec) * sampleRate));
  const endIdx = Math.min(samples.length, Math.ceil(Math.max(startSec, endSec) * sampleRate));
  if (endIdx <= startIdx) return new Float32Array(0);
  return samples.slice(startIdx, endIdx);
}

function buildHannWindow(winLength) {
  const window = new Float64Array(winLength);
  if (winLength <= 1) {
    window[0] = 1.0;
    return window;
  }
  const denom = winLength - 1;
  for (let n = 0; n < winLength; n += 1) {
    const phase = (2.0 * Math.PI * n) / denom;
    window[n] = 0.5 - 0.5 * Math.cos(phase);
  }
  return window;
}

function buildFftPlan(nfft) {
  const bits = Math.round(Math.log2(nfft));
  if (1 << bits !== nfft) {
    throw new Error(`nfft must be power of two, got ${nfft}`);
  }
  const bitrev = new Uint32Array(nfft);
  for (let i = 0; i < nfft; i += 1) {
    let x = i;
    let y = 0;
    for (let b = 0; b < bits; b += 1) {
      y = (y << 1) | (x & 1);
      x >>= 1;
    }
    bitrev[i] = y;
  }
  const cos = new Float64Array(nfft / 2);
  const sin = new Float64Array(nfft / 2);
  for (let i = 0; i < nfft / 2; i += 1) {
    const angle = (-2.0 * Math.PI * i) / nfft;
    cos[i] = Math.cos(angle);
    sin[i] = Math.sin(angle);
  }
  return { nfft, bitrev, cos, sin };
}

function fftInPlace(re, im, plan) {
  const n = plan.nfft;
  for (let i = 0; i < n; i += 1) {
    const j = plan.bitrev[i];
    if (j > i) {
      const tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      const ti = im[i];
      im[i] = im[j];
      im[j] = ti;
    }
  }

  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const step = n / len;
    for (let start = 0; start < n; start += len) {
      for (let k = 0; k < half; k += 1) {
        const tableIndex = k * step;
        const wr = plan.cos[tableIndex];
        const wi = plan.sin[tableIndex];
        const even = start + k;
        const odd = even + half;
        const tr = wr * re[odd] - wi * im[odd];
        const ti = wr * im[odd] + wi * re[odd];
        const ur = re[even];
        const ui = im[even];
        re[even] = ur + tr;
        im[even] = ui + ti;
        re[odd] = ur - tr;
        im[odd] = ui - ti;
      }
    }
  }
}

function pitchToHz(midi) {
  return 440.0 * Math.pow(2.0, (midi - 69.0) / 12.0);
}

function buildHarmonicMaps(sampleRate, nfft, midiMin, midiMax, maxHarmonics, localBandwidthBins) {
  const nyquist = sampleRate * 0.5;
  const hzPerBin = sampleRate / nfft;
  const lastBin = Math.floor(nfft / 2);
  const out = [];
  for (let midi = midiMin; midi <= midiMax; midi += 1) {
    const f0 = pitchToHz(midi);
    const ranges = [];
    for (let h = 1; h <= maxHarmonics; h += 1) {
      const harmonicHz = f0 * h;
      if (harmonicHz >= nyquist) break;
      const center = Math.round(harmonicHz / hzPerBin);
      const start = Math.max(0, center - localBandwidthBins);
      const end = Math.min(lastBin, center + localBandwidthBins);
      ranges.push({ start, end });
    }
    out.push({ midi, f0_hz: f0, ranges });
  }
  return out;
}

function makeMapIndex(maps) {
  const byMidi = new Map();
  for (const map of maps) byMidi.set(map.midi, map);
  return byMidi;
}

function isBetterIntervalMatch(candidate, previous) {
  return (
    candidate.har > previous.har ||
    (candidate.har === previous.har && Math.abs(candidate.centError) < Math.abs(previous.centError)) ||
    (candidate.har === previous.har &&
      Math.abs(candidate.centError) === Math.abs(previous.centError) &&
      candidate.h > previous.h) ||
    (candidate.har === previous.har &&
      Math.abs(candidate.centError) === Math.abs(previous.centError) &&
      candidate.h === previous.h &&
      candidate.mbw > previous.mbw)
  );
}

function buildRuntime() {
  const frontend = {
    winLength: 4096,
    hopLength: 512,
    nfft: 4096,
    window: buildHannWindow(4096),
    fft: buildFftPlan(4096),
    bins: 4096 / 2 + 1,
    re: new Float64Array(4096),
    im: new Float64Array(4096),
    mag: new Float32Array(4096 / 2 + 1),
  };
  const maps = buildHarmonicMaps(
    MASP_TUNED_PARAMS.strictSampleRate,
    frontend.nfft,
    MASP_TUNED_PARAMS.midiMin,
    88,
    MASP_TUNED_PARAMS.maxHarmonics,
    2
  );
  return {
    audioCache: new Map(),
    frontend,
    maps,
    mapByMidi: makeMapIndex(maps),
  };
}

function loadAudioCached(runtime, audioPath) {
  let audio = runtime.audioCache.get(audioPath);
  if (!audio) {
    audio = parseWavMono(path.resolve(repoRoot, audioPath));
    runtime.audioCache.set(audioPath, audio);
  }
  return audio;
}

function computeSegmentMetrics(runtime, req) {
  const audio = loadAudioCached(runtime, req.audio_path);
  const start = Math.max(0.0, req.start_sec - NOTE_CONTEXT_MARGIN_SEC);
  const end = Math.max(start, req.end_sec + NOTE_CONTEXT_MARGIN_SEC);
  const raw = trimAudioRegion(audio.samples, audio.sampleRate, start, end);
  if (!raw.length) {
    throw new Error(`empty segment for ${req.id ?? req.audio_path}`);
  }

  const noiseRms = Math.max(0.0, Math.min(1.0, rms(raw)));
  let segment =
    MASP_TUNED_PARAMS.mode === 'strict' && audio.sampleRate > MASP_TUNED_PARAMS.strictSampleRate
      ? resampleLinear(raw, audio.sampleRate, MASP_TUNED_PARAMS.strictSampleRate)
      : Float32Array.from(raw);
  const effectiveSampleRate =
    MASP_TUNED_PARAMS.mode === 'strict' && audio.sampleRate > MASP_TUNED_PARAMS.strictSampleRate
      ? MASP_TUNED_PARAMS.strictSampleRate
      : audio.sampleRate;
  normalizeRms(segment, TARGET_RMS);

  if (segment.length < runtime.frontend.winLength) {
    const padded = new Float32Array(runtime.frontend.winLength);
    padded.set(segment, 0);
    segment = padded;
  }

  let best = null;
  const frameCount = 1 + Math.floor((segment.length - runtime.frontend.winLength) / runtime.frontend.hopLength);
  for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
    const startIdx = frameIndex * runtime.frontend.hopLength;
    runtime.frontend.re.fill(0.0);
    runtime.frontend.im.fill(0.0);
    for (let i = 0; i < runtime.frontend.winLength; i += 1) {
      runtime.frontend.re[i] = segment[startIdx + i] * runtime.frontend.window[i];
    }
    fftInPlace(runtime.frontend.re, runtime.frontend.im, runtime.frontend.fft);
    for (let bin = 0; bin < runtime.frontend.bins; bin += 1) {
      const re = runtime.frontend.re[bin];
      const im = runtime.frontend.im[bin];
      runtime.frontend.mag[bin] = Math.hypot(re, im);
    }

    const pitchSpectrum = scoreMaspMidiFrame(runtime.frontend.mag, runtime.maps, MASP_TUNED_PARAMS);
    const candidate = {
      h: computeHarmonicityH(
        runtime.frontend.mag,
        pitchSpectrum,
        effectiveSampleRate,
        runtime.frontend.nfft,
        MASP_TUNED_PARAMS.midiMin
      ),
      har: computeHar(pitchSpectrum, req.expected_midis, MASP_TUNED_PARAMS.midiMin),
      mbw: computeMbw(runtime.frontend.mag, req.expected_midis, runtime.maps),
      centError: computeCentError(pitchSpectrum, req.expected_midis, MASP_TUNED_PARAMS.midiMin),
      noiseRms,
      pitchSpectrum,
    };
    if (!best || isBetterIntervalMatch(candidate, best)) {
      best = candidate;
    }
  }

  if (!best) {
    throw new Error(`no frontend frames for ${req.id ?? req.audio_path}`);
  }
  return best;
}

function evaluateRequest(runtime, req) {
  const t0 = performance.now();
  const metrics = computeSegmentMetrics(runtime, req);
  const decision = computeValidationDecision({
    metrics: {
      h: metrics.h,
      har: metrics.har,
      mbw: metrics.mbw,
      centError: metrics.centError,
      noiseRms: metrics.noiseRms,
    },
    hTarget: 1.0,
    params: MASP_TUNED_PARAMS,
  });
  const executionMs = performance.now() - t0;
  return {
    id: req.id ?? null,
    audio_path: req.audio_path,
    expected_midis: req.expected_midis,
    pass: decision.pass,
    reason: decision.reason,
    h_real: metrics.h,
    h_target: 1.0,
    h_dynamic_threshold: decision.hDynamicThreshold,
    har: metrics.har,
    mbw: metrics.mbw,
    cent_error: metrics.centError,
    cent_tolerance: MASP_TUNED_PARAMS.centTolerance,
    noise_rms: metrics.noiseRms,
    weighted_score: decision.weightedScore,
    score_threshold: MASP_TUNED_PARAMS.scoreThreshold,
    execution_ms: executionMs,
    expected_valid: req.expected_valid,
    false_accept: decision.pass && req.expected_valid === false,
    false_reject: !decision.pass && req.expected_valid === true,
  };
}

function compareWithRust(tsResults, rustResultsById) {
  let missing = 0;
  let passMismatch = 0;
  let scoreMismatch = 0;
  let harMismatch = 0;
  let mbwMismatch = 0;
  let centMismatch = 0;
  let hMismatch = 0;
  let maxScoreAbsDiff = 0.0;
  let maxHarAbsDiff = 0.0;
  let maxMbwAbsDiff = 0.0;
  let maxHAbsDiff = 0.0;

  for (const item of tsResults) {
    const rust = rustResultsById.get(item.id);
    if (!rust) {
      missing += 1;
      continue;
    }
    if (item.pass !== rust.pass) passMismatch += 1;
    const scoreDiff = Math.abs(item.weighted_score - rust.weighted_score);
    const harDiff = Math.abs(item.har - rust.har);
    const mbwDiff = Math.abs(item.mbw - rust.mbw);
    const hDiff = Math.abs(item.h_real - rust.h_real);
    const centDiff = Math.abs(item.cent_error - rust.cent_error);
    maxScoreAbsDiff = Math.max(maxScoreAbsDiff, scoreDiff);
    maxHarAbsDiff = Math.max(maxHarAbsDiff, harDiff);
    maxMbwAbsDiff = Math.max(maxMbwAbsDiff, mbwDiff);
    maxHAbsDiff = Math.max(maxHAbsDiff, hDiff);
    if (scoreDiff > 1e-4) scoreMismatch += 1;
    if (harDiff > 1e-4) harMismatch += 1;
    if (mbwDiff > 1e-4) mbwMismatch += 1;
    if (hDiff > 1e-4) hMismatch += 1;
    if (centDiff > REPORT_EPS) centMismatch += 1;
  }

  return {
    total: tsResults.length,
    missing,
    passMismatch,
    scoreMismatch,
    harMismatch,
    mbwMismatch,
    centMismatch,
    hMismatch,
    maxScoreAbsDiff,
    maxHarAbsDiff,
    maxMbwAbsDiff,
    maxHAbsDiff,
  };
}

function loadRustResultsById(filePath) {
  const map = new Map();
  for (const row of readJsonl(filePath)) {
    map.set(row.id, row);
  }
  return map;
}

function runSplit(runtime, splitName, requestsPath, rustResultsPath, outDir) {
  const args = parseArgs(process.argv.slice(2));
  let requests = readJsonl(requestsPath);
  if (args.limit !== null) {
    requests = requests.slice(0, args.limit);
  }
  const rustResultsById = fs.existsSync(rustResultsPath)
    ? loadRustResultsById(rustResultsPath)
    : new Map();

  const results = [];
  const started = performance.now();
  for (let i = 0; i < requests.length; i += 1) {
    const result = evaluateRequest(runtime, requests[i]);
    results.push(result);
    if ((i + 1) % 2000 === 0 || i + 1 === requests.length) {
      const elapsed = (performance.now() - started) / 1000.0;
      console.log(`[${splitName}] ${i + 1}/${requests.length} requests in ${elapsed.toFixed(1)}s`);
    }
  }

  ensureDir(outDir);
  writeJsonl(path.join(outDir, 'ts_masp_validation_results.jsonl'), results);
  const summary = summarizeClassification(results);
  const timings = summarizeTimings(results);
  const rustConsistency = compareWithRust(results, rustResultsById);
  const report = {
    split: splitName,
    total: requests.length,
    metrics: summary,
    timings,
    rust_consistency: rustConsistency,
  };
  writeJson(path.join(outDir, 'ts_masp_validation_summary.json'), report);
  return { requests, results, summary, timings, rustConsistency };
}

function main() {
  assertNodeV22();
  const args = parseArgs(process.argv.slice(2));

  const runtime = buildRuntime();
  const mono =
    args.split === 'all' || args.split === 'mono'
      ? runSplit(
          runtime,
          'mono',
          path.join(outputDir, 'masp_eval_mono_requests.jsonl'),
          path.join(outputDir, 'masp_eval_mono', 'masp_validation_results.jsonl'),
          path.join(outputDir, 'ts_masp_eval_mono')
        )
      : { results: [], summary: summarizeClassification([]), timings: summarizeTimings([]), rustConsistency: compareWithRust([], new Map()) };
  const comp =
    args.split === 'all' || args.split === 'comp'
      ? runSplit(
          runtime,
          'comp',
          path.join(outputDir, 'masp_eval_comp_requests.jsonl'),
          path.join(outputDir, 'masp_eval_comp', 'masp_validation_results.jsonl'),
          path.join(outputDir, 'ts_masp_eval_comp')
        )
      : { results: [], summary: summarizeClassification([]), timings: summarizeTimings([]), rustConsistency: compareWithRust([], new Map()) };

  const overallResults = mono.results.concat(comp.results);
  const overallSummary = summarizeClassification(overallResults);
  const overallTimings = summarizeTimings(overallResults);
  const overallRustConsistency = compareWithRust(
    overallResults,
    new Map([
      ...loadRustResultsById(path.join(outputDir, 'masp_eval_mono', 'masp_validation_results.jsonl')),
      ...loadRustResultsById(path.join(outputDir, 'masp_eval_comp', 'masp_validation_results.jsonl')),
    ])
  );

  const report = {
    params: MASP_TUNED_PARAMS,
    mono: { metrics: mono.summary, timings: mono.timings, rust_consistency: mono.rustConsistency },
    comp: { metrics: comp.summary, timings: comp.timings, rust_consistency: comp.rustConsistency },
    overall: {
      metrics: overallSummary,
      timings: overallTimings,
      rust_consistency: overallRustConsistency,
    },
  };

  writeJson(path.join(outputDir, 'ts_masp_precision_recall_report.json'), report);
  writeText(
    path.join(outputDir, 'ts_masp_precision_recall_report.md'),
    [
      '# TS MASP Benchmark',
      '',
      `Parameters: bExponent=${MASP_TUNED_PARAMS.bExponent}, scoreThreshold=${MASP_TUNED_PARAMS.scoreThreshold}`,
      `Weights: har=${MASP_TUNED_PARAMS.scoreWeights.har}, mbw=${MASP_TUNED_PARAMS.scoreWeights.mbw}, cent=${MASP_TUNED_PARAMS.scoreWeights.cent}, rms=${MASP_TUNED_PARAMS.scoreWeights.rms}`,
      '',
      formatMetricBlock('MONO', mono.summary, mono.timings),
      '',
      formatMetricBlock('COMP', comp.summary, comp.timings),
      '',
      formatMetricBlock('OVERALL', overallSummary, overallTimings),
      '',
      'Rust Consistency',
      '',
      `MONO pass mismatches: ${mono.rustConsistency.passMismatch} / ${mono.rustConsistency.total}`,
      `COMP pass mismatches: ${comp.rustConsistency.passMismatch} / ${comp.rustConsistency.total}`,
      `OVERALL pass mismatches: ${overallRustConsistency.passMismatch} / ${overallRustConsistency.total}`,
      `OVERALL max abs diffs: score=${overallRustConsistency.maxScoreAbsDiff.toExponential(6)}, har=${overallRustConsistency.maxHarAbsDiff.toExponential(6)}, mbw=${overallRustConsistency.maxMbwAbsDiff.toExponential(6)}, h=${overallRustConsistency.maxHAbsDiff.toExponential(6)}`,
    ].join('\n') + '\n'
  );

  console.log('TS MASP benchmark completed');
  console.log(formatMetricBlock('MONO', mono.summary, mono.timings));
  console.log('');
  console.log(formatMetricBlock('COMP', comp.summary, comp.timings));
  console.log('');
  console.log(formatMetricBlock('OVERALL', overallSummary, overallTimings));
  console.log('');
  console.log(
    `Rust consistency: overall pass mismatches ${overallRustConsistency.passMismatch}/${overallRustConsistency.total}, max score diff ${overallRustConsistency.maxScoreAbsDiff}`
  );
}

main();

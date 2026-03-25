import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import * as masp from "./maspCore.mjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const fixturePath = path.join(__dirname, "consistencyFixtures.json");
const tsPath = path.join(__dirname, "maspCore.ts");
const jsPath = path.join(__dirname, "maspCore.mjs");
const fixture = JSON.parse(fs.readFileSync(fixturePath, "utf8"));

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function approxEqual(actual, expected, eps = 1e-9, label = "value") {
  if (Math.abs(actual - expected) > eps) {
    throw new Error(`${label} mismatch: got ${actual}, expected ${expected}`);
  }
}

function deepEqual(actual, expected, label = "value") {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  if (a !== e) {
    throw new Error(`${label} mismatch:\nactual:   ${a}\nexpected: ${e}`);
  }
}

function run() {
  assert(
    process.versions.node.startsWith("22."),
    `Node v22 is required for these checks, got ${process.version}`
  );

  const tsSource = fs.readFileSync(tsPath, "utf8");
  const jsSource = fs.readFileSync(jsPath, "utf8");
  assert(
    tsSource === jsSource,
    "maspCore.ts and maspCore.mjs must stay byte-identical"
  );

  for (const item of fixture.smoothingFactor) {
    approxEqual(
      masp.smoothingFactor(item.k, item.bExponent),
      item.expected,
      1e-12,
      `smoothingFactor(k=${item.k})`
    );
  }

  for (const item of fixture.centScore) {
    approxEqual(
      masp.centScore(item.centError, item.tolerance),
      item.expected,
      1e-12,
      `centScore(${item.centError})`
    );
  }

  const frame = fixture.frameScoring;
  const pitchSpectrum = masp.scoreMaspMidiFrame(frame.mag, frame.maps, {
    ...fixture.params,
    maxHarmonics: 3,
  });
  for (let i = 0; i < pitchSpectrum.length; i += 1) {
    approxEqual(
      pitchSpectrum[i],
      frame.pitchSpectrum[i],
      1e-12,
      `pitchSpectrum[${i}]`
    );
  }

  approxEqual(
    masp.computeHarmonicityH(
      frame.mag,
      pitchSpectrum,
      frame.sampleRate,
      frame.nfft,
      frame.midiMin
    ),
    frame.metrics.h,
    1e-12,
    "computeHarmonicityH"
  );
  approxEqual(
    masp.computeHar(pitchSpectrum, frame.expectedMidis, frame.midiMin),
    frame.metrics.har,
    1e-12,
    "computeHar"
  );
  approxEqual(
    masp.computeMbw(frame.mag, frame.expectedMidis, frame.maps),
    frame.metrics.mbw,
    1e-12,
    "computeMbw"
  );
  approxEqual(
    masp.computeCentError(pitchSpectrum, frame.expectedMidis, frame.midiMin),
    frame.metrics.centError,
    1e-12,
    "computeCentError"
  );
  approxEqual(
    masp.weightedMetricsScore(
      frame.metrics,
      fixture.params.scoreWeights,
      fixture.params.centTolerance
    ),
    frame.weightedMetricsScore,
    1e-12,
    "weightedMetricsScore"
  );
  deepEqual(
    masp.computeValidationDecision({
      metrics: frame.metrics,
      hTarget: 1.75,
      params: fixture.params,
    }),
    frame.validationDecision,
    "computeValidationDecision"
  );

  const playhead = fixture.playheadResolution;
  deepEqual(
    masp.resolveTablaturePlayhead(playhead.events, playhead.playheadSec),
    playhead.resolved,
    "resolveTablaturePlayhead"
  );

  console.log("TS MASP consistency checks passed");
}

run();

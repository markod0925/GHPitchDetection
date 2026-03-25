use std::time::{Duration, Instant};

use crate::{
    error::RuntimeError,
    model::FretNetRuntime,
    types::{FeatureBatch, ModelOutput},
};

#[derive(Debug, Clone)]
pub struct BenchmarkSummary {
    pub load_time: Duration,
    pub first_run: Duration,
    pub average_run: Duration,
    pub min_run: Duration,
    pub max_run: Duration,
    pub p95_run: Option<Duration>,
    pub iterations: usize,
    pub last_output: ModelOutput,
}

pub fn benchmark_runtime(
    runtime: &mut FretNetRuntime,
    features: &FeatureBatch,
    iterations: usize,
    load_time: Duration,
) -> Result<BenchmarkSummary, RuntimeError> {
    let iterations = iterations.max(1);

    let first_start = Instant::now();
    let mut last_output = runtime.infer_features(features)?;
    let first_run = first_start.elapsed();

    let mut runs = Vec::with_capacity(iterations);
    runs.push(first_run);

    for _ in 1..iterations {
        let start = Instant::now();
        last_output = runtime.infer_features(features)?;
        runs.push(start.elapsed());
    }

    let total = runs.iter().copied().fold(Duration::ZERO, |acc, next| acc + next);
    let average_run = total / runs.len() as u32;
    let min_run = runs.iter().copied().min().unwrap_or(Duration::ZERO);
    let max_run = runs.iter().copied().max().unwrap_or(Duration::ZERO);

    let mut sorted = runs.clone();
    sorted.sort_unstable();
    let p95_index = ((sorted.len() as f32 * 0.95).ceil() as usize).saturating_sub(1);
    let p95_run = sorted.get(p95_index).copied();

    Ok(BenchmarkSummary {
        load_time,
        first_run,
        average_run,
        min_run,
        max_run,
        p95_run,
        iterations: runs.len(),
        last_output,
    })
}


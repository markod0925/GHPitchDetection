use std::path::PathBuf;

use anyhow::{anyhow, bail};
use clap::Parser;
use fretnet_runtime::{
    benchmark_frontend_profile_audio, default_existing_stage_audio_inputs, FeatureExtractor,
    FrontendConfig, FrontendHarmonicProfileSummary, FrontendInitProfile,
    FrontendProfileBenchmarkSummary, ScalarProfileSummary,
};
use serde::Serialize;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    audio_path: Vec<PathBuf>,
    #[arg(long, default_value_t = 10)]
    iterations: usize,
    #[arg(long)]
    max_frames: Option<usize>,
    #[arg(long)]
    json_output: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct AudioBenchmarkTarget {
    display_name: String,
    audio_path: PathBuf,
}

#[derive(Debug, Serialize)]
struct AggregateSummary {
    input_count: usize,
    frontend_mean_seconds: f64,
    slowest_harmonic: Option<f32>,
    slowest_harmonic_mean_seconds: f64,
    slowest_octave: Option<usize>,
    slowest_octave_mean_seconds: f64,
    average_hcqt_assembly_fraction: f64,
    average_batch_conversion_fraction: f64,
    downsample_convolution_mean_seconds: f64,
    downsample_scale_mean_seconds: f64,
    downsample_allocation_mean_seconds: f64,
    downsample_kernel_prepare_mean_seconds: f64,
    top_cost_centers: Vec<CostCenter>,
    internal_cost_centers: Vec<CostCenter>,
    repeated_setup_candidates: Vec<CostCenter>,
    init_only_summary: InitOnlySummary,
}

#[derive(Debug, Serialize)]
struct CostCenter {
    name: String,
    mean_seconds: f64,
    mean_fraction_of_frontend: f64,
}

#[derive(Debug, Serialize)]
struct InitOnlySummary {
    total_seconds: f64,
    fft_plan_seconds: f64,
    filter_build_seconds: f64,
    sparsify_seconds: f64,
    fft_basis_bytes: usize,
}

#[derive(Debug, Serialize)]
struct JsonReport {
    extractor_init: FrontendInitProfile,
    inputs: Vec<FrontendProfileBenchmarkSummary>,
    aggregate: AggregateSummary,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let targets = benchmark_targets(&args.audio_path)?;
    if targets.is_empty() {
        bail!("no frontend benchmark audio inputs available");
    }

    let frontend_config = FrontendConfig::default();
    let (extractor, init_profile) = FeatureExtractor::new_profiled(frontend_config)?;

    println!("Frontend profiling benchmark");
    println!("Iterations per input: {}", args.iterations.max(1));
    if let Some(max_frames) = args.max_frames {
        println!("Max frames: {}", max_frames);
    } else {
        println!("Max frames: full audio");
    }
    print_init_profile(&init_profile);

    let mut summaries = Vec::with_capacity(targets.len());
    for target in targets {
        let summary = benchmark_frontend_profile_audio(
            &extractor,
            &target.audio_path,
            args.iterations,
            args.max_frames,
        )?;
        print_input_summary(&target.display_name, &summary);
        summaries.push(summary);
    }

    let aggregate = build_aggregate_summary(&init_profile, &summaries)?;
    print_aggregate_summary(&aggregate);

    if let Some(json_output) = args.json_output {
        let report = JsonReport {
            extractor_init: init_profile,
            inputs: summaries,
            aggregate,
        };
        if let Some(parent) = json_output.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&json_output, serde_json::to_vec_pretty(&report)?)?;
        println!();
        println!("JSON report: {}", json_output.display());
    }

    Ok(())
}

fn benchmark_targets(
    explicit_audio_paths: &[PathBuf],
) -> anyhow::Result<Vec<AudioBenchmarkTarget>> {
    if !explicit_audio_paths.is_empty() {
        return Ok(explicit_audio_paths
            .iter()
            .map(|path| AudioBenchmarkTarget {
                display_name: path
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .unwrap_or("explicit_audio")
                    .to_owned(),
                audio_path: path.clone(),
            })
            .collect());
    }

    let discovered = default_existing_stage_audio_inputs()?;
    if discovered.is_empty() {
        bail!(
            "no default real-audio profiling inputs found; generate the stage fixtures or pass --audio-path explicitly"
        );
    }

    Ok(discovered
        .into_iter()
        .map(|input| AudioBenchmarkTarget {
            display_name: input.fixture_name,
            audio_path: input.audio_path,
        })
        .collect())
}

fn print_init_profile(profile: &FrontendInitProfile) {
    let mut fft_plan_seconds = 0.0f64;
    let mut filter_build_seconds = 0.0f64;
    let mut sparsify_seconds = 0.0f64;
    let mut fft_basis_bytes = 0usize;
    let mut basis_nonzero_count = 0usize;
    let mut basis_total_count = 0usize;

    for harmonic in &profile.harmonic_profiles {
        for octave in &harmonic.octave_profiles {
            fft_plan_seconds += octave.fft_plan_seconds;
            filter_build_seconds += octave.filter_build_seconds;
            sparsify_seconds += octave.sparsify_seconds;
            fft_basis_bytes += octave.fft_basis_bytes;
            basis_nonzero_count += octave.basis_nonzero_count;
            basis_total_count += octave.row_count * octave.positive_bins;
        }
    }

    println!("Extractor init:");
    println!("  total: {}", fmt_seconds(profile.total_seconds));
    println!(
        "  init-only work: fft_plan={} filter_build={} sparsify={} fft_basis={} active_basis={:.1}%",
        fmt_seconds(fft_plan_seconds),
        fmt_seconds(filter_build_seconds),
        fmt_seconds(sparsify_seconds),
        fmt_bytes(fft_basis_bytes as f64),
        safe_fraction(basis_nonzero_count as f64, basis_total_count as f64) * 100.0
    );
    println!("  note: basis construction and FFT planning occur only here, not during steady-state extract()");
}

fn print_input_summary(display_name: &str, summary: &FrontendProfileBenchmarkSummary) {
    println!();
    println!("Input: {}", display_name);
    println!("  Audio path: {}", summary.audio_path.display());
    println!(
        "  Prepared audio: duration={:.3}s sr={} samples={} prep={}",
        summary.duration_seconds,
        summary.prepared_sample_rate,
        summary.prepared_num_samples,
        fmt_seconds(summary.audio_prepare_seconds)
    );
    println!("  HCQT shape: {:?}", summary.hcqt_shape);
    println!("  Batch shape: {:?}", summary.batch_shape);
    print_scalar("  Frontend total", &summary.frontend_total);
    print_scalar("  Shared octave pyramid", &summary.octave_pyramid);
    print_scalar("  HCQT assembly", &summary.hcqt_assembly);
    print_scalar("  HCQT -> batch", &summary.batch_conversion);
    println!(
        "  Shared octave pyramid allocation: {}",
        fmt_bytes(summary.octave_pyramid_allocation_bytes)
    );
    for octave in &summary.octave_pyramid_levels {
        println!(
            "    shared octave {}: hop={} input_samples={} output_samples={} total={} downsample={} conv={} kernel_prep={} alloc={} resize={} scale={} taps=({}, {}) specialized={}",
            octave.octave_index,
            octave.hop_length,
            octave.input_samples,
            octave.output_samples,
            fmt_seconds(octave.total.mean_seconds),
            fmt_seconds(octave.downsample.mean_seconds),
            fmt_seconds(octave.convolution.mean_seconds),
            fmt_seconds(octave.kernel_prepare.mean_seconds),
            fmt_seconds(octave.output_allocation.mean_seconds),
            fmt_seconds(octave.resize.mean_seconds),
            fmt_seconds(octave.scale.mean_seconds),
            octave.mean_left_tap_count.round() as usize,
            octave.mean_right_tap_count.round() as usize,
            octave.specialized_path
        );
    }

    for harmonic in &summary.harmonics {
        let fraction = safe_fraction(
            harmonic.total.mean_seconds,
            summary.frontend_total.mean_seconds,
        );
        println!(
            "  Harmonic {:>3}: mean={} ({:.1}% of frontend) temp_alloc={}",
            harmonic.harmonic,
            fmt_seconds(harmonic.total.mean_seconds),
            fraction * 100.0,
            fmt_bytes(harmonic.mean_temporary_allocation_bytes)
        );
        println!(
            "    breakdown: clone={} trim_stack={} normalize={} flatten={} post_process={}",
            fmt_seconds(harmonic.audio_clone.mean_seconds),
            fmt_seconds(harmonic.trim_stack.mean_seconds),
            fmt_seconds(harmonic.normalization.mean_seconds),
            fmt_seconds(harmonic.flatten.mean_seconds),
            fmt_seconds(harmonic.post_process.mean_seconds)
        );
        for octave in &harmonic.octaves {
            println!(
                "    octave {}: total={} transform={} fft={} dot={} loop_overhead={} lookup={} mac={} write={} copy={} alloc={} array={} downsample={} active_basis={:.1}% alloc_bytes={}",
                octave.octave_index,
                fmt_seconds(octave.total.mean_seconds),
                fmt_seconds(octave.transform.mean_seconds),
                fmt_seconds(octave.fft_execution.mean_seconds),
                fmt_seconds(octave.dot_product.mean_seconds),
                fmt_seconds(octave.dot_loop_overhead.mean_seconds),
                fmt_seconds(octave.basis_lookup.mean_seconds),
                fmt_seconds(octave.multiply_accumulate.mean_seconds),
                fmt_seconds(octave.output_write.mean_seconds),
                fmt_seconds(octave.frame_copy.mean_seconds),
                fmt_seconds(octave.buffer_allocation.mean_seconds),
                fmt_seconds(octave.array_materialization.mean_seconds),
                fmt_seconds(octave.downsample.mean_seconds),
                safe_fraction(
                    octave.mean_active_basis_coefficients,
                    octave.mean_basis_coefficients_total,
                ) * 100.0,
                fmt_bytes(octave.mean_buffer_allocation_bytes)
            );
        }
    }
}

fn build_aggregate_summary(
    init_profile: &FrontendInitProfile,
    summaries: &[FrontendProfileBenchmarkSummary],
) -> anyhow::Result<AggregateSummary> {
    if summaries.is_empty() {
        return Err(anyhow!("no frontend summaries collected"));
    }

    let frontend_mean_seconds = summaries
        .iter()
        .map(|summary| summary.frontend_total.mean_seconds)
        .sum::<f64>()
        / summaries.len() as f64;
    let average_hcqt_assembly_fraction = summaries
        .iter()
        .map(|summary| {
            safe_fraction(
                summary.hcqt_assembly.mean_seconds,
                summary.frontend_total.mean_seconds,
            )
        })
        .sum::<f64>()
        / summaries.len() as f64;
    let average_octave_pyramid_fraction = summaries
        .iter()
        .map(|summary| {
            safe_fraction(
                summary.octave_pyramid.mean_seconds,
                summary.frontend_total.mean_seconds,
            )
        })
        .sum::<f64>()
        / summaries.len() as f64;
    let average_batch_conversion_fraction = summaries
        .iter()
        .map(|summary| {
            safe_fraction(
                summary.batch_conversion.mean_seconds,
                summary.frontend_total.mean_seconds,
            )
        })
        .sum::<f64>()
        / summaries.len() as f64;

    let harmonic_count = summaries
        .first()
        .map(|summary| summary.harmonics.len())
        .unwrap_or(0);
    let mut harmonic_means = Vec::with_capacity(harmonic_count);
    for harmonic_index in 0..harmonic_count {
        let harmonic = summaries[0].harmonics[harmonic_index].harmonic;
        let mean_seconds = summaries
            .iter()
            .map(|summary| summary.harmonics[harmonic_index].total.mean_seconds)
            .sum::<f64>()
            / summaries.len() as f64;
        harmonic_means.push((harmonic, mean_seconds));
    }
    let slowest_harmonic = harmonic_means
        .iter()
        .max_by(|left, right| left.1.partial_cmp(&right.1).unwrap())
        .copied();
    let octave_count = summaries
        .first()
        .map(|summary| summary.octave_pyramid_levels.len())
        .unwrap_or(0);
    let mut octave_means = Vec::with_capacity(octave_count);
    for octave_index in 0..octave_count {
        let mean_seconds = summaries
            .iter()
            .map(|summary| {
                summary.octave_pyramid_levels[octave_index]
                    .total
                    .mean_seconds
            })
            .sum::<f64>()
            / summaries.len() as f64;
        octave_means.push((octave_index, mean_seconds));
    }
    let slowest_octave = octave_means
        .iter()
        .max_by(|left, right| left.1.partial_cmp(&right.1).unwrap())
        .copied();

    let mut top_cost_centers = harmonic_means
        .iter()
        .map(|(harmonic, mean_seconds)| CostCenter {
            name: format!("harmonic_{harmonic}"),
            mean_seconds: *mean_seconds,
            mean_fraction_of_frontend: safe_fraction(*mean_seconds, frontend_mean_seconds),
        })
        .collect::<Vec<_>>();
    top_cost_centers.push(CostCenter {
        name: "hcqt_assembly".to_owned(),
        mean_seconds: summaries
            .iter()
            .map(|summary| summary.hcqt_assembly.mean_seconds)
            .sum::<f64>()
            / summaries.len() as f64,
        mean_fraction_of_frontend: average_hcqt_assembly_fraction,
    });
    top_cost_centers.push(CostCenter {
        name: "hcqt_to_batch".to_owned(),
        mean_seconds: summaries
            .iter()
            .map(|summary| summary.batch_conversion.mean_seconds)
            .sum::<f64>()
            / summaries.len() as f64,
        mean_fraction_of_frontend: average_batch_conversion_fraction,
    });
    top_cost_centers
        .sort_by(|left, right| right.mean_seconds.partial_cmp(&left.mean_seconds).unwrap());
    top_cost_centers.truncate(3);

    let mut internal_cost_centers = vec![
        CostCenter {
            name: "octave_downsample".to_owned(),
            mean_seconds: summaries
                .iter()
                .map(|summary| summary.octave_pyramid.mean_seconds)
                .sum::<f64>()
                / summaries.len() as f64,
            mean_fraction_of_frontend: average_octave_pyramid_fraction,
        },
        CostCenter {
            name: "octave_downsample_convolution".to_owned(),
            mean_seconds: total_over_pyramid_levels(summaries, |octave| {
                octave.convolution.mean_seconds
            }),
            mean_fraction_of_frontend: safe_fraction(
                total_over_pyramid_levels(summaries, |octave| octave.convolution.mean_seconds),
                frontend_mean_seconds,
            ),
        },
        CostCenter {
            name: "octave_downsample_scale".to_owned(),
            mean_seconds: total_over_pyramid_levels(summaries, |octave| octave.scale.mean_seconds),
            mean_fraction_of_frontend: safe_fraction(
                total_over_pyramid_levels(summaries, |octave| octave.scale.mean_seconds),
                frontend_mean_seconds,
            ),
        },
        CostCenter {
            name: "octave_downsample_allocation".to_owned(),
            mean_seconds: total_over_pyramid_levels(summaries, |octave| {
                octave.output_allocation.mean_seconds
            }),
            mean_fraction_of_frontend: safe_fraction(
                total_over_pyramid_levels(summaries, |octave| {
                    octave.output_allocation.mean_seconds
                }),
                frontend_mean_seconds,
            ),
        },
        CostCenter {
            name: "octave_downsample_kernel_prepare".to_owned(),
            mean_seconds: total_over_pyramid_levels(summaries, |octave| {
                octave.kernel_prepare.mean_seconds
            }),
            mean_fraction_of_frontend: safe_fraction(
                total_over_pyramid_levels(summaries, |octave| octave.kernel_prepare.mean_seconds),
                frontend_mean_seconds,
            ),
        },
        CostCenter {
            name: "octave_dot_product".to_owned(),
            mean_seconds: mean_stage_total(
                summaries,
                |_, octave| octave.dot_product.mean_seconds,
                |_| 0.0,
            ),
            mean_fraction_of_frontend: safe_fraction(
                mean_stage_total(
                    summaries,
                    |_, octave| octave.dot_product.mean_seconds,
                    |_| 0.0,
                ),
                frontend_mean_seconds,
            ),
        },
        CostCenter {
            name: "octave_dot_loop_overhead".to_owned(),
            mean_seconds: mean_stage_total(
                summaries,
                |_, octave| octave.dot_loop_overhead.mean_seconds,
                |_| 0.0,
            ),
            mean_fraction_of_frontend: safe_fraction(
                mean_stage_total(
                    summaries,
                    |_, octave| octave.dot_loop_overhead.mean_seconds,
                    |_| 0.0,
                ),
                frontend_mean_seconds,
            ),
        },
        CostCenter {
            name: "octave_mac".to_owned(),
            mean_seconds: mean_stage_total(
                summaries,
                |_, octave| octave.multiply_accumulate.mean_seconds,
                |_| 0.0,
            ),
            mean_fraction_of_frontend: safe_fraction(
                mean_stage_total(
                    summaries,
                    |_, octave| octave.multiply_accumulate.mean_seconds,
                    |_| 0.0,
                ),
                frontend_mean_seconds,
            ),
        },
        CostCenter {
            name: "octave_basis_lookup".to_owned(),
            mean_seconds: mean_stage_total(
                summaries,
                |_, octave| octave.basis_lookup.mean_seconds,
                |_| 0.0,
            ),
            mean_fraction_of_frontend: safe_fraction(
                mean_stage_total(
                    summaries,
                    |_, octave| octave.basis_lookup.mean_seconds,
                    |_| 0.0,
                ),
                frontend_mean_seconds,
            ),
        },
        CostCenter {
            name: "octave_output_write".to_owned(),
            mean_seconds: mean_stage_total(
                summaries,
                |_, octave| octave.output_write.mean_seconds,
                |_| 0.0,
            ),
            mean_fraction_of_frontend: safe_fraction(
                mean_stage_total(
                    summaries,
                    |_, octave| octave.output_write.mean_seconds,
                    |_| 0.0,
                ),
                frontend_mean_seconds,
            ),
        },
        CostCenter {
            name: "octave_fft_execution".to_owned(),
            mean_seconds: mean_stage_total(
                summaries,
                |_, octave| octave.fft_execution.mean_seconds,
                |_| 0.0,
            ),
            mean_fraction_of_frontend: safe_fraction(
                mean_stage_total(
                    summaries,
                    |_, octave| octave.fft_execution.mean_seconds,
                    |_| 0.0,
                ),
                frontend_mean_seconds,
            ),
        },
        CostCenter {
            name: "harmonic_post_process".to_owned(),
            mean_seconds: mean_stage_total(
                summaries,
                |_, _| 0.0,
                |harmonic| harmonic.post_process.mean_seconds,
            ),
            mean_fraction_of_frontend: safe_fraction(
                mean_stage_total(
                    summaries,
                    |_, _| 0.0,
                    |harmonic| harmonic.post_process.mean_seconds,
                ),
                frontend_mean_seconds,
            ),
        },
        CostCenter {
            name: "harmonic_trim_stack".to_owned(),
            mean_seconds: mean_stage_total(
                summaries,
                |_, _| 0.0,
                |harmonic| harmonic.trim_stack.mean_seconds,
            ),
            mean_fraction_of_frontend: safe_fraction(
                mean_stage_total(
                    summaries,
                    |_, _| 0.0,
                    |harmonic| harmonic.trim_stack.mean_seconds,
                ),
                frontend_mean_seconds,
            ),
        },
        CostCenter {
            name: "hcqt_assembly".to_owned(),
            mean_seconds: summaries
                .iter()
                .map(|summary| summary.hcqt_assembly.mean_seconds)
                .sum::<f64>()
                / summaries.len() as f64,
            mean_fraction_of_frontend: average_hcqt_assembly_fraction,
        },
        CostCenter {
            name: "hcqt_to_batch".to_owned(),
            mean_seconds: summaries
                .iter()
                .map(|summary| summary.batch_conversion.mean_seconds)
                .sum::<f64>()
                / summaries.len() as f64,
            mean_fraction_of_frontend: average_batch_conversion_fraction,
        },
    ];
    internal_cost_centers
        .sort_by(|left, right| right.mean_seconds.partial_cmp(&left.mean_seconds).unwrap());
    internal_cost_centers.truncate(5);

    let repeated_setup_candidates = vec![
        CostCenter {
            name: "harmonic_audio_clone".to_owned(),
            mean_seconds: mean_over_harmonics(summaries, |harmonic| {
                harmonic.audio_clone.mean_seconds
            }),
            mean_fraction_of_frontend: safe_fraction(
                mean_over_harmonics(summaries, |harmonic| harmonic.audio_clone.mean_seconds),
                frontend_mean_seconds,
            ),
        },
        CostCenter {
            name: "octave_buffer_allocation".to_owned(),
            mean_seconds: mean_over_octaves(summaries, |octave| {
                octave.buffer_allocation.mean_seconds
            }),
            mean_fraction_of_frontend: safe_fraction(
                mean_over_octaves(summaries, |octave| octave.buffer_allocation.mean_seconds),
                frontend_mean_seconds,
            ),
        },
        CostCenter {
            name: "octave_array_materialization".to_owned(),
            mean_seconds: mean_over_octaves(summaries, |octave| {
                octave.array_materialization.mean_seconds
            }),
            mean_fraction_of_frontend: safe_fraction(
                mean_over_octaves(summaries, |octave| {
                    octave.array_materialization.mean_seconds
                }),
                frontend_mean_seconds,
            ),
        },
        CostCenter {
            name: "octave_pyramid_output_allocation".to_owned(),
            mean_seconds: total_over_pyramid_levels(summaries, |octave| {
                octave.output_allocation.mean_seconds
            }),
            mean_fraction_of_frontend: safe_fraction(
                total_over_pyramid_levels(summaries, |octave| {
                    octave.output_allocation.mean_seconds
                }),
                frontend_mean_seconds,
            ),
        },
        CostCenter {
            name: "octave_pyramid_kernel_prepare".to_owned(),
            mean_seconds: total_over_pyramid_levels(summaries, |octave| {
                octave.kernel_prepare.mean_seconds
            }),
            mean_fraction_of_frontend: safe_fraction(
                total_over_pyramid_levels(summaries, |octave| octave.kernel_prepare.mean_seconds),
                frontend_mean_seconds,
            ),
        },
    ];

    let mut fft_plan_seconds = 0.0f64;
    let mut filter_build_seconds = 0.0f64;
    let mut sparsify_seconds = 0.0f64;
    let mut fft_basis_bytes = 0usize;
    for harmonic in &init_profile.harmonic_profiles {
        for octave in &harmonic.octave_profiles {
            fft_plan_seconds += octave.fft_plan_seconds;
            filter_build_seconds += octave.filter_build_seconds;
            sparsify_seconds += octave.sparsify_seconds;
            fft_basis_bytes += octave.fft_basis_bytes;
        }
    }

    Ok(AggregateSummary {
        input_count: summaries.len(),
        frontend_mean_seconds,
        slowest_harmonic: slowest_harmonic.map(|value| value.0),
        slowest_harmonic_mean_seconds: slowest_harmonic.map(|value| value.1).unwrap_or(0.0),
        slowest_octave: slowest_octave.map(|value| value.0),
        slowest_octave_mean_seconds: slowest_octave.map(|value| value.1).unwrap_or(0.0),
        average_hcqt_assembly_fraction,
        average_batch_conversion_fraction,
        downsample_convolution_mean_seconds: total_over_pyramid_levels(summaries, |octave| {
            octave.convolution.mean_seconds
        }),
        downsample_scale_mean_seconds: total_over_pyramid_levels(summaries, |octave| {
            octave.scale.mean_seconds
        }),
        downsample_allocation_mean_seconds: total_over_pyramid_levels(summaries, |octave| {
            octave.output_allocation.mean_seconds
        }),
        downsample_kernel_prepare_mean_seconds: total_over_pyramid_levels(summaries, |octave| {
            octave.kernel_prepare.mean_seconds
        }),
        top_cost_centers,
        internal_cost_centers,
        repeated_setup_candidates,
        init_only_summary: InitOnlySummary {
            total_seconds: init_profile.total_seconds,
            fft_plan_seconds,
            filter_build_seconds,
            sparsify_seconds,
            fft_basis_bytes,
        },
    })
}

fn print_aggregate_summary(summary: &AggregateSummary) {
    println!();
    println!("Aggregate summary:");
    println!(
        "  Inputs profiled: {}  mean frontend total: {}",
        summary.input_count,
        fmt_seconds(summary.frontend_mean_seconds)
    );
    if let Some(harmonic) = summary.slowest_harmonic {
        println!(
            "  Slowest harmonic on average: {} ({})",
            harmonic,
            fmt_seconds(summary.slowest_harmonic_mean_seconds)
        );
    }
    if let Some(octave) = summary.slowest_octave {
        println!(
            "  Slowest shared octave on average: {} ({})",
            octave,
            fmt_seconds(summary.slowest_octave_mean_seconds)
        );
    }
    println!(
        "  Average non-transform overhead: octave_pyramid={:.1}% hcqt_assembly={:.1}% hcqt_to_batch={:.1}%",
        summary
            .internal_cost_centers
            .iter()
            .find(|cost| cost.name == "octave_downsample")
            .map(|cost| cost.mean_fraction_of_frontend * 100.0)
            .unwrap_or(0.0),
        summary.average_hcqt_assembly_fraction * 100.0,
        summary.average_batch_conversion_fraction * 100.0
    );
    println!(
        "  Shared octave-pyramid breakdown: conv={} scale={} alloc={} kernel_prep={}",
        fmt_seconds(summary.downsample_convolution_mean_seconds),
        fmt_seconds(summary.downsample_scale_mean_seconds),
        fmt_seconds(summary.downsample_allocation_mean_seconds),
        fmt_seconds(summary.downsample_kernel_prepare_mean_seconds)
    );
    println!("  Top frontend cost centers:");
    for cost in &summary.top_cost_centers {
        println!(
            "    {}: {} ({:.1}% of frontend)",
            cost.name,
            fmt_seconds(cost.mean_seconds),
            cost.mean_fraction_of_frontend * 100.0
        );
    }
    println!("  Top internal frontend stages:");
    for cost in &summary.internal_cost_centers {
        println!(
            "    {}: {} ({:.1}% of frontend)",
            cost.name,
            fmt_seconds(cost.mean_seconds),
            cost.mean_fraction_of_frontend * 100.0
        );
    }
    println!("  Repeated setup candidates:");
    for cost in &summary.repeated_setup_candidates {
        println!(
            "    {}: {} ({:.1}% of frontend)",
            cost.name,
            fmt_seconds(cost.mean_seconds),
            cost.mean_fraction_of_frontend * 100.0
        );
    }
    println!(
        "  Init-only work summary: total={} fft_plan={} filter_build={} sparsify={} basis={}",
        fmt_seconds(summary.init_only_summary.total_seconds),
        fmt_seconds(summary.init_only_summary.fft_plan_seconds),
        fmt_seconds(summary.init_only_summary.filter_build_seconds),
        fmt_seconds(summary.init_only_summary.sparsify_seconds),
        fmt_bytes(summary.init_only_summary.fft_basis_bytes as f64)
    );
}

fn print_scalar(label: &str, stats: &ScalarProfileSummary) {
    println!(
        "{label}: first={} mean={} min={} max={} p95={} runs={}",
        fmt_seconds(stats.first_seconds),
        fmt_seconds(stats.mean_seconds),
        fmt_seconds(stats.min_seconds),
        fmt_seconds(stats.max_seconds),
        stats
            .p95_seconds
            .map(fmt_seconds)
            .unwrap_or_else(|| "n/a".to_owned()),
        stats.iterations
    );
}

fn mean_over_harmonics(
    summaries: &[FrontendProfileBenchmarkSummary],
    selector: impl Fn(&FrontendHarmonicProfileSummary) -> f64,
) -> f64 {
    let mut values = Vec::new();
    for summary in summaries {
        for harmonic in &summary.harmonics {
            values.push(selector(harmonic));
        }
    }

    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn mean_stage_total(
    summaries: &[FrontendProfileBenchmarkSummary],
    octave_selector: impl Fn(
        &FrontendHarmonicProfileSummary,
        &fretnet_runtime::FrontendOctaveProfileSummary,
    ) -> f64,
    harmonic_selector: impl Fn(&FrontendHarmonicProfileSummary) -> f64,
) -> f64 {
    if summaries.is_empty() {
        return 0.0;
    }

    let mut totals = Vec::with_capacity(summaries.len());
    for summary in summaries {
        let mut total = 0.0f64;
        for harmonic in &summary.harmonics {
            total += harmonic_selector(harmonic);
            for octave in &harmonic.octaves {
                total += octave_selector(harmonic, octave);
            }
        }
        totals.push(total);
    }

    totals.iter().sum::<f64>() / totals.len() as f64
}

fn mean_over_octaves(
    summaries: &[FrontendProfileBenchmarkSummary],
    selector: impl Fn(&fretnet_runtime::FrontendOctaveProfileSummary) -> f64,
) -> f64 {
    let mut values = Vec::new();
    for summary in summaries {
        for harmonic in &summary.harmonics {
            for octave in &harmonic.octaves {
                values.push(selector(octave));
            }
        }
    }

    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn total_over_pyramid_levels(
    summaries: &[FrontendProfileBenchmarkSummary],
    selector: impl Fn(&fretnet_runtime::FrontendOctavePyramidSummary) -> f64,
) -> f64 {
    let mut totals = Vec::new();
    for summary in summaries {
        let mut total = 0.0f64;
        for octave in &summary.octave_pyramid_levels {
            total += selector(octave);
        }
        totals.push(total);
    }

    if totals.is_empty() {
        0.0
    } else {
        totals.iter().sum::<f64>() / totals.len() as f64
    }
}

fn safe_fraction(numerator: f64, denominator: f64) -> f64 {
    if denominator <= f64::EPSILON {
        0.0
    } else {
        numerator / denominator
    }
}

fn fmt_seconds(seconds: f64) -> String {
    if seconds >= 1.0 {
        format!("{seconds:.3}s")
    } else if seconds >= 0.001 {
        format!("{:.3}ms", seconds * 1_000.0)
    } else {
        format!("{:.3}us", seconds * 1_000_000.0)
    }
}

fn fmt_bytes(bytes: f64) -> String {
    if bytes >= 1024.0 * 1024.0 {
        format!("{:.2} MiB", bytes / (1024.0 * 1024.0))
    } else if bytes >= 1024.0 {
        format!("{:.2} KiB", bytes / 1024.0)
    } else {
        format!("{bytes:.0} B")
    }
}

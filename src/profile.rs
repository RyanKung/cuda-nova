//! Structured, reusable-parameter weighted-forward profiling boundary.

use std::{
    fs::OpenOptions,
    io::{self, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use cuda_nova::{CudaNovaEngine, CudaNovaError, GpuBackendStats, GpuMsmStats};
use halo2curves::bn256::Fr;
use halo2curves::ff::Field;
use serde::Serialize;
use thiserror::Error;
use topology_nova::{CsrTopology, WeightedForwardProver, WeightedForwardWitness};

use crate::cli::ForwardProfileRequest;

/// Stable schema identifier for machine-readable profile receipts.
const PROFILE_RECEIPT_SCHEMA: &str = "cuda-nova-weighted-forward-profile-v3";

/// Complete machine-readable record for one reusable-parameter profile run.
#[derive(Debug, Serialize)]
pub(crate) struct ForwardProfileReceipt {
    /// Stable schema identifier for downstream parsers.
    schema: &'static str,
    /// Wall-clock start time as seconds since the Unix epoch.
    generated_at_unix_seconds: u64,
    /// `cuda-nova` package version embedded at compile time.
    package_version: &'static str,
    /// Optional operator-supplied source revision.
    source_revision: Option<String>,
    /// `standard`, `smoke`, or `custom` plan classification.
    mode: &'static str,
    /// Positive recursive lengths measured in ascending order.
    step_plan: Vec<usize>,
    /// Primary Nova polynomial-commitment backend selected at compile time.
    commitment_backend: &'static str,
    /// Whether the Linux CUDA arithmetic backend is compiled into this runner.
    cuda_backend_enabled: bool,
    /// Whether the optional official Blitzar MSM provider is active.
    gpu_msm_enabled: bool,
    /// CUDA device ordinal selected by the runner.
    device_ordinal: usize,
    /// PTX module loaded by the sidecar.
    ptx_module: String,
    /// Trusted setup directory supplied to a `HyperKZG` run, when applicable.
    ptau_directory: Option<String>,
    /// One-time reusable public-parameter setup measurement.
    setup: SetupMeasurement,
    /// Proof and verification measurements for every requested recursive length.
    measurements: Vec<ForwardMeasurement>,
    /// Separate two-chunk commitment regression measurement.
    chunked_commitment: ChunkedMeasurement,
    /// Accelerator telemetry attributable to the complete profile run.
    accelerators: AcceleratorMeasurement,
}

/// One-time public-parameter setup metrics shared by the recursive-length curve.
#[derive(Debug, Serialize)]
struct SetupMeasurement {
    /// Setup wall time in microseconds.
    microseconds: u128,
    /// Primary constraint count for the fixed three-neuron circuit shape.
    primary_constraints: usize,
    /// Primary variable count for the fixed three-neuron circuit shape.
    primary_variables: usize,
    /// Accelerator work attributable to reusable parameter setup.
    accelerators: AcceleratorMeasurement,
}

/// Proof and verification metrics for one requested recursive length.
#[derive(Debug, Serialize)]
struct ForwardMeasurement {
    /// Requested number of recursive weighted-forward steps.
    requested_steps: usize,
    /// Number of steps reported by the verified Nova proof.
    proof_steps: usize,
    /// Proof wall time in microseconds.
    prove_microseconds: u128,
    /// Base-case circuit construction and Nova initialization time.
    prover_initialization_microseconds: u128,
    /// Time spent pushing all witnesses through recursive folding.
    recursive_folding_microseconds: u128,
    /// Time spent converting the streaming prover into its proof wrapper.
    finalization_microseconds: u128,
    /// Integer proof wall time divided by the requested step count.
    prove_per_step_microseconds: u128,
    /// Verification wall time in microseconds.
    verify_microseconds: u128,
    /// Whether the proof bound the expected topology, initial input, and output.
    verified: bool,
    /// Accelerator work attributable only to proof construction.
    prove_accelerators: AcceleratorMeasurement,
    /// Accelerator work attributable only to verification.
    verify_accelerators: AcceleratorMeasurement,
}

/// Metrics for the separate twelve-neuron, two-Poseidon-chunk regression.
#[derive(Debug, Serialize)]
struct ChunkedMeasurement {
    /// Fixed neuron count used by this regression.
    neurons: usize,
    /// Number of Poseidon-rate chunks exercised by each vector commitment.
    commitment_chunks: usize,
    /// Reusable parameter setup wall time in microseconds.
    setup_microseconds: u128,
    /// Proof wall time in microseconds.
    prove_microseconds: u128,
    /// Verification wall time in microseconds.
    verify_microseconds: u128,
    /// Primary constraint count for the twelve-neuron circuit shape.
    primary_constraints: usize,
    /// Primary variable count for the twelve-neuron circuit shape.
    primary_variables: usize,
    /// Whether the proof bound the expected public commitments.
    verified: bool,
    /// Accelerator work attributable to this regression's setup.
    setup_accelerators: AcceleratorMeasurement,
    /// Accelerator work attributable to this regression's proof construction.
    prove_accelerators: AcceleratorMeasurement,
    /// Accelerator work attributable to this regression's verification.
    verify_accelerators: AcceleratorMeasurement,
}

/// One pair of cumulative accelerator snapshots.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct AcceleratorSnapshot {
    /// CUDA arithmetic backend snapshot.
    cuda: GpuBackendStats,
    /// Optional Blitzar MSM provider snapshot.
    msm: GpuMsmStats,
}

/// Serializable accelerator work isolated to one measured phase.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
struct AcceleratorMeasurement {
    /// CUDA arithmetic backend work.
    cuda: CudaArithmeticMeasurement,
    /// Optional Blitzar MSM provider work.
    msm: GpuMsmMeasurement,
}

/// Serializable CUDA arithmetic work and resource-use deltas.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
struct CudaArithmeticMeasurement {
    /// Logical R1CS sparse matrix-vector products.
    spmv_calls: u64,
    /// Physical CUDA launches used for the logical sparse products.
    spmv_batches: u64,
    /// Relaxed-witness vector-fold launches.
    fold_calls: u64,
    /// NIFS cross-term launches.
    cross_term_calls: u64,
    /// End-to-end CSR backend time in microseconds.
    spmv_microseconds: u64,
    /// End-to-end vector-fold backend time in microseconds.
    fold_microseconds: u64,
    /// End-to-end cross-term backend time in microseconds.
    cross_term_microseconds: u64,
    /// Exact static CSR cache hits.
    csr_cache_hits: u64,
    /// Exact static CSR cache misses.
    csr_cache_misses: u64,
    /// Reusable arithmetic-output cache hits.
    output_cache_hits: u64,
    /// Reusable arithmetic-output cache misses.
    output_cache_misses: u64,
    /// Device buffers allocated by arithmetic requests.
    device_buffer_allocations: u64,
    /// Stream synchronizations issued by arithmetic requests.
    stream_synchronizations: u64,
}

/// Serializable Blitzar MSM provider deltas.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
struct GpuMsmMeasurement {
    /// Number of single or batched provider calls.
    calls: u64,
    /// Number of commitment outputs requested.
    batches: u64,
    /// Number of scalar-base pairs processed.
    scalars: u64,
    /// End-to-end provider time in microseconds.
    microseconds: u64,
}

/// Failures at the receipt timestamp, serialization, or filesystem boundary.
#[derive(Debug, Error)]
pub(crate) enum ProfileOutputError {
    /// The host clock was earlier than the Unix epoch.
    #[error("the host clock is earlier than the Unix epoch")]
    ClockBeforeUnixEpoch,
    /// The receipt could not be encoded as JSON.
    #[error("the weighted-forward profile receipt could not be serialized: {source}")]
    Serialization {
        /// JSON encoder failure.
        source: serde_json::Error,
    },
    /// The requested receipt path already exists and must not be overwritten.
    #[error("the profile receipt destination already exists: {path}")]
    ReceiptAlreadyExists {
        /// Existing path protected from overwrite.
        path: PathBuf,
    },
    /// A named receipt or stdout operation failed.
    #[error("cannot {operation} profile receipt {path}: {source}")]
    Io {
        /// Filesystem or stream operation being attempted.
        operation: &'static str,
        /// Receipt path, or `<stdout>` for standard output.
        path: PathBuf,
        /// Underlying IO failure.
        source: io::Error,
    },
}

/// Returns the current Unix timestamp used to anchor one receipt.
pub(crate) fn current_unix_seconds() -> Result<u64, ProfileOutputError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| ProfileOutputError::ClockBeforeUnixEpoch)
        .map(|duration| duration.as_secs())
}

/// Rejects an existing receipt before an expensive profile starts.
pub(crate) fn validate_receipt_destination(path: &Path) -> Result<(), ProfileOutputError> {
    match path.try_exists() {
        Ok(false) => Ok(()),
        Ok(true) => Err(ProfileOutputError::ReceiptAlreadyExists {
            path: path.to_path_buf(),
        }),
        Err(source) => Err(ProfileOutputError::Io {
            operation: "inspect",
            path: path.to_path_buf(),
            source,
        }),
    }
}

/// Executes one setup, all requested recursive lengths, and the chunked case.
///
/// # Errors
///
/// Returns a typed sidecar or Nova error when fixture construction, setup,
/// proving, verification, or CUDA accounting violates its invariant.
pub(crate) fn run_forward_profile(
    engine: &CudaNovaEngine,
    ptau_dir: Option<&Path>,
    request: &ForwardProfileRequest,
    generated_at_unix_seconds: u64,
    device_ordinal: usize,
    ptx_module: &Path,
) -> Result<ForwardProfileReceipt, CudaNovaError> {
    #[cfg(not(feature = "hyperkzg"))]
    let _ = ptau_dir;
    let profile_stats_before = accelerator_snapshot(engine);
    let (topology, witnesses) = forward_fixture(request.maximum_step_count().get())?;
    let first = witnesses.first().ok_or(CudaNovaError::InvalidTranscript {
        proposition: "the forward profile fixture has a first witness",
    })?;
    let setup_stats_before = accelerator_snapshot(engine);
    let setup_start = Instant::now();
    #[cfg(feature = "hyperkzg")]
    let parameters = engine.setup_weighted_forward_with_ptau_dir(
        topology.clone(),
        first,
        required_ptau_dir(ptau_dir)?,
    )?;
    #[cfg(not(feature = "hyperkzg"))]
    let parameters = engine.setup_weighted_forward(topology.clone(), first)?;
    let setup = SetupMeasurement {
        microseconds: setup_start.elapsed().as_micros(),
        primary_constraints: parameters.primary_constraints(),
        primary_variables: parameters.primary_variables(),
        accelerators: accelerator_delta(setup_stats_before, accelerator_snapshot(engine))?,
    };
    let mut measurements = Vec::with_capacity(request.step_counts().len());
    for requested_steps in request.step_counts() {
        measurements.push(measure_forward_length(
            engine,
            &parameters,
            &topology,
            &witnesses,
            first,
            *requested_steps,
        )?);
    }
    let chunked_commitment = measure_chunked_forward(engine, ptau_dir)?;
    let profile_stats_after = accelerator_snapshot(engine);
    let accelerators = accelerator_delta(profile_stats_before, profile_stats_after)?;
    Ok(ForwardProfileReceipt {
        schema: PROFILE_RECEIPT_SCHEMA,
        generated_at_unix_seconds,
        package_version: env!("CARGO_PKG_VERSION"),
        source_revision: request.source_revision().map(str::to_owned),
        mode: request.mode_label(),
        step_plan: request
            .step_counts()
            .iter()
            .map(|value| value.get())
            .collect(),
        commitment_backend: commitment_backend_label(),
        cuda_backend_enabled: cuda_nova::cuda_backend_enabled(),
        gpu_msm_enabled: cuda_nova::gpu_msm_enabled(),
        device_ordinal,
        ptx_module: ptx_module.display().to_string(),
        ptau_directory: receipt_ptau_directory(ptau_dir),
        setup,
        measurements,
        chunked_commitment,
        accelerators,
    })
}

/// Emits a receipt to stdout or creates the requested destination without overwrite.
pub(crate) fn emit_receipt(
    receipt: &ForwardProfileReceipt,
    destination: Option<&Path>,
) -> Result<(), ProfileOutputError> {
    let mut encoded = serde_json::to_vec_pretty(receipt)
        .map_err(|source| ProfileOutputError::Serialization { source })?;
    encoded.push(b'\n');
    if let Some(path) = destination {
        return write_new_receipt(path, &encoded);
    }
    let mut stdout = io::stdout().lock();
    stdout
        .write_all(&encoded)
        .map_err(|source| ProfileOutputError::Io {
            operation: "write",
            path: PathBuf::from("<stdout>"),
            source,
        })?;
    stdout.flush().map_err(|source| ProfileOutputError::Io {
        operation: "flush",
        path: PathBuf::from("<stdout>"),
        source,
    })
}

/// Measures one recursive prefix and proves the requested/provided lengths agree.
fn measure_forward_length(
    engine: &CudaNovaEngine,
    parameters: &topology_nova::WeightedForwardParameters,
    topology: &CsrTopology,
    witnesses: &[WeightedForwardWitness],
    first: &WeightedForwardWitness,
    requested_steps: std::num::NonZeroUsize,
) -> Result<ForwardMeasurement, CudaNovaError> {
    let selected =
        witnesses
            .get(..requested_steps.get())
            .ok_or(CudaNovaError::InvalidTranscript {
                proposition: "the forward profile fixture has every requested witness",
            })?;
    let final_witness = selected.last().ok_or(CudaNovaError::InvalidTranscript {
        proposition: "the selected forward profile is non-empty",
    })?;
    let prove_stats_before = accelerator_snapshot(engine);
    let prove_start = Instant::now();
    let initialization_start = Instant::now();
    let mut prover = WeightedForwardProver::new_with_parameters(parameters, first)?;
    let prover_initialization_microseconds = initialization_start.elapsed().as_micros();
    let folding_start = Instant::now();
    for witness in selected {
        prover.push_step(witness)?;
    }
    let recursive_folding_microseconds = folding_start.elapsed().as_micros();
    let finalization_start = Instant::now();
    let proof = prover.finish()?;
    let finalization_microseconds = finalization_start.elapsed().as_micros();
    let prove_microseconds = prove_start.elapsed().as_micros();
    let prove_stats_after = accelerator_snapshot(engine);
    if proof.steps() != requested_steps.get() {
        return Err(CudaNovaError::InvalidTranscript {
            proposition: "the Nova proof step count equals the requested profile length",
        });
    }
    let verify_stats_before = prove_stats_after;
    let verify_start = Instant::now();
    let verified = proof.verify_against(
        topology.root(),
        first.input_commitment(),
        final_witness.output_commitment(),
    )?;
    let verify_microseconds = verify_start.elapsed().as_micros();
    if !verified {
        return Err(CudaNovaError::Nova(
            topology_nova::TopologyNovaError::InvalidFinalState,
        ));
    }
    let verify_stats_after = accelerator_snapshot(engine);
    let divisor =
        u64::try_from(requested_steps.get()).map_err(|_| CudaNovaError::SizeOverflow {
            target: "forward profile step count",
        })?;
    Ok(ForwardMeasurement {
        requested_steps: requested_steps.get(),
        proof_steps: proof.steps(),
        prove_microseconds,
        prover_initialization_microseconds,
        recursive_folding_microseconds,
        finalization_microseconds,
        prove_per_step_microseconds: prove_microseconds / u128::from(divisor),
        verify_microseconds,
        verified,
        prove_accelerators: accelerator_delta(prove_stats_before, prove_stats_after)?,
        verify_accelerators: accelerator_delta(verify_stats_before, verify_stats_after)?,
    })
}

/// Measures one twelve-neuron identity pass spanning two Poseidon chunks.
fn measure_chunked_forward(
    engine: &CudaNovaEngine,
    ptau_dir: Option<&Path>,
) -> Result<ChunkedMeasurement, CudaNovaError> {
    #[cfg(not(feature = "hyperkzg"))]
    let _ = ptau_dir;
    let row_offsets = (0_u32..=12).collect::<Vec<_>>();
    let column_indices = (0_u32..12).collect::<Vec<_>>();
    let topology = Arc::new(CsrTopology::new(12, &row_offsets, &column_indices)?);
    let input = (1_u64..=12).map(Fr::from).collect::<Vec<_>>();
    let weights = vec![Fr::ONE; 12];
    let first = WeightedForwardWitness::new(&topology, &input, &weights)?;
    let setup_stats_before = accelerator_snapshot(engine);
    let setup_start = Instant::now();
    #[cfg(feature = "hyperkzg")]
    let parameters = engine.setup_weighted_forward_with_ptau_dir(
        topology.clone(),
        &first,
        required_ptau_dir(ptau_dir)?,
    )?;
    #[cfg(not(feature = "hyperkzg"))]
    let parameters = engine.setup_weighted_forward(topology.clone(), &first)?;
    let setup_microseconds = setup_start.elapsed().as_micros();
    let setup_stats_after = accelerator_snapshot(engine);
    let prove_stats_before = setup_stats_after;
    let prove_start = Instant::now();
    let proof =
        engine.prove_weighted_forward_with_parameters(&parameters, std::slice::from_ref(&first))?;
    let prove_microseconds = prove_start.elapsed().as_micros();
    let prove_stats_after = accelerator_snapshot(engine);
    let verify_stats_before = prove_stats_after;
    let verify_start = Instant::now();
    let verified = proof.verify_against(
        topology.root(),
        first.input_commitment(),
        first.output_commitment(),
    )?;
    let verify_microseconds = verify_start.elapsed().as_micros();
    if !verified {
        return Err(CudaNovaError::Nova(
            topology_nova::TopologyNovaError::InvalidFinalState,
        ));
    }
    let verify_stats_after = accelerator_snapshot(engine);
    Ok(ChunkedMeasurement {
        neurons: 12,
        commitment_chunks: 2,
        setup_microseconds,
        prove_microseconds,
        verify_microseconds,
        primary_constraints: parameters.primary_constraints(),
        primary_variables: parameters.primary_variables(),
        verified,
        setup_accelerators: accelerator_delta(setup_stats_before, setup_stats_after)?,
        prove_accelerators: accelerator_delta(prove_stats_before, prove_stats_after)?,
        verify_accelerators: accelerator_delta(verify_stats_before, verify_stats_after)?,
    })
}

/// Builds a deterministic chain of private forward witnesses for profiling.
fn forward_fixture(
    steps: usize,
) -> Result<(Arc<CsrTopology>, Vec<WeightedForwardWitness>), CudaNovaError> {
    let topology = Arc::new(CsrTopology::new(3, &[0, 2, 3, 4], &[0, 2, 1, 0])?);
    let weights = vec![Fr::from(2_u64), -Fr::ONE, Fr::from(3_u64), Fr::from(4_u64)];
    let mut input = vec![Fr::ONE, Fr::from(2_u64), Fr::from(5_u64)];
    let mut witnesses = Vec::with_capacity(steps);
    for _ in 0..steps {
        let witness = WeightedForwardWitness::new(&topology, &input, &weights)?;
        input = fixed_forward_output(&input);
        witnesses.push(witness);
    }
    Ok((topology, witnesses))
}

/// Evaluates the fixed three-neuron profile topology for the next input.
fn fixed_forward_output(input: &[Fr]) -> Vec<Fr> {
    let x0 = input.first().copied().unwrap_or(Fr::ZERO);
    let x1 = input.get(1).copied().unwrap_or(Fr::ZERO);
    let x2 = input.get(2).copied().unwrap_or(Fr::ZERO);
    vec![
        Fr::from(2_u64) * x0 - x2,
        Fr::from(3_u64) * x1,
        Fr::from(4_u64) * x0,
    ]
}

/// Captures both accelerator counters at one phase boundary.
fn accelerator_snapshot(engine: &CudaNovaEngine) -> AcceleratorSnapshot {
    AcceleratorSnapshot {
        cuda: engine.gpu_backend_stats(),
        msm: engine.gpu_msm_stats(),
    }
}

/// Computes non-decreasing CUDA and MSM deltas for one measured phase.
fn accelerator_delta(
    before: AcceleratorSnapshot,
    after: AcceleratorSnapshot,
) -> Result<AcceleratorMeasurement, CudaNovaError> {
    Ok(AcceleratorMeasurement {
        cuda: cuda_arithmetic_delta(before.cuda, after.cuda)?,
        msm: gpu_msm_delta(before.msm, after.msm)?,
    })
}

/// Computes every CUDA arithmetic counter delta without accepting wraparound.
fn cuda_arithmetic_delta(
    before: GpuBackendStats,
    after: GpuBackendStats,
) -> Result<CudaArithmeticMeasurement, CudaNovaError> {
    Ok(CudaArithmeticMeasurement {
        spmv_calls: monotonic_delta(
            before.spmv_calls,
            after.spmv_calls,
            "the CUDA SpMV call counter is monotonic",
        )?,
        spmv_batches: monotonic_delta(
            before.spmv_batches,
            after.spmv_batches,
            "the CUDA SpMV batch counter is monotonic",
        )?,
        fold_calls: monotonic_delta(
            before.fold_calls,
            after.fold_calls,
            "the CUDA fold call counter is monotonic",
        )?,
        cross_term_calls: monotonic_delta(
            before.cross_term_calls,
            after.cross_term_calls,
            "the CUDA cross-term call counter is monotonic",
        )?,
        spmv_microseconds: monotonic_delta(
            before.spmv_microseconds,
            after.spmv_microseconds,
            "the CUDA SpMV time counter is monotonic",
        )?,
        fold_microseconds: monotonic_delta(
            before.fold_microseconds,
            after.fold_microseconds,
            "the CUDA fold time counter is monotonic",
        )?,
        cross_term_microseconds: monotonic_delta(
            before.cross_term_microseconds,
            after.cross_term_microseconds,
            "the CUDA cross-term time counter is monotonic",
        )?,
        csr_cache_hits: monotonic_delta(
            before.csr_cache_hits,
            after.csr_cache_hits,
            "the CUDA CSR cache-hit counter is monotonic",
        )?,
        csr_cache_misses: monotonic_delta(
            before.csr_cache_misses,
            after.csr_cache_misses,
            "the CUDA CSR cache-miss counter is monotonic",
        )?,
        output_cache_hits: monotonic_delta(
            before.output_cache_hits,
            after.output_cache_hits,
            "the CUDA output cache-hit counter is monotonic",
        )?,
        output_cache_misses: monotonic_delta(
            before.output_cache_misses,
            after.output_cache_misses,
            "the CUDA output cache-miss counter is monotonic",
        )?,
        device_buffer_allocations: monotonic_delta(
            before.device_buffer_allocations,
            after.device_buffer_allocations,
            "the CUDA device-buffer allocation counter is monotonic",
        )?,
        stream_synchronizations: monotonic_delta(
            before.stream_synchronizations,
            after.stream_synchronizations,
            "the CUDA stream-synchronization counter is monotonic",
        )?,
    })
}

/// Computes every optional Blitzar provider counter delta.
fn gpu_msm_delta(
    before: GpuMsmStats,
    after: GpuMsmStats,
) -> Result<GpuMsmMeasurement, CudaNovaError> {
    Ok(GpuMsmMeasurement {
        calls: monotonic_delta(
            before.calls,
            after.calls,
            "the GPU MSM call counter is monotonic",
        )?,
        batches: monotonic_delta(
            before.batches,
            after.batches,
            "the GPU MSM batch counter is monotonic",
        )?,
        scalars: monotonic_delta(
            before.scalars,
            after.scalars,
            "the GPU MSM scalar counter is monotonic",
        )?,
        microseconds: monotonic_delta(
            before.microseconds,
            after.microseconds,
            "the GPU MSM time counter is monotonic",
        )?,
    })
}

/// Subtracts one cumulative counter while rejecting non-monotonic snapshots.
fn monotonic_delta(
    before: u64,
    after: u64,
    proposition: &'static str,
) -> Result<u64, CudaNovaError> {
    after
        .checked_sub(before)
        .ok_or(CudaNovaError::InvalidTranscript { proposition })
}

/// Returns the primary commitment backend selected for this binary.
const fn commitment_backend_label() -> &'static str {
    if cfg!(feature = "hyperkzg") {
        "hyperkzg"
    } else {
        "pedersen-ipa"
    }
}

/// Records a setup directory only when the selected backend consumes it.
fn receipt_ptau_directory(ptau_dir: Option<&Path>) -> Option<String> {
    if cfg!(feature = "hyperkzg") {
        ptau_dir.map(|path| path.display().to_string())
    } else {
        None
    }
}

/// Requires a trusted Powers-of-Tau directory for a `HyperKZG` profile.
#[cfg(feature = "hyperkzg")]
fn required_ptau_dir(ptau_dir: Option<&Path>) -> Result<&Path, CudaNovaError> {
    ptau_dir.ok_or(CudaNovaError::Nova(
        topology_nova::TopologyNovaError::HyperKzgSetupRequired,
    ))
}

/// Creates and durably writes one receipt without replacing an existing file.
fn write_new_receipt(path: &Path, encoded: &[u8]) -> Result<(), ProfileOutputError> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|source| receipt_open_error(path, source))?;
    file.write_all(encoded)
        .map_err(|source| ProfileOutputError::Io {
            operation: "write",
            path: path.to_path_buf(),
            source,
        })?;
    file.sync_all().map_err(|source| ProfileOutputError::Io {
        operation: "sync",
        path: path.to_path_buf(),
        source,
    })
}

/// Classifies create-new collisions separately from other open failures.
fn receipt_open_error(path: &Path, source: io::Error) -> ProfileOutputError {
    if source.kind() == io::ErrorKind::AlreadyExists {
        ProfileOutputError::ReceiptAlreadyExists {
            path: path.to_path_buf(),
        }
    } else {
        ProfileOutputError::Io {
            operation: "create",
            path: path.to_path_buf(),
            source,
        }
    }
}

#[cfg(test)]
mod tests {
    use cuda_nova::{GpuBackendStats, GpuMsmStats};

    use super::{
        AcceleratorMeasurement, AcceleratorSnapshot, ChunkedMeasurement, CudaArithmeticMeasurement,
        ForwardMeasurement, ForwardProfileReceipt, GpuMsmMeasurement, PROFILE_RECEIPT_SCHEMA,
        ProfileOutputError, SetupMeasurement, accelerator_delta, write_new_receipt,
    };

    /// Builds a deterministic receipt used only to validate the JSON schema.
    fn sample_receipt() -> ForwardProfileReceipt {
        let no_accelerators = AcceleratorMeasurement::default();
        ForwardProfileReceipt {
            schema: PROFILE_RECEIPT_SCHEMA,
            generated_at_unix_seconds: 1,
            package_version: "0.1.0",
            source_revision: Some("65738a2".to_owned()),
            mode: "smoke",
            step_plan: vec![1, 2],
            commitment_backend: "pedersen-ipa",
            cuda_backend_enabled: true,
            gpu_msm_enabled: false,
            device_ordinal: 0,
            ptx_module: "cuda_nova.ptx".to_owned(),
            ptau_directory: None,
            setup: SetupMeasurement {
                microseconds: 10,
                primary_constraints: 20,
                primary_variables: 30,
                accelerators: no_accelerators,
            },
            measurements: vec![ForwardMeasurement {
                requested_steps: 1,
                proof_steps: 1,
                prove_microseconds: 40,
                prover_initialization_microseconds: 10,
                recursive_folding_microseconds: 20,
                finalization_microseconds: 1,
                prove_per_step_microseconds: 40,
                verify_microseconds: 50,
                verified: true,
                prove_accelerators: no_accelerators,
                verify_accelerators: no_accelerators,
            }],
            chunked_commitment: ChunkedMeasurement {
                neurons: 12,
                commitment_chunks: 2,
                setup_microseconds: 60,
                prove_microseconds: 70,
                verify_microseconds: 80,
                primary_constraints: 90,
                primary_variables: 100,
                verified: true,
                setup_accelerators: no_accelerators,
                prove_accelerators: no_accelerators,
                verify_accelerators: no_accelerators,
            },
            accelerators: no_accelerators,
        }
    }

    #[test]
    fn accelerator_delta_preserves_cuda_and_msm_counters() {
        let before = AcceleratorSnapshot {
            cuda: GpuBackendStats {
                spmv_calls: 2,
                spmv_batches: 1,
                fold_calls: 3,
                cross_term_calls: 5,
                device_buffer_allocations: 7,
                ..GpuBackendStats::default()
            },
            msm: GpuMsmStats {
                calls: 11,
                batches: 13,
                scalars: 17,
                microseconds: 19,
            },
        };
        let after = AcceleratorSnapshot {
            cuda: GpuBackendStats {
                spmv_calls: 7,
                spmv_batches: 3,
                fold_calls: 11,
                cross_term_calls: 18,
                device_buffer_allocations: 30,
                ..GpuBackendStats::default()
            },
            msm: GpuMsmStats {
                calls: 14,
                batches: 18,
                scalars: 24,
                microseconds: 30,
            },
        };
        assert_eq!(
            accelerator_delta(before, after).ok(),
            Some(AcceleratorMeasurement {
                cuda: CudaArithmeticMeasurement {
                    spmv_calls: 5,
                    spmv_batches: 2,
                    fold_calls: 8,
                    cross_term_calls: 13,
                    device_buffer_allocations: 23,
                    ..CudaArithmeticMeasurement::default()
                },
                msm: GpuMsmMeasurement {
                    calls: 3,
                    batches: 5,
                    scalars: 7,
                    microseconds: 11,
                },
            })
        );
    }

    #[test]
    fn receipt_writer_never_overwrites_an_existing_result() {
        let directory = tempfile::tempdir();
        assert!(directory.is_ok());
        if let Ok(directory) = directory {
            let receipt = directory.path().join("profile.json");
            assert!(write_new_receipt(&receipt, b"first\n").is_ok());
            assert!(matches!(
                write_new_receipt(&receipt, b"second\n"),
                Err(ProfileOutputError::ReceiptAlreadyExists { .. })
            ));
            assert_eq!(std::fs::read(receipt).ok(), Some(b"first\n".to_vec()));
        }
    }

    #[test]
    fn receipt_json_exposes_the_versioned_schema_and_step_plan() {
        let encoded = serde_json::to_value(sample_receipt());
        assert!(encoded.is_ok());
        if let Ok(encoded) = encoded {
            let expected_step_plan = vec![serde_json::Value::from(1), serde_json::Value::from(2)];
            assert_eq!(
                encoded.get("schema").and_then(serde_json::Value::as_str),
                Some(PROFILE_RECEIPT_SCHEMA)
            );
            assert_eq!(
                encoded
                    .get("step_plan")
                    .and_then(serde_json::Value::as_array),
                Some(&expected_step_plan)
            );
        }
    }
}

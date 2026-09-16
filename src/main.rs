//! V100 smoke runner for the `cuda-nova` sidecar.

#![deny(missing_docs)]

use std::{path::Path, sync::Arc, time::Instant};

use cuda_nova::{CudaNovaEngine, CudaNovaError};
use halo2curves::bn256::Fr;
use halo2curves::ff::Field;
use thiserror::Error;
use topology_commitment::{Commitment, TopologyFoldStep, commit_topology_with_trace};
use topology_nova::{CsrTopology, WeightedForwardWitness};

mod cli;
mod profile;

/// CUDA device selected by the reproducible smoke and profile runner.
const DEVICE_ORDINAL: usize = 0;

/// cuda-oxide PTX module generated for the target GPU architecture.
const PTX_MODULE: &str = "cuda_nova.ptx";

/// Top-level failures from argument parsing, sidecar execution, or receipt IO.
#[derive(Debug, Error)]
enum RunnerError {
    /// The command line did not describe a valid execution plan.
    #[error(transparent)]
    Cli(#[from] cli::CliError),
    /// The CUDA sidecar or official Nova backend rejected an operation.
    #[error(transparent)]
    Sidecar(#[from] CudaNovaError),
    /// The canonical topology commitment could not be generated.
    #[error(transparent)]
    Commitment(#[from] topology_commitment::CommitmentError),
    /// The structured profile receipt could not be emitted safely.
    #[error(transparent)]
    ProfileOutput(#[from] profile::ProfileOutputError),
}

/// Generates a two-step canonical transcript and validates it on CUDA.
fn main() -> Result<(), RunnerError> {
    let raw_arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let options = cli::parse_arguments(&raw_arguments)?;
    if let Some(request) = options.profile()
        && let Some(destination) = request.receipt_path()
    {
        profile::validate_receipt_destination(destination)?;
    }
    let row_offsets: Vec<u32> = (0_u32..=50).collect();
    let column_indices: Vec<u32> = (0_u32..50).collect();
    let mut steps = Vec::new();
    let root = commit_topology_with_trace(50, &row_offsets, &column_indices, |step| {
        steps.push(step);
    })?;
    let engine = CudaNovaEngine::new(DEVICE_ORDINAL, PTX_MODULE)?;
    let report = engine.benchmark_steps(&steps, 5)?;
    println!(
        "cuda-nova resident preflight passed: {} step(s) x {} iteration(s), {} encoded byte(s), upload={}us kernel={}us download={}us end_to_end={}us",
        report.steps,
        report.iterations,
        report.encoded_bytes,
        report.upload_microseconds,
        report.kernel_microseconds,
        report.download_microseconds,
        report.end_to_end_microseconds
    );
    let poseidon_report = engine.validate_poseidon_steps(&steps)?;
    println!(
        "cuda-nova Poseidon preflight passed: {} step(s), upload={}us kernel={}us download={}us end_to_end={}us",
        poseidon_report.steps,
        poseidon_report.upload_microseconds,
        poseidon_report.kernel_microseconds,
        poseidon_report.download_microseconds,
        poseidon_report.end_to_end_microseconds
    );
    let mut invalid_steps = steps.clone();
    if let Some(step) = invalid_steps.first_mut() {
        step.index = 1;
    }
    match engine.validate_steps(&invalid_steps) {
        Err(CudaNovaError::GpuValidationFailed { .. }) => {
            println!("cuda-nova negative preflight passed: invalid index rejected");
        }
        Ok(_) | Err(_) => {
            return Err(CudaNovaError::InvalidTranscript {
                proposition: "the CUDA kernel rejects a non-canonical transition index",
            }
            .into());
        }
    }
    let mut broken_link_steps = steps.clone();
    if let Some(step) = broken_link_steps.get_mut(1) {
        step.previous = [1_u8; 32];
    }
    match engine.validate_steps(&broken_link_steps) {
        Err(CudaNovaError::GpuValidationFailed { index: 1 }) => {
            println!("cuda-nova negative preflight passed: broken link rejected");
        }
        Ok(_) | Err(_) => {
            return Err(CudaNovaError::InvalidTranscript {
                proposition: "the CUDA kernel rejects a broken accumulator link",
            }
            .into());
        }
    }
    let mut broken_digest_steps = steps.clone();
    if let Some(step) = broken_digest_steps.first_mut() {
        step.next[0] ^= 1;
    }
    match engine.validate_poseidon_steps(&broken_digest_steps) {
        Err(CudaNovaError::GpuPoseidonMismatch { index: 0, .. }) => {
            println!("cuda-nova negative preflight passed: Poseidon digest rejected");
        }
        Ok(_) | Err(_) => {
            return Err(CudaNovaError::InvalidTranscript {
                proposition: "the CUDA Poseidon kernel rejects a forged next digest",
            }
            .into());
        }
    }
    if options.should_prove() {
        run_proof(&engine, &steps, root, options.ptau_dir())?;
        run_weighted_forward(&engine, options.ptau_dir())?;
    }
    if let Some(request) = options.profile() {
        let generated_at_unix_seconds = profile::current_unix_seconds()?;
        let receipt = profile::run_forward_profile(
            &engine,
            options.ptau_dir(),
            request,
            generated_at_unix_seconds,
            DEVICE_ORDINAL,
            Path::new(PTX_MODULE),
        )?;
        profile::emit_receipt(&receipt, request.receipt_path())?;
    }
    Ok(())
}

/// Proves and verifies the smoke transcript with official Nova.
fn run_proof(
    engine: &CudaNovaEngine,
    steps: &[TopologyFoldStep],
    claimed_root: Commitment,
    ptau_dir: Option<&Path>,
) -> Result<(), CudaNovaError> {
    #[cfg(not(feature = "hyperkzg"))]
    let _ = ptau_dir;
    let proof_start = Instant::now();
    #[cfg(feature = "hyperkzg")]
    let proof =
        engine.prove_for_root_with_ptau_dir(steps, claimed_root, required_ptau_dir(ptau_dir)?)?;
    #[cfg(not(feature = "hyperkzg"))]
    let proof = engine.prove_for_root(steps, claimed_root)?;
    let proof_microseconds = proof_start.elapsed().as_micros();
    if !proof.verify_against_root(claimed_root)? {
        return Err(CudaNovaError::Nova(
            topology_nova::TopologyNovaError::InvalidFinalState,
        ));
    }
    println!(
        "official Nova proof path passed: {} step(s), prove={}us",
        proof.steps(),
        proof_microseconds
    );
    let stats = engine.gpu_backend_stats();
    println!(
        "cuda-nova arithmetic backend passed: spmv={} fold={} cross_term={}",
        stats.spmv_calls, stats.fold_calls, stats.cross_term_calls
    );
    if cuda_nova::gpu_msm_enabled() {
        if !engine.gpu_msm_self_test() {
            return Err(CudaNovaError::InvalidTranscript {
                proposition: "the official Nova MSM provider agrees with the CPU reference",
            });
        }
        println!("official Nova GPU MSM path passed");
    } else {
        println!("official Nova CPU MSM path active");
    }
    Ok(())
}

/// Proves two private weighted forward passes over a fixed three-neuron CSR
/// topology and checks the public topology/input/output commitments.
fn run_weighted_forward(
    engine: &CudaNovaEngine,
    ptau_dir: Option<&Path>,
) -> Result<(), CudaNovaError> {
    #[cfg(not(feature = "hyperkzg"))]
    let _ = ptau_dir;
    let topology = Arc::new(CsrTopology::new(3, &[0, 2, 3, 4], &[0, 2, 1, 0])?);
    let weights = vec![Fr::from(2_u64), -Fr::ONE, Fr::from(3_u64), Fr::from(4_u64)];
    let first = WeightedForwardWitness::new(
        &topology,
        &[Fr::ONE, Fr::from(2_u64), Fr::from(5_u64)],
        &weights,
    )?;
    let second = WeightedForwardWitness::new(
        &topology,
        &[-Fr::from(3_u64), Fr::from(6_u64), Fr::from(4_u64)],
        &weights,
    )?;
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
    let proof_start = Instant::now();
    let proof = engine
        .prove_weighted_forward_with_parameters(&parameters, &[first.clone(), second.clone()])?;
    let proof_microseconds = proof_start.elapsed().as_micros();
    if !proof.verify_against(
        topology.root(),
        first.input_commitment(),
        second.output_commitment(),
    )? {
        return Err(CudaNovaError::Nova(
            topology_nova::TopologyNovaError::InvalidFinalState,
        ));
    }
    let stats = engine.gpu_backend_stats();
    println!(
        "weighted-forward Nova proof passed: {} step(s), setup={}us prove={}us constraints={} variables={} spmv={} fold={} cross_term={}",
        proof.steps(),
        setup_microseconds,
        proof_microseconds,
        proof.primary_constraints(),
        proof.primary_variables(),
        stats.spmv_calls,
        stats.fold_calls,
        stats.cross_term_calls
    );
    Ok(())
}

/// Requires a trusted Powers-of-Tau directory for a `HyperKZG` build.
#[cfg(feature = "hyperkzg")]
fn required_ptau_dir(ptau_dir: Option<&Path>) -> Result<&Path, CudaNovaError> {
    ptau_dir.ok_or(CudaNovaError::Nova(
        topology_nova::TopologyNovaError::HyperKzgSetupRequired,
    ))
}

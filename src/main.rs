//! V100 smoke runner for the `cuda-nova` sidecar.

use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use cuda_nova::{CudaNovaEngine, CudaNovaError};
use halo2curves::bn256::Fr;
use halo2curves::ff::Field;
use zkfly_commitment::{Commitment, TopologyFoldStep, commit_topology_with_trace};
use zkfly_nova::{CsrTopology, WeightedForwardWitness};

/// Generates a two-step canonical transcript and validates it on CUDA.
fn main() -> Result<(), CudaNovaError> {
    let should_prove = std::env::args()
        .skip(1)
        .any(|argument| argument == "--prove");
    let profile_forward = std::env::args()
        .skip(1)
        .any(|argument| argument == "--profile-forward");
    let ptau_dir = ptau_dir_from_args();
    let row_offsets: Vec<u32> = (0_u32..=50).collect();
    let column_indices: Vec<u32> = (0_u32..50).collect();
    let mut steps = Vec::new();
    let root = commit_topology_with_trace(50, &row_offsets, &column_indices, |step| {
        steps.push(step);
    })?;
    let engine = CudaNovaEngine::new(0, "cuda_nova.ptx")?;
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
            });
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
            });
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
            });
        }
    }
    if should_prove {
        run_proof(&engine, &steps, root, ptau_dir.as_deref())?;
        run_weighted_forward(&engine, ptau_dir.as_deref())?;
    }
    if profile_forward {
        run_forward_profile(&engine, ptau_dir.as_deref())?;
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
            zkfly_nova::TopologyNovaError::InvalidFinalState,
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
            zkfly_nova::TopologyNovaError::InvalidFinalState,
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

/// Measures reusable weighted-forward proving over several recursive lengths.
fn run_forward_profile(
    engine: &CudaNovaEngine,
    ptau_dir: Option<&Path>,
) -> Result<(), CudaNovaError> {
    #[cfg(not(feature = "hyperkzg"))]
    let _ = ptau_dir;
    let (topology, witnesses) = forward_fixture(16)?;
    let first = witnesses.first().ok_or(CudaNovaError::InvalidTranscript {
        proposition: "the forward profile fixture has a first witness",
    })?;
    let setup_start = Instant::now();
    #[cfg(feature = "hyperkzg")]
    let parameters = engine.setup_weighted_forward_with_ptau_dir(
        topology.clone(),
        first,
        required_ptau_dir(ptau_dir)?,
    )?;
    #[cfg(not(feature = "hyperkzg"))]
    let parameters = engine.setup_weighted_forward(topology.clone(), first)?;
    let setup_microseconds = setup_start.elapsed().as_micros();
    println!(
        "weighted-forward profile setup: steps={}us constraints={} variables={}",
        setup_microseconds,
        parameters.primary_constraints(),
        parameters.primary_variables()
    );
    for step_count in [1_usize, 2, 8, 16] {
        let selected = witnesses
            .get(..step_count)
            .ok_or(CudaNovaError::InvalidTranscript {
                proposition: "the forward profile fixture has enough witnesses",
            })?;
        let prove_start = Instant::now();
        let proof = engine.prove_weighted_forward_with_parameters(&parameters, selected)?;
        let prove_microseconds = prove_start.elapsed().as_micros();
        let final_witness = selected.last().ok_or(CudaNovaError::InvalidTranscript {
            proposition: "the selected forward profile is non-empty",
        })?;
        let verify_start = Instant::now();
        let verified = proof.verify_against(
            topology.root(),
            first.input_commitment(),
            final_witness.output_commitment(),
        )?;
        let verify_microseconds = verify_start.elapsed().as_micros();
        if !verified {
            return Err(CudaNovaError::Nova(
                zkfly_nova::TopologyNovaError::InvalidFinalState,
            ));
        }
        let step_count_u64 =
            u64::try_from(step_count).map_err(|_| CudaNovaError::SizeOverflow {
                target: "forward profile step count",
            })?;
        println!(
            "weighted-forward profile: steps={} prove={}us prove_per_step={}us verify={}us",
            proof.steps(),
            prove_microseconds,
            prove_microseconds / u128::from(step_count_u64),
            verify_microseconds
        );
    }
    run_chunked_forward(engine, ptau_dir)?;
    Ok(())
}

/// Proves one twelve-neuron identity pass to exercise a second Poseidon chunk.
fn run_chunked_forward(
    engine: &CudaNovaEngine,
    ptau_dir: Option<&Path>,
) -> Result<(), CudaNovaError> {
    #[cfg(not(feature = "hyperkzg"))]
    let _ = ptau_dir;
    let row_offsets = (0_u32..=12).collect::<Vec<_>>();
    let column_indices = (0_u32..12).collect::<Vec<_>>();
    let topology = Arc::new(CsrTopology::new(12, &row_offsets, &column_indices)?);
    let input = (1_u64..=12).map(Fr::from).collect::<Vec<_>>();
    let weights = vec![Fr::ONE; 12];
    let first = WeightedForwardWitness::new(&topology, &input, &weights)?;
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
    let prove_start = Instant::now();
    let proof =
        engine.prove_weighted_forward_with_parameters(&parameters, std::slice::from_ref(&first))?;
    let prove_microseconds = prove_start.elapsed().as_micros();
    let verify_start = Instant::now();
    let verified = proof.verify_against(
        topology.root(),
        first.input_commitment(),
        first.output_commitment(),
    )?;
    let verify_microseconds = verify_start.elapsed().as_micros();
    if !verified {
        return Err(CudaNovaError::Nova(
            zkfly_nova::TopologyNovaError::InvalidFinalState,
        ));
    }
    println!(
        "weighted-forward chunked profile: neurons=12 chunks=2 setup={}us prove={}us verify={}us constraints={} variables={}",
        setup_microseconds,
        prove_microseconds,
        verify_microseconds,
        parameters.primary_constraints(),
        parameters.primary_variables()
    );
    Ok(())
}

/// Extracts the optional trusted `HyperKZG` setup directory from command-line
/// arguments in either `--ptau-dir PATH` or `--ptau-dir=PATH` form.
fn ptau_dir_from_args() -> Option<PathBuf> {
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        if argument == "--ptau-dir" {
            return arguments.next().map(PathBuf::from);
        }
        if let Some(path) = argument.strip_prefix("--ptau-dir=") {
            return Some(PathBuf::from(path));
        }
    }
    None
}

/// Requires a trusted Powers-of-Tau directory for a `HyperKZG` build.
#[cfg(feature = "hyperkzg")]
fn required_ptau_dir(ptau_dir: Option<&Path>) -> Result<&Path, CudaNovaError> {
    ptau_dir.ok_or(CudaNovaError::Nova(
        zkfly_nova::TopologyNovaError::HyperKzgSetupRequired,
    ))
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

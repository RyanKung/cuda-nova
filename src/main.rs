//! V100 smoke runner for the `cuda-nova` sidecar.

use cuda_nova::{CudaNovaEngine, CudaNovaError};
use zkfly_commitment::commit_topology_with_trace;

/// Generates a two-step canonical transcript and validates it on CUDA.
fn main() -> Result<(), CudaNovaError> {
    let run_proof = std::env::args()
        .skip(1)
        .any(|argument| argument == "--prove");
    let row_offsets: Vec<u32> = (0_u32..=50).collect();
    let column_indices: Vec<u32> = (0_u32..50).collect();
    let mut steps = Vec::new();
    let _root = commit_topology_with_trace(50, &row_offsets, &column_indices, |step| {
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
    if run_proof {
        let proof = engine.prove(&steps)?;
        if !proof.verify()? {
            return Err(CudaNovaError::Nova(
                zkfly_nova::TopologyNovaError::InvalidFinalState,
            ));
        }
        println!("official Nova proof path passed: {} step(s)", proof.steps());
    }
    Ok(())
}

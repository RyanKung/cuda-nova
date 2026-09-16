//! CUDA sidecar for the official `nova-snark` topology prover.
//!
//! The crate is intentionally shaped as a future standalone repository. The
//! official Nova proof relation lives in [`zkfly_nova`], while this crate owns
//! the CUDA context, PTX module, device buffers, and arithmetic backend. Nova
//! still constructs the Bellpepper circuit and controls the recursive protocol;
//! arithmetic-heavy R1CS `SpMV`, NIFS cross-terms, and relaxed-witness folds can
//! be dispatched to CUDA. MSM remains Nova's official provider with a CPU
//! fallback. This keeps the protocol and transcript ABI unchanged.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

use std::path::Path;

use thiserror::Error;
use zkfly_commitment::{Commitment, POSEIDON_RATE, TopologyFoldStep};

#[cfg(all(feature = "cuda", target_os = "linux"))]
mod cuda_backend;
#[cfg(all(feature = "cuda", target_os = "linux"))]
mod cuda_runtime;

/// Number of bytes in one fixed-width encoded topology fold step.
pub const ENCODED_STEP_BYTES: usize = 432;

/// Number of bytes reserved for the little-endian transition index.
pub const INDEX_BYTES: usize = std::mem::size_of::<u64>();

/// A summary of a successful GPU transcript preflight.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ValidationReport {
    /// Number of fold transitions checked by the device kernel.
    pub steps: usize,
    /// Number of encoded transcript bytes resident during validation.
    pub encoded_bytes: usize,
    /// Host-to-device allocation and transfer time in microseconds.
    pub upload_microseconds: u64,
    /// Kernel execution plus stream synchronization time in microseconds.
    pub kernel_microseconds: u64,
    /// Device-to-host result transfer time in microseconds.
    pub download_microseconds: u64,
    /// End-to-end validation time, including encoding, in microseconds.
    pub end_to_end_microseconds: u64,
}

/// A summary of repeated validation with one resident device allocation.
///
/// The encoded transcript and device output buffer are uploaded/allocated once;
/// the kernel is then launched `iterations` times before one result download.
/// This separates steady-state kernel cost from per-call transfer overhead.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidentValidationReport {
    /// Number of fold transitions checked by each device launch.
    pub steps: usize,
    /// Number of repeated kernel launches using the resident transcript.
    pub iterations: usize,
    /// Number of encoded transcript bytes resident during validation.
    pub encoded_bytes: usize,
    /// Host-to-device allocation and transfer time in microseconds.
    pub upload_microseconds: u64,
    /// All repeated kernel executions plus one stream synchronization in
    /// microseconds.
    pub kernel_microseconds: u64,
    /// Device-to-host result transfer time in microseconds.
    pub download_microseconds: u64,
    /// End-to-end benchmark time, including encoding, in microseconds.
    pub end_to_end_microseconds: u64,
}

/// Counts successful Nova arithmetic operations dispatched to CUDA.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuBackendStats {
    /// CSR matrix-vector launches used by Nova's R1CS relation.
    pub spmv_calls: u64,
    /// Vector fold launches used by Nova's relaxed witness updates.
    pub fold_calls: u64,
    /// Cross-term launches used by Nova's NIFS fold.
    pub cross_term_calls: u64,
}

/// Errors returned by the CUDA sidecar engine.
#[derive(Debug, Error)]
pub enum CudaNovaError {
    /// The fold transcript is empty and therefore has no Nova base case.
    #[error("the topology trace must contain at least one step")]
    EmptyTrace,
    /// The host cannot represent a required device-side size or index.
    #[error("value does not fit the CUDA representation: {target}")]
    SizeOverflow {
        /// Name of the representation that overflowed.
        target: &'static str,
    },
    /// The transcript contains a structural predicate that the GPU relation
    /// cannot accept.
    #[error("invalid topology transcript: {proposition}")]
    InvalidTranscript {
        /// Structural proposition that failed.
        proposition: &'static str,
    },
    /// The host could not prepare the canonical Poseidon parameter table.
    #[error("Poseidon parameters could not be prepared: {message}")]
    PoseidonParameters {
        /// Parameter preparation detail.
        message: String,
    },
    /// A device lane rejected one encoded transition.
    #[error("CUDA transcript preflight rejected transition {index}")]
    GpuValidationFailed {
        /// Zero-based transition index reported by the device.
        index: usize,
    },
    /// The device computed a digest that differs from the transcript's
    /// claimed `next` accumulator.
    #[error("CUDA Poseidon digest mismatch at transition {index}")]
    GpuPoseidonMismatch {
        /// Zero-based transition index reported by the device.
        index: usize,
        /// Digest claimed by the encoded transcript.
        expected: [u8; 32],
        /// Digest computed by the device.
        actual: [u8; 32],
    },
    /// This target was built without the Linux CUDA sidecar.
    #[error("CUDA Nova sidecar is unavailable in this build or on this host")]
    BackendUnavailable,
    /// The official Nova relation rejected a transition or proof operation.
    #[error(transparent)]
    Nova(#[from] zkfly_nova::TopologyNovaError),
    /// The shared topology commitment rejected the input fixture.
    #[error(transparent)]
    Commitment(#[from] zkfly_commitment::CommitmentError),
    /// A CUDA driver operation failed.
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[error(transparent)]
    Driver(#[from] cuda_core::DriverError),
    /// The generated PTX file could not be read.
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[error("cannot read generated PTX {path}: {source}")]
    PtxIo {
        /// Path of the PTX artifact.
        path: std::path::PathBuf,
        /// Underlying filesystem error.
        source: std::io::Error,
    },
    /// The typed cuda-oxide module could not be attached to the loaded PTX.
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[error(transparent)]
    Module(#[from] cuda_host::EmbeddedModuleError),
    /// The serialized CUDA execution state was poisoned by a host panic.
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[error("CUDA execution state was poisoned")]
    RuntimeStatePoisoned,
}

/// CUDA-backed transcript validator and official Nova proving adapter.
pub struct CudaNovaEngine {
    /// Linux CUDA runtime and generated kernel module when the feature exists.
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    runtime: std::sync::Arc<cuda_runtime::CudaRuntime>,
}

impl CudaNovaEngine {
    /// Loads a CUDA PTX module for one device ordinal.
    ///
    /// The PTX is normally generated from the Rust kernels with `cargo oxide`
    /// on the target V100 host. Keeping the path explicit makes the future
    /// standalone repository independent from this workspace's artifact
    /// layout.
    ///
    /// # Errors
    ///
    /// Returns [`CudaNovaError::BackendUnavailable`] on non-Linux or feature-
    /// free builds, or a typed driver/PTX error when CUDA initialization fails.
    pub fn new<P>(device_ordinal: usize, ptx_path: P) -> Result<Self, CudaNovaError>
    where
        P: AsRef<Path>,
    {
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        {
            let runtime = std::sync::Arc::new(cuda_runtime::CudaRuntime::new(
                device_ordinal,
                ptx_path.as_ref(),
            )?);
            nova_snark::provider::gpu::install_gpu_backend(std::sync::Arc::new(
                cuda_backend::CudaGpuBackend::new(runtime.clone()),
            ));
            return Ok(Self { runtime });
        }

        #[cfg(not(all(feature = "cuda", target_os = "linux")))]
        {
            let _ = (device_ordinal, ptx_path.as_ref());
            Err(CudaNovaError::BackendUnavailable)
        }
    }

    /// Runs the fixed transcript-shape kernel on the selected CUDA device.
    ///
    /// This check covers canonical indices, accumulator adjacency, data-length
    /// bounds, and zero padding. It does not perform Poseidon arithmetic; the
    /// subsequent official Nova proof remains the cryptographic check.
    ///
    /// # Errors
    ///
    /// Returns an error when the trace is malformed, the device rejects a
    /// transition, or CUDA is unavailable in this build.
    pub fn validate_steps(
        &self,
        steps: &[TopologyFoldStep],
    ) -> Result<ValidationReport, CudaNovaError> {
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        {
            return self.runtime.validate_steps(steps);
        }

        #[cfg(not(all(feature = "cuda", target_os = "linux")))]
        {
            let _ = steps;
            Err(CudaNovaError::BackendUnavailable)
        }
    }

    /// Validates the complete Poseidon transition on the selected CUDA device.
    ///
    /// In addition to the fixed transcript-shape predicates checked by
    /// [`Self::validate_steps`], this kernel performs the BN254 scalar-field
    /// permutation over canonical little-endian limbs and compares its digest
    /// with each encoded `next` accumulator. It establishes the device-side
    /// hash relation before the official Nova synthesis and fold.
    ///
    /// # Errors
    ///
    /// Returns an error when the trace is malformed, a device lane rejects a
    /// transition or digest, Poseidon parameters cannot be prepared, or CUDA
    /// is unavailable in this build.
    pub fn validate_poseidon_steps(
        &self,
        steps: &[TopologyFoldStep],
    ) -> Result<ValidationReport, CudaNovaError> {
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        {
            return self.runtime.validate_poseidon_steps(steps);
        }

        #[cfg(not(all(feature = "cuda", target_os = "linux")))]
        {
            let _ = steps;
            Err(CudaNovaError::BackendUnavailable)
        }
    }

    /// Repeatedly validates one transcript while keeping device buffers
    /// resident for the whole benchmark.
    ///
    /// This is the first reusable GPU boundary for larger batches: upload and
    /// allocation happen once, while the kernel is launched `iterations`
    /// times against the same encoded topology. The method does not create a
    /// proof and therefore does not claim that Poseidon or Nova synthesis is
    /// accelerated.
    ///
    /// # Errors
    ///
    /// Returns an error when the transcript is malformed, `iterations` is zero,
    /// a device lane rejects the transcript, or CUDA is unavailable in this
    /// build.
    pub fn benchmark_steps(
        &self,
        steps: &[TopologyFoldStep],
        iterations: usize,
    ) -> Result<ResidentValidationReport, CudaNovaError> {
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        {
            return self.runtime.benchmark_steps(steps, iterations);
        }

        #[cfg(not(all(feature = "cuda", target_os = "linux")))]
        {
            let _ = (steps, iterations);
            Err(CudaNovaError::BackendUnavailable)
        }
    }

    /// GPU-preflights a transcript and then proves it with official Nova.
    ///
    /// Nova owns the complete Bellpepper R1CS synthesis and recursive protocol.
    /// When CUDA is available, the vendored Nova arithmetic hooks dispatch
    /// A/B/C `SpMV`, cross-term evaluation, and relaxed-witness vector folds to
    /// this engine; MSM remains the official provider with a CPU fallback.
    ///
    /// # Errors
    ///
    /// Returns a CUDA sidecar error or the official Nova proving error.
    pub fn prove(
        &self,
        steps: &[TopologyFoldStep],
    ) -> Result<zkfly_nova::TopologyNovaProof, CudaNovaError> {
        self.validate_poseidon_steps(steps)?;
        Ok(zkfly_nova::TopologyNovaProof::prove(steps)?)
    }

    /// GPU-preflights a transcript and proves it against an external root.
    ///
    /// This is the preferred application boundary when the CSR topology root
    /// is already a public instance. Nova still performs the complete R1CS
    /// synthesis and recursive fold; this method only adds the root assertion
    /// around that official proof.
    ///
    /// # Errors
    ///
    /// Returns a CUDA sidecar error or a root/Nova proving error.
    pub fn prove_for_root(
        &self,
        steps: &[TopologyFoldStep],
        claimed_root: Commitment,
    ) -> Result<zkfly_nova::TopologyNovaProof, CudaNovaError> {
        self.validate_poseidon_steps(steps)?;
        Ok(zkfly_nova::TopologyNovaProof::prove_for_root(
            steps,
            claimed_root,
        )?)
    }

    /// Proves a sequence of private weighted forward passes for one fixed CSR
    /// topology.
    ///
    /// The topology, input commitments, and output commitment are carried by
    /// the official Nova circuit; edge weights and vectors remain private
    /// witnesses. Once this engine is initialized, Nova's arithmetic backend
    /// hooks route supported R1CS `SpMV`, relaxed-witness folds, and NIFS
    /// cross-terms through the resident CUDA runtime. Bellpepper synthesis,
    /// recursive orchestration, and MSM retain Nova's official CPU path.
    ///
    /// # Errors
    ///
    /// Returns [`CudaNovaError::BackendUnavailable`] when this build does not
    /// include the Linux CUDA sidecar, or the official Nova error when witness
    /// chaining or circuit synthesis fails.
    pub fn prove_weighted_forward(
        &self,
        topology: std::sync::Arc<zkfly_nova::CsrTopology>,
        witnesses: &[zkfly_nova::WeightedForwardWitness],
    ) -> Result<zkfly_nova::WeightedForwardProof, CudaNovaError> {
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        {
            let _ = &self.runtime;
            return Ok(zkfly_nova::WeightedForwardProof::prove(
                topology, witnesses,
            )?);
        }

        #[cfg(not(all(feature = "cuda", target_os = "linux")))]
        {
            let _ = (topology, witnesses);
            Err(CudaNovaError::BackendUnavailable)
        }
    }

    /// Returns the successful CUDA arithmetic launches performed by this engine.
    #[must_use]
    pub fn gpu_backend_stats(&self) -> GpuBackendStats {
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        {
            let (spmv_calls, fold_calls, cross_term_calls) = self.runtime.backend_stats();
            return GpuBackendStats {
                spmv_calls,
                fold_calls,
                cross_term_calls,
            };
        }

        #[cfg(not(all(feature = "cuda", target_os = "linux")))]
        {
            GpuBackendStats {
                spmv_calls: 0,
                fold_calls: 0,
                cross_term_calls: 0,
            }
        }
    }

    /// Runs a deterministic check of Nova's official GPU MSM provider.
    #[must_use]
    pub fn gpu_msm_self_test(&self) -> bool {
        #[cfg(all(
            feature = "cuda",
            feature = "gpu-msm",
            target_os = "linux",
            target_arch = "x86_64"
        ))]
        {
            return cuda_backend::msm_self_test();
        }

        #[cfg(not(all(
            feature = "cuda",
            feature = "gpu-msm",
            target_os = "linux",
            target_arch = "x86_64"
        )))]
        {
            false
        }
    }
}

/// Returns whether the optional official Blitzar MSM provider is compiled in.
#[must_use]
pub const fn gpu_msm_enabled() -> bool {
    cfg!(all(
        feature = "cuda",
        feature = "gpu-msm",
        target_os = "linux",
        target_arch = "x86_64"
    ))
}

/// Returns whether this build includes the Linux CUDA sidecar.
#[must_use]
pub const fn cuda_backend_enabled() -> bool {
    cfg!(all(feature = "cuda", target_os = "linux"))
}

/// Encodes a transcript into the stable byte layout consumed by the device.
///
/// Each step is laid out as `index || previous || data[11] || data_len ||
/// padding[7] || next`, using little-endian bytes for the index and canonical
/// commitment bytes for every field. This function is public so a future
/// standalone verifier can reproduce the device input without depending on
/// private CUDA types.
///
/// # Errors
///
/// Returns an error when the trace is empty, its encoded size overflows, a
/// data-length marker exceeds the fixed Poseidon rate, or padding is non-zero.
pub fn encode_steps(steps: &[TopologyFoldStep]) -> Result<Vec<u8>, CudaNovaError> {
    if steps.is_empty() {
        return Err(CudaNovaError::EmptyTrace);
    }
    let byte_count =
        steps
            .len()
            .checked_mul(ENCODED_STEP_BYTES)
            .ok_or(CudaNovaError::SizeOverflow {
                target: "encoded transcript byte count",
            })?;
    let mut encoded = Vec::with_capacity(byte_count);
    for step in steps {
        if usize::from(step.data_len) > POSEIDON_RATE {
            return Err(CudaNovaError::InvalidTranscript {
                proposition: "data_len does not exceed the Poseidon rate",
            });
        }
        if step
            .data
            .iter()
            .skip(usize::from(step.data_len))
            .any(|field| field.iter().any(|byte| *byte != 0))
        {
            return Err(CudaNovaError::Nova(
                zkfly_nova::TopologyNovaError::NonZeroPadding { index: step.index },
            ));
        }
        encoded.extend_from_slice(&step.index.to_le_bytes());
        encoded.extend_from_slice(&step.previous);
        for field in &step.data {
            encoded.extend_from_slice(field);
        }
        encoded.push(step.data_len);
        encoded.extend(std::iter::repeat_n(0_u8, 7));
        encoded.extend_from_slice(&step.next);
    }
    debug_assert_eq!(encoded.len(), byte_count);
    Ok(encoded)
}

#[cfg(test)]
mod tests {
    use super::{ENCODED_STEP_BYTES, encode_steps};
    use zkfly_commitment::commit_topology_with_trace;

    /// Ensures the host layout remains a fixed-width projection of the shared
    /// commitment transcript.
    #[test]
    fn encodes_one_canonical_step_at_fixed_width() {
        let mut steps = Vec::new();
        let result = commit_topology_with_trace(1, &[0, 0], &[], |step| steps.push(step));
        assert!(result.is_ok());
        let encoded = encode_steps(&steps);
        assert!(encoded.is_ok());
        assert_eq!(encoded.map_or(0, |value| value.len()), ENCODED_STEP_BYTES);
    }
}

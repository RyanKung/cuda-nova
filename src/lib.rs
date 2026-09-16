//! CUDA sidecar for the official `nova-snark` topology prover.
//!
//! The crate is intentionally shaped as a future standalone repository. The
//! CPU proof relation lives in [`zkfly_nova`], while this crate owns the CUDA
//! context, PTX module, device buffers, and GPU preflight boundary. The first
//! implementation validates the canonical fold transcript on the GPU and then
//! delegates cryptographic Nova synthesis to the official CPU implementation.
//! That explicit boundary prevents a GPU data-movement benchmark from being
//! mistaken for a complete GPU Nova prover.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

use std::path::Path;

use thiserror::Error;
use zkfly_commitment::{POSEIDON_RATE, TopologyFoldStep};

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
    /// A device lane rejected one encoded transition.
    #[error("CUDA transcript preflight rejected transition {index}")]
    GpuValidationFailed {
        /// Zero-based transition index reported by the device.
        index: usize,
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
    runtime: cuda_runtime::CudaRuntime,
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
            return Ok(Self {
                runtime: cuda_runtime::CudaRuntime::new(device_ordinal, ptx_path.as_ref())?,
            });
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

    /// GPU-preflights a transcript and then proves it with official Nova.
    ///
    /// The method is deliberately honest about the current acceleration
    /// boundary: device validation and transfer are CUDA-backed, while R1CS
    /// synthesis, Poseidon witness arithmetic, and recursive folding are still
    /// performed by [`zkfly_nova::TopologyNovaProof`].
    ///
    /// # Errors
    ///
    /// Returns a CUDA sidecar error or the official Nova proving error.
    pub fn prove(
        &self,
        steps: &[TopologyFoldStep],
    ) -> Result<zkfly_nova::TopologyNovaProof, CudaNovaError> {
        self.validate_steps(steps)?;
        Ok(zkfly_nova::TopologyNovaProof::prove(steps)?)
    }
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

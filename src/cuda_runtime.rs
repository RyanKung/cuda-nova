//! Linux CUDA runtime for the standalone `cuda-nova` sidecar.

use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, launch_bounds, thread};
use cuda_host::cuda_module;
use zkfly_commitment::TopologyFoldStep;

use crate::{CudaNovaError, ENCODED_STEP_BYTES, ValidationReport, encode_steps};

/// Number of CUDA threads in one launch block.
const BLOCK_SIZE: u32 = 256;

/// Offset of the transition index in one encoded step.
const INDEX_OFFSET: usize = 0;

/// Offset of the previous accumulator in one encoded step.
const PREVIOUS_OFFSET: usize = 8;

/// Offset of the eleven fixed data fields in one encoded step.
const DATA_OFFSET: usize = 40;

/// Offset of the active data-field count in one encoded step.
const DATA_LEN_OFFSET: usize = 392;

/// Offset of the next accumulator in one encoded step.
const NEXT_OFFSET: usize = 400;

/// Typed CUDA context and generated validation module.
pub(crate) struct CudaRuntime {
    /// Context retained for all device allocations and launches.
    context: Arc<CudaContext>,
    /// PTX module generated from the Rust kernel below.
    module: kernels::LoadedModule,
    /// Serializes context binding and buffer teardown for this MVP.
    execution_lock: Mutex<()>,
}

#[cuda_module]
mod kernels {
    use super::*;

    /// Checks the canonical fixed-width transcript shape one transition per
    /// CUDA lane, including the adjacency relation between neighboring steps.
    #[kernel]
    #[launch_bounds(256)]
    pub fn validate_topology_steps(
        encoded: &[u8],
        step_size: u32,
        step_count: u32,
        mut valid: DisjointSlice<u32>,
    ) {
        let index = thread::index_1d();
        let index_value = index.get();
        let is_valid =
            if let Some(base) = index_value.checked_mul(usize::try_from(step_size).unwrap_or(0)) {
                let expected_index = u64::try_from(index_value).unwrap_or(u64::MAX);
                let actual_index = read_u64_le(encoded, base + INDEX_OFFSET).unwrap_or(u64::MAX);
                let mut is_valid = actual_index == expected_index;
                is_valid = is_valid
                    && has_expected_previous(
                        encoded,
                        base,
                        index_value,
                        usize::try_from(step_size).unwrap_or(0),
                    );
                if let Some(data_len) = encoded.get(base + DATA_LEN_OFFSET).copied() {
                    is_valid = is_valid && usize::from(data_len) <= zkfly_commitment::POSEIDON_RATE;
                    let mut field = usize::from(data_len);
                    while field < zkfly_commitment::POSEIDON_RATE {
                        let field_offset = field.checked_mul(32).unwrap_or(usize::MAX);
                        is_valid = is_valid
                            && bytes_are_zero(encoded, base + DATA_OFFSET + field_offset, 32);
                        field = field.saturating_add(1);
                    }
                    is_valid = is_valid && bytes_are_zero(encoded, base + DATA_LEN_OFFSET + 1, 7);
                } else {
                    is_valid = false;
                }
                is_valid
            } else {
                false
            };
        let within_count = index_value < usize::try_from(step_count).unwrap_or(0);
        if let Some(destination) = valid.get_mut(index) {
            *destination = if within_count && is_valid {
                1_u32
            } else {
                0_u32
            };
        }
    }

    /// Reads an eight-byte little-endian integer without unchecked indexing.
    fn read_u64_le(encoded: &[u8], offset: usize) -> Option<u64> {
        let mut value = 0_u64;
        let mut byte = 0_usize;
        while byte < 8 {
            let position = offset.checked_add(byte)?;
            let value_byte = u64::from(*encoded.get(position)?);
            value |= value_byte << (byte * 8);
            byte = byte.checked_add(1)?;
        }
        Some(value)
    }

    /// Compares two fixed-width byte ranges in the encoded transcript.
    fn bytes_equal(encoded: &[u8], left_offset: usize, right_offset: usize, width: usize) -> bool {
        let mut byte = 0_usize;
        while byte < width {
            let Some(left) = left_offset.checked_add(byte).and_then(|i| encoded.get(i)) else {
                return false;
            };
            let Some(right) = right_offset.checked_add(byte).and_then(|i| encoded.get(i)) else {
                return false;
            };
            if left != right {
                return false;
            }
            byte = match byte.checked_add(1) {
                Some(next) => next,
                None => return false,
            };
        }
        true
    }

    /// Checks that a fixed-width byte range contains only zero bytes.
    fn bytes_are_zero(encoded: &[u8], offset: usize, width: usize) -> bool {
        let mut byte = 0_usize;
        while byte < width {
            let Some(value) = offset.checked_add(byte).and_then(|i| encoded.get(i)) else {
                return false;
            };
            if *value != 0 {
                return false;
            }
            byte = match byte.checked_add(1) {
                Some(next) => next,
                None => return false,
            };
        }
        true
    }

    /// Checks the zero root or the previous step's next accumulator.
    fn has_expected_previous(
        encoded: &[u8],
        base: usize,
        lane_index: usize,
        step_size: usize,
    ) -> bool {
        if lane_index == 0 {
            return bytes_are_zero(encoded, base + PREVIOUS_OFFSET, 32);
        }
        let Some(previous_base) = lane_index
            .checked_sub(1)
            .and_then(|index| index.checked_mul(step_size))
        else {
            return false;
        };
        bytes_equal(
            encoded,
            base + PREVIOUS_OFFSET,
            previous_base + NEXT_OFFSET,
            32,
        )
    }
}

impl CudaRuntime {
    /// Creates a CUDA context and loads the caller-provided PTX module.
    pub(crate) fn new(device_ordinal: usize, ptx_path: &Path) -> Result<Self, CudaNovaError> {
        let context = CudaContext::new(device_ordinal)?;
        let ptx = fs::read_to_string(ptx_path).map_err(|source| CudaNovaError::PtxIo {
            path: ptx_path.to_path_buf(),
            source,
        })?;
        let module = context.load_module_from_ptx_src(&ptx)?;
        let module = kernels::from_module(module)?;
        Ok(Self {
            context,
            module,
            execution_lock: Mutex::new(()),
        })
    }

    /// Uploads and validates a canonical transcript on the CUDA device.
    pub(crate) fn validate_steps(
        &self,
        steps: &[TopologyFoldStep],
    ) -> Result<ValidationReport, CudaNovaError> {
        let end_to_end_start = Instant::now();
        let encoded = encode_steps(steps)?;
        let step_count = u32::try_from(steps.len()).map_err(|_| CudaNovaError::SizeOverflow {
            target: "CUDA step count",
        })?;
        let step_size =
            u32::try_from(ENCODED_STEP_BYTES).map_err(|_| CudaNovaError::SizeOverflow {
                target: "CUDA encoded step size",
            })?;
        let _guard = self
            .execution_lock
            .lock()
            .map_err(|_| CudaNovaError::RuntimeStatePoisoned)?;
        self.context.bind_to_thread()?;
        let stream = self.context.default_stream();
        let upload_start = Instant::now();
        let encoded_device = DeviceBuffer::from_host(&stream, &encoded)?;
        let mut valid_device = DeviceBuffer::<u32>::zeroed(&stream, steps.len())?;
        stream.synchronize()?;
        let upload_microseconds = duration_microseconds(upload_start.elapsed());
        let launch = launch_config(steps.len())?;
        let kernel_start = Instant::now();
        self.module.validate_topology_steps(
            &stream,
            launch,
            &encoded_device,
            step_size,
            step_count,
            &mut valid_device,
        )?;
        stream.synchronize()?;
        let kernel_microseconds = duration_microseconds(kernel_start.elapsed());
        let download_start = Instant::now();
        let valid = valid_device.to_host_vec(&stream)?;
        let download_microseconds = duration_microseconds(download_start.elapsed());
        if let Some((index, _)) = valid.iter().enumerate().find(|(_, value)| **value != 1_u32) {
            return Err(CudaNovaError::GpuValidationFailed { index });
        }
        Ok(ValidationReport {
            steps: steps.len(),
            encoded_bytes: encoded.len(),
            upload_microseconds,
            kernel_microseconds,
            download_microseconds,
            end_to_end_microseconds: duration_microseconds(end_to_end_start.elapsed()),
        })
    }
}

/// Builds a one-dimensional launch covering one lane per transcript step.
fn launch_config(step_count: usize) -> Result<LaunchConfig, CudaNovaError> {
    let step_count = u32::try_from(step_count).map_err(|_| CudaNovaError::SizeOverflow {
        target: "CUDA launch step count",
    })?;
    Ok(LaunchConfig {
        grid_dim: (step_count.div_ceil(BLOCK_SIZE), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    })
}

/// Converts a host duration into a bounded microsecond report value.
fn duration_microseconds(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

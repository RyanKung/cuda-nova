//! Bounded device-buffer reuse for Nova's serialized CUDA arithmetic path.

use std::collections::VecDeque;

use cuda_core::{CudaStream, DeviceBuffer};

use crate::CudaNovaError;

/// Number of 64-bit limbs in one device field element.
pub(crate) type FieldLimbs = [u64; 4];

/// Maximum number of exact CSR matrices retained by one runtime.
const MAX_CSR_MATRICES: usize = 16;

/// Maximum number of exact three-matrix CSR batches retained by one runtime.
const MAX_CSR_BATCHES: usize = 8;

/// Maximum number of reusable arithmetic output lengths retained by one runtime.
const MAX_OUTPUT_BUFFERS: usize = 8;

/// Borrowed canonical inputs identifying one sparse matrix.
#[derive(Clone, Copy)]
pub(crate) struct CsrMatrixInput<'a> {
    /// Canonical CSR row offsets.
    pub(crate) row_offsets: &'a [usize],
    /// Canonical CSR column indices.
    pub(crate) column_indices: &'a [usize],
    /// Canonical packed field values.
    pub(crate) values: &'a [u8],
}

/// One cache access together with the allocation work it required.
pub(crate) struct CacheAccess<T> {
    /// Cached value selected for the current launch.
    pub(crate) value: T,
    /// Whether an existing exact-key entry satisfied the request.
    pub(crate) hit: bool,
    /// Number of new CUDA device buffers allocated for this access.
    pub(crate) device_buffer_allocations: u64,
}

/// Exact host identity and resident buffers for one sparse matrix.
pub(crate) struct CachedCsrMatrix {
    /// Original row offsets used for collision-free cache matching.
    row_offsets_key: Vec<usize>,
    /// Original column indices used for collision-free cache matching.
    column_indices_key: Vec<usize>,
    /// Canonical field bytes used for collision-free cache matching.
    values_key: Vec<u8>,
    /// Width of each canonical field encoding.
    field_width_key: usize,
    /// Canonical scalar modulus identifying the Nova cycle field.
    modulus_key: Vec<u8>,
    /// Converted row offsets retained until their initial upload completes.
    _row_offsets_source: Vec<u32>,
    /// Converted column indices retained until their initial upload completes.
    _column_indices_source: Vec<u32>,
    /// Montgomery values retained until their initial upload completes.
    _values_source: Vec<FieldLimbs>,
    /// Device-resident CSR row offsets.
    pub(crate) row_offsets_device: DeviceBuffer<u32>,
    /// Device-resident CSR column indices.
    pub(crate) column_indices_device: DeviceBuffer<u32>,
    /// Device-resident Montgomery matrix values.
    pub(crate) values_device: DeviceBuffer<FieldLimbs>,
    /// Device-resident output overwritten by every matching launch.
    pub(crate) output_device: DeviceBuffer<FieldLimbs>,
}

impl CachedCsrMatrix {
    /// Returns whether all canonical host inputs match this resident matrix.
    fn matches(
        &self,
        row_offsets: &[usize],
        column_indices: &[usize],
        values: &[u8],
        field_width: usize,
        modulus: &[u8],
    ) -> bool {
        self.row_offsets_key == row_offsets
            && self.column_indices_key == column_indices
            && self.values_key == values
            && self.field_width_key == field_width
            && self.modulus_key == modulus
    }
}

/// Exact canonical identity for one member of a batched CSR request.
struct CsrMatrixKey {
    /// Original row offsets used for collision-free cache matching.
    row_offsets_key: Vec<usize>,
    /// Original column indices used for collision-free cache matching.
    column_indices_key: Vec<usize>,
    /// Canonical field bytes used for collision-free cache matching.
    values_key: Vec<u8>,
}

impl CsrMatrixKey {
    /// Returns whether all canonical host inputs match this resident matrix.
    fn matches(&self, input: CsrMatrixInput<'_>) -> bool {
        self.row_offsets_key == input.row_offsets
            && self.column_indices_key == input.column_indices
            && self.values_key == input.values
    }
}

/// Three exact CSR matrices and their shared device output allocation.
pub(crate) struct CachedCsrBatch {
    /// A, B, and C canonical identities in Nova relation order.
    matrices: [CsrMatrixKey; 3],
    /// Width of each canonical field encoding.
    field_width_key: usize,
    /// Canonical scalar modulus identifying the Nova cycle field.
    modulus_key: Vec<u8>,
    /// Number of rows in each of the three matrices.
    rows: usize,
    /// Combined row offsets retained until their initial upload completes.
    _row_offsets_source: Vec<u32>,
    /// Combined column indices retained until their initial upload completes.
    _column_indices_source: Vec<u32>,
    /// Combined Montgomery values retained until their initial upload completes.
    _values_source: Vec<FieldLimbs>,
    /// Device-resident CSR row offsets for the combined A/B/C matrix.
    pub(crate) row_offsets_device: DeviceBuffer<u32>,
    /// Device-resident CSR column indices for the combined A/B/C matrix.
    pub(crate) column_indices_device: DeviceBuffer<u32>,
    /// Device-resident Montgomery values for the combined A/B/C matrix.
    pub(crate) values_device: DeviceBuffer<FieldLimbs>,
    /// Device output laid out as contiguous A, B, and C row ranges.
    pub(crate) output_device: DeviceBuffer<FieldLimbs>,
}

impl CachedCsrBatch {
    /// Returns whether the complete three-matrix request matches this batch.
    fn matches(
        &self,
        inputs: [CsrMatrixInput<'_>; 3],
        field_width: usize,
        modulus: &[u8],
        rows: usize,
    ) -> bool {
        self.field_width_key == field_width
            && self.modulus_key == modulus
            && self.rows == rows
            && self
                .matrices
                .iter()
                .zip(inputs)
                .all(|(matrix, input)| matrix.matches(input))
    }
}

/// One reusable output allocation for vector-fold or cross-term kernels.
struct CachedOutput {
    /// Number of field elements stored in the output.
    len: usize,
    /// Device allocation overwritten completely by each kernel launch.
    device: DeviceBuffer<FieldLimbs>,
}

/// Host buffers for A/B/C concatenated as one vertical CSR matrix.
struct CombinedCsrSources {
    /// Row offsets with B and C non-zero counts shifted after prior matrices.
    row_offsets: Vec<u32>,
    /// A, B, and C column indices in relation order.
    column_indices: Vec<u32>,
    /// A, B, and C Montgomery values in relation order.
    values: Vec<FieldLimbs>,
}

/// Serialized, bounded set of device allocations reused across Nova steps.
#[derive(Default)]
pub(crate) struct CudaWorkspace {
    /// Exact static sparse matrices in oldest-to-newest insertion order.
    csr_matrices: VecDeque<CachedCsrMatrix>,
    /// Exact A/B/C sparse-matrix batches in oldest-to-newest insertion order.
    csr_batches: VecDeque<CachedCsrBatch>,
    /// Arithmetic outputs keyed by exact element count.
    outputs: VecDeque<CachedOutput>,
}

impl CudaWorkspace {
    /// Selects or uploads an exact CSR matrix and its reusable output buffer.
    pub(crate) fn csr_matrix<F>(
        &mut self,
        stream: &CudaStream,
        row_offsets: &[usize],
        column_indices: &[usize],
        values: &[u8],
        field_width: usize,
        modulus: &[u8],
        rows: usize,
        prepare_values: F,
    ) -> Result<CacheAccess<&mut CachedCsrMatrix>, CudaNovaError>
    where
        F: FnOnce() -> Result<Vec<FieldLimbs>, CudaNovaError>,
    {
        if let Some(position) = self.csr_matrices.iter().position(|entry| {
            entry.matches(row_offsets, column_indices, values, field_width, modulus)
        }) {
            let entry =
                self.csr_matrices
                    .get_mut(position)
                    .ok_or(CudaNovaError::InvalidTranscript {
                        proposition: "the located CSR cache entry remains present",
                    })?;
            return Ok(CacheAccess {
                value: entry,
                hit: true,
                device_buffer_allocations: 0,
            });
        }

        let row_offsets_source = convert_indices(row_offsets, "CSR row offset")?;
        let column_indices_source = convert_indices(column_indices, "CSR column index")?;
        let values_source = prepare_values()?;
        let row_offsets_device = DeviceBuffer::from_host(stream, &row_offsets_source)?;
        let column_indices_device = DeviceBuffer::from_host(stream, &column_indices_source)?;
        let values_device = DeviceBuffer::from_host(stream, &values_source)?;
        let output_device = DeviceBuffer::<FieldLimbs>::zeroed(stream, rows)?;
        if self.csr_matrices.len() == MAX_CSR_MATRICES {
            let _ = self.csr_matrices.pop_front();
        }
        self.csr_matrices.push_back(CachedCsrMatrix {
            row_offsets_key: row_offsets.to_vec(),
            column_indices_key: column_indices.to_vec(),
            values_key: values.to_vec(),
            field_width_key: field_width,
            modulus_key: modulus.to_vec(),
            _row_offsets_source: row_offsets_source,
            _column_indices_source: column_indices_source,
            _values_source: values_source,
            row_offsets_device,
            column_indices_device,
            values_device,
            output_device,
        });
        let entry = self
            .csr_matrices
            .back_mut()
            .ok_or(CudaNovaError::InvalidTranscript {
                proposition: "a newly inserted CSR cache entry is available",
            })?;
        Ok(CacheAccess {
            value: entry,
            hit: false,
            device_buffer_allocations: 4,
        })
    }

    /// Selects or uploads one exact A/B/C batch and its shared output buffer.
    pub(crate) fn csr_batch<F>(
        &mut self,
        stream: &CudaStream,
        inputs: [CsrMatrixInput<'_>; 3],
        field_width: usize,
        modulus: &[u8],
        rows: usize,
        prepare_values: F,
    ) -> Result<CacheAccess<&mut CachedCsrBatch>, CudaNovaError>
    where
        F: FnOnce() -> Result<[Vec<FieldLimbs>; 3], CudaNovaError>,
    {
        if let Some(position) = self
            .csr_batches
            .iter()
            .position(|entry| entry.matches(inputs, field_width, modulus, rows))
        {
            let entry =
                self.csr_batches
                    .get_mut(position)
                    .ok_or(CudaNovaError::InvalidTranscript {
                        proposition: "the located CSR batch cache entry remains present",
                    })?;
            return Ok(CacheAccess {
                value: entry,
                hit: true,
                device_buffer_allocations: 0,
            });
        }

        let batch = build_cached_csr_batch(stream, inputs, prepare_values()?, rows)?;
        if self.csr_batches.len() == MAX_CSR_BATCHES {
            let _ = self.csr_batches.pop_front();
        }
        self.csr_batches.push_back(CachedCsrBatch {
            field_width_key: field_width,
            modulus_key: modulus.to_vec(),
            rows,
            ..batch
        });
        let entry = self
            .csr_batches
            .back_mut()
            .ok_or(CudaNovaError::InvalidTranscript {
                proposition: "a newly inserted CSR batch cache entry is available",
            })?;
        Ok(CacheAccess {
            value: entry,
            hit: false,
            device_buffer_allocations: 4,
        })
    }

    /// Selects or allocates an output buffer with the requested exact length.
    pub(crate) fn output(
        &mut self,
        stream: &CudaStream,
        len: usize,
    ) -> Result<CacheAccess<&mut DeviceBuffer<FieldLimbs>>, CudaNovaError> {
        if let Some(position) = self.outputs.iter().position(|entry| entry.len == len) {
            let entry = self
                .outputs
                .get_mut(position)
                .ok_or(CudaNovaError::InvalidTranscript {
                    proposition: "the located output cache entry remains present",
                })?;
            return Ok(CacheAccess {
                value: &mut entry.device,
                hit: true,
                device_buffer_allocations: 0,
            });
        }

        let device = DeviceBuffer::<FieldLimbs>::zeroed(stream, len)?;
        if self.outputs.len() == MAX_OUTPUT_BUFFERS {
            let _ = self.outputs.pop_front();
        }
        self.outputs.push_back(CachedOutput { len, device });
        let entry = self
            .outputs
            .back_mut()
            .ok_or(CudaNovaError::InvalidTranscript {
                proposition: "a newly inserted output cache entry is available",
            })?;
        Ok(CacheAccess {
            value: &mut entry.device,
            hit: false,
            device_buffer_allocations: 1,
        })
    }
}

/// Combines A/B/C vertically and uploads one immutable CSR representation.
fn build_cached_csr_batch(
    stream: &CudaStream,
    inputs: [CsrMatrixInput<'_>; 3],
    values: [Vec<FieldLimbs>; 3],
    rows: usize,
) -> Result<CachedCsrBatch, CudaNovaError> {
    let matrices = inputs.map(csr_matrix_key);
    let sources = combine_csr_sources(inputs, values)?;
    let row_offsets_device = DeviceBuffer::from_host(stream, &sources.row_offsets)?;
    let column_indices_device = DeviceBuffer::from_host(stream, &sources.column_indices)?;
    let values_device = DeviceBuffer::from_host(stream, &sources.values)?;
    let output_len = rows.checked_mul(3).ok_or(CudaNovaError::SizeOverflow {
        target: "three-matrix CUDA output length",
    })?;
    let output_device = DeviceBuffer::<FieldLimbs>::zeroed(stream, output_len)?;
    Ok(CachedCsrBatch {
        matrices,
        field_width_key: 0,
        modulus_key: Vec::new(),
        rows,
        _row_offsets_source: sources.row_offsets,
        _column_indices_source: sources.column_indices,
        _values_source: sources.values,
        row_offsets_device,
        column_indices_device,
        values_device,
        output_device,
    })
}

/// Concatenates three equal-width CSR matrices without changing column indices.
fn combine_csr_sources(
    inputs: [CsrMatrixInput<'_>; 3],
    values: [Vec<FieldLimbs>; 3],
) -> Result<CombinedCsrSources, CudaNovaError> {
    let mut row_offsets_source = vec![0_u32];
    let mut column_indices_source = Vec::new();
    let mut value_base = 0_usize;
    for input in inputs {
        for offset in input.row_offsets.iter().copied().skip(1) {
            let combined = value_base
                .checked_add(offset)
                .ok_or(CudaNovaError::SizeOverflow {
                    target: "combined CSR row offset",
                })?;
            row_offsets_source.push(u32::try_from(combined).map_err(|_| {
                CudaNovaError::SizeOverflow {
                    target: "combined CSR row offset",
                }
            })?);
        }
        column_indices_source.extend(convert_indices(input.column_indices, "CSR column index")?);
        value_base = value_base.checked_add(input.column_indices.len()).ok_or(
            CudaNovaError::SizeOverflow {
                target: "combined CSR value count",
            },
        )?;
    }
    let mut values_source = Vec::new();
    for matrix_values in values {
        values_source.extend(matrix_values);
    }
    Ok(CombinedCsrSources {
        row_offsets: row_offsets_source,
        column_indices: column_indices_source,
        values: values_source,
    })
}

/// Owns the collision-free canonical cache identity for one matrix.
fn csr_matrix_key(input: CsrMatrixInput<'_>) -> CsrMatrixKey {
    CsrMatrixKey {
        row_offsets_key: input.row_offsets.to_vec(),
        column_indices_key: input.column_indices.to_vec(),
        values_key: input.values.to_vec(),
    }
}

/// Converts host indices to the CUDA kernel's checked 32-bit representation.
fn convert_indices(values: &[usize], target: &'static str) -> Result<Vec<u32>, CudaNovaError> {
    values
        .iter()
        .copied()
        .map(|value| u32::try_from(value).map_err(|_| CudaNovaError::SizeOverflow { target }))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Proves that vertical A/B/C concatenation shifts only CSR row offsets.
    #[test]
    fn combines_three_csr_matrices_in_relation_order() {
        let a_rows = [0_usize, 2, 2];
        let b_rows = [0_usize, 1, 3];
        let c_rows = [0_usize, 0, 1];
        let a_columns = [0_usize, 2];
        let b_columns = [1_usize, 0, 2];
        let c_columns = [1_usize];
        let a_bytes = [1_u8, 2];
        let b_bytes = [3_u8, 4, 5];
        let c_bytes = [6_u8];
        let inputs = [
            CsrMatrixInput {
                row_offsets: &a_rows,
                column_indices: &a_columns,
                values: &a_bytes,
            },
            CsrMatrixInput {
                row_offsets: &b_rows,
                column_indices: &b_columns,
                values: &b_bytes,
            },
            CsrMatrixInput {
                row_offsets: &c_rows,
                column_indices: &c_columns,
                values: &c_bytes,
            },
        ];
        let values = [
            vec![[11_u64, 0, 0, 0], [12_u64, 0, 0, 0]],
            vec![[21_u64, 0, 0, 0], [22_u64, 0, 0, 0], [23_u64, 0, 0, 0]],
            vec![[31_u64, 0, 0, 0]],
        ];
        let combined = combine_csr_sources(inputs, values);
        assert!(combined.is_ok());
        let Ok(combined) = combined else {
            return;
        };
        assert_eq!(combined.row_offsets, [0_u32, 2, 2, 3, 5, 5, 6]);
        assert_eq!(combined.column_indices, [0_u32, 2, 1, 0, 2, 1]);
        assert_eq!(
            combined.values,
            [
                [11_u64, 0, 0, 0],
                [12_u64, 0, 0, 0],
                [21_u64, 0, 0, 0],
                [22_u64, 0, 0, 0],
                [23_u64, 0, 0, 0],
                [31_u64, 0, 0, 0],
            ]
        );
    }
}

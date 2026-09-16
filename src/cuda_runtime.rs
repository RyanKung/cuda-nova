//! Linux CUDA runtime for the standalone `cuda-nova` sidecar.

use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ark_bn254::Fr;
use ark_ff::{BigInteger, PrimeField};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, launch_bounds, thread};
use cuda_host::cuda_module;
use light_poseidon::parameters::bn254_x5::get_poseidon_parameters;
use topology_commitment::{COMMITMENT_BYTES, POSEIDON_INPUTS, TOPOLOGY_DOMAIN, TopologyFoldStep};

use crate::cuda_workspace::{CsrMatrixInput, CudaWorkspace, FieldLimbs};
use crate::{
    CudaNovaError, ENCODED_STEP_BYTES, GpuBackendStats, ResidentValidationReport, ValidationReport,
    encode_steps,
};

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

/// Number of 64-bit limbs in one BN254 scalar.
const FIELD_LIMBS: usize = 4;

/// Width of the fixed Poseidon state used by the topology relation.
const POSEIDON_WIDTH: usize = 13;

/// Number of data fields in one fixed Poseidon fold call.
const POSEIDON_RATE_DEVICE: usize = 11;

/// Least-significant limb of the BN254 scalar modulus.
const FIELD_MODULUS_0: u64 = 0x43e1_f593_f000_0001;

/// Second limb of the BN254 scalar modulus.
const FIELD_MODULUS_1: u64 = 0x2833_e848_79b9_7091;

/// Third limb of the BN254 scalar modulus.
const FIELD_MODULUS_2: u64 = 0xb850_45b6_8181_585d;

/// Most-significant limb of the BN254 scalar modulus.
const FIELD_MODULUS_3: u64 = 0x3064_4e72_e131_a029;

/// Montgomery reduction parameters for one Nova cycle scalar field.
#[derive(Clone, Copy)]
struct DeviceFieldConfig {
    /// Modulus in little-endian 64-bit limbs.
    modulus: FieldLimbs,
    /// Negative least-significant modulus inverse modulo `2^32`.
    modulus_inverse_word: u32,
    /// `R^2 mod modulus`, where `R = 2^256`.
    montgomery_r2: FieldLimbs,
}

/// Least-significant limb of `R mod p`, the Montgomery representation of one.
const MONTGOMERY_ONE_0: u64 = 0xac96_341c_4fff_fffb;

/// Second limb of `R mod p`.
const MONTGOMERY_ONE_1: u64 = 0x36fc_7695_9f60_cd29;

/// Third limb of `R mod p`.
const MONTGOMERY_ONE_2: u64 = 0x666e_a36f_7879_462e;

/// Most-significant limb of `R mod p`.
const MONTGOMERY_ONE_3: u64 = 0x0e0a_77c1_9a07_df2f;

/// Negative primary modulus inverse modulo `2^32`.
const MONTGOMERY_MODULUS_INVERSE_WORD: u32 = 0xefff_ffff;

/// `R^2 mod p` for the primary field, least-significant limb.
const MONTGOMERY_R2_0: u64 = 0x1bb8_e645_ae21_6da7;

/// `R^2 mod p` for the primary field, second limb.
const MONTGOMERY_R2_1: u64 = 0x53fe_3ab1_e35c_59e3;

/// `R^2 mod p` for the primary field, third limb.
const MONTGOMERY_R2_2: u64 = 0x8c49_833d_53bb_8085;

/// `R^2 mod p` for the primary field, most-significant limb.
const MONTGOMERY_R2_3: u64 = 0x0216_d0b1_7f4e_44a5;

/// Least-significant limb of the Grumpkin scalar modulus (BN254 base field).
const GRUMPKIN_MODULUS_0: u64 = 0x3c20_8c16_d87c_fd47;

/// Second limb of the Grumpkin scalar modulus.
const GRUMPKIN_MODULUS_1: u64 = 0x9781_6a91_6871_ca8d;

/// Third limb of the Grumpkin scalar modulus.
const GRUMPKIN_MODULUS_2: u64 = 0xb850_45b6_8181_585d;

/// Most-significant limb of the Grumpkin scalar modulus.
const GRUMPKIN_MODULUS_3: u64 = 0x3064_4e72_e131_a029;

/// Negative Grumpkin modulus inverse modulo `2^32`.
const GRUMPKIN_MODULUS_INVERSE_WORD: u32 = 0xe486_6389;

/// `R^2 mod p` for the Grumpkin scalar field, least-significant limb.
const GRUMPKIN_R2_0: u64 = 0xf32c_fc5b_538a_fa89;

/// `R^2 mod p` for the Grumpkin scalar field, second limb.
const GRUMPKIN_R2_1: u64 = 0xb5e7_1911_d445_01fb;

/// `R^2 mod p` for the Grumpkin scalar field, third limb.
const GRUMPKIN_R2_2: u64 = 0x47ab_1eff_0a41_7ff6;

/// `R^2 mod p` for the Grumpkin scalar field, most-significant limb.
const GRUMPKIN_R2_3: u64 = 0x06d8_9f71_cab8_351f;

/// Primary BN254 scalar field configuration.
const PRIMARY_FIELD_CONFIG: DeviceFieldConfig = DeviceFieldConfig {
    modulus: [
        FIELD_MODULUS_0,
        FIELD_MODULUS_1,
        FIELD_MODULUS_2,
        FIELD_MODULUS_3,
    ],
    modulus_inverse_word: MONTGOMERY_MODULUS_INVERSE_WORD,
    montgomery_r2: [
        MONTGOMERY_R2_0,
        MONTGOMERY_R2_1,
        MONTGOMERY_R2_2,
        MONTGOMERY_R2_3,
    ],
};

/// Secondary Grumpkin scalar field configuration.
const SECONDARY_FIELD_CONFIG: DeviceFieldConfig = DeviceFieldConfig {
    modulus: [
        GRUMPKIN_MODULUS_0,
        GRUMPKIN_MODULUS_1,
        GRUMPKIN_MODULUS_2,
        GRUMPKIN_MODULUS_3,
    ],
    modulus_inverse_word: GRUMPKIN_MODULUS_INVERSE_WORD,
    montgomery_r2: [GRUMPKIN_R2_0, GRUMPKIN_R2_1, GRUMPKIN_R2_2, GRUMPKIN_R2_3],
};

/// Host-side Poseidon constants converted to the GPU's Montgomery limb form.
struct PoseidonHostParameters {
    /// Flattened round constants in round-major order.
    ark: Vec<FieldLimbs>,
    /// Flattened MDS matrix in row-major order.
    mds: Vec<FieldLimbs>,
    /// Number of full S-box rounds in the fixed permutation.
    full_rounds: u32,
    /// Number of partial S-box rounds in the fixed permutation.
    partial_rounds: u32,
    /// Montgomery-limb representation of the topology domain.
    domain: FieldLimbs,
}

/// Successful arithmetic result plus auditable CUDA resource accounting.
struct ArithmeticRun<T> {
    /// Canonical result returned to Nova.
    output: T,
    /// Whether the request launched a non-empty CUDA kernel.
    launched: bool,
    /// Number of device buffers allocated for this request.
    device_buffer_allocations: u64,
    /// Exact CSR cache outcome and logical matrix count for an `SpMV` request.
    csr_cache_hit: Option<bool>,
    /// Number of logical matrices represented by the CSR cache outcome.
    csr_cache_operations: u64,
    /// Reusable output cache outcome for a fold or cross-term request.
    output_cache_hit: Option<bool>,
    /// Explicit or download-implied stream synchronizations for this request.
    stream_synchronizations: u64,
}

impl ArithmeticRun<Vec<u8>> {
    /// Returns a successful empty request that did not reach the device.
    const fn empty(output: Vec<u8>) -> Self {
        Self {
            output,
            launched: false,
            device_buffer_allocations: 0,
            csr_cache_hit: None,
            csr_cache_operations: 0,
            output_cache_hit: None,
            stream_synchronizations: 0,
        }
    }
}

/// Typed CUDA context and generated validation module.
pub(crate) struct CudaRuntime {
    /// Context retained for all device allocations and launches.
    context: Arc<CudaContext>,
    /// PTX module generated from the Rust kernel below.
    module: kernels::LoadedModule,
    /// Device-resident round constants for the fixed Poseidon permutation.
    poseidon_ark: DeviceBuffer<FieldLimbs>,
    /// Device-resident MDS matrix for the fixed Poseidon permutation.
    poseidon_mds: DeviceBuffer<FieldLimbs>,
    /// Number of full S-box rounds in the device permutation.
    poseidon_full_rounds: u32,
    /// Number of partial S-box rounds in the device permutation.
    poseidon_partial_rounds: u32,
    /// Montgomery-limb topology domain separator.
    poseidon_domain: FieldLimbs,
    /// Serializes launches and owns bounded device allocations reused by them.
    execution_workspace: Mutex<CudaWorkspace>,
    /// Number of successful CSR launches through the Nova backend hook.
    spmv_calls: AtomicU64,
    /// Number of physical kernels used for successful CSR requests.
    spmv_batches: AtomicU64,
    /// Number of successful vector-fold launches through the Nova backend hook.
    fold_calls: AtomicU64,
    /// Number of successful cross-term launches through the Nova backend hook.
    cross_term_calls: AtomicU64,
    /// End-to-end host and device time spent in successful CSR requests.
    spmv_microseconds: AtomicU64,
    /// End-to-end host and device time spent in successful vector folds.
    fold_microseconds: AtomicU64,
    /// End-to-end host and device time spent in successful cross-term requests.
    cross_term_microseconds: AtomicU64,
    /// Number of exact CSR cache hits.
    csr_cache_hits: AtomicU64,
    /// Number of exact CSR cache misses.
    csr_cache_misses: AtomicU64,
    /// Number of reusable arithmetic-output cache hits.
    output_cache_hits: AtomicU64,
    /// Number of reusable arithmetic-output cache misses.
    output_cache_misses: AtomicU64,
    /// Number of device-buffer allocations issued by arithmetic requests.
    device_buffer_allocations: AtomicU64,
    /// Number of stream synchronizations issued by arithmetic requests.
    stream_synchronizations: AtomicU64,
}

#[cuda_module]
mod kernels {
    use super::*;

    /// Returns the BN254 scalar modulus as a runtime device aggregate.
    fn device_field_modulus() -> FieldLimbs {
        [
            FIELD_MODULUS_0,
            FIELD_MODULUS_1,
            FIELD_MODULUS_2,
            FIELD_MODULUS_3,
        ]
    }

    // Keep the arithmetic implementation separate from the transcript
    // kernels so each source file remains small and independently reviewable.
    include!("cuda_runtime_field.rs");

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
        let is_valid = if let Some(base) =
            index_value.checked_mul(usize::try_from(step_size).unwrap_or(0))
        {
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
                is_valid = is_valid && usize::from(data_len) <= topology_commitment::POSEIDON_RATE;
                let mut field = usize::from(data_len);
                while field < topology_commitment::POSEIDON_RATE {
                    let field_offset = field.checked_mul(32).unwrap_or(usize::MAX);
                    is_valid =
                        is_valid && bytes_are_zero(encoded, base + DATA_OFFSET + field_offset, 32);
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

    /// Computes one fixed BN254 Poseidon fold per CUDA lane and checks its
    /// digest against the encoded `next` accumulator.
    #[kernel]
    #[launch_bounds(256)]
    pub fn validate_poseidon_steps(
        encoded: &[u8],
        step_size: u32,
        step_count: u32,
        ark: &[FieldLimbs],
        mds: &[FieldLimbs],
        full_rounds: u32,
        partial_rounds: u32,
        domain: FieldLimbs,
        mut valid: DisjointSlice<u32>,
        mut digests: DisjointSlice<FieldLimbs>,
    ) {
        let index = thread::index_1d();
        let index_value = index.get();
        let within_count = index_value < usize::try_from(step_count).unwrap_or(0);
        let (is_valid, digest) = if within_count {
            if let Some(base) = index_value.checked_mul(usize::try_from(step_size).unwrap_or(0)) {
                validate_poseidon_transition(
                    encoded,
                    base,
                    index_value,
                    usize::try_from(step_size).unwrap_or(0),
                    ark,
                    mds,
                    full_rounds,
                    partial_rounds,
                    domain,
                )
            } else {
                (false, [0_u64; 4])
            }
        } else {
            (false, [0_u64; 4])
        };
        if let Some(destination) = valid.get_mut(index) {
            *destination = if is_valid { 1_u32 } else { 0_u32 };
        }
        if let Some(destination) = digests.get_mut(thread::index_1d()) {
            *destination = digest;
        }
    }

    /// Multiplies one CSR row by a dense Montgomery field vector per CUDA lane.
    #[kernel]
    #[launch_bounds(256)]
    pub fn spmv_rows(
        row_offsets: &[u32],
        column_indices: &[u32],
        values: &[FieldLimbs],
        vector: &[FieldLimbs],
        rows: u32,
        modulus: FieldLimbs,
        modulus_inverse_word: u32,
        mut output: DisjointSlice<FieldLimbs>,
    ) {
        let lane = thread::index_1d();
        let row = lane.get();
        if row >= usize::try_from(rows).unwrap_or(0) {
            return;
        }
        let Some(start) = row_offsets.get(row).copied() else {
            return;
        };
        let Some(end) = row_offsets.get(row.saturating_add(1)).copied() else {
            return;
        };
        let mut sum = [0_u64; 4];
        let mut entry = start;
        while entry < end {
            let Some(column) = column_indices.get(usize::try_from(entry).unwrap_or(usize::MAX))
            else {
                return;
            };
            let Some(value) = values
                .get(usize::try_from(entry).unwrap_or(usize::MAX))
                .copied()
            else {
                return;
            };
            let Some(vector_value) = vector
                .get(usize::try_from(*column).unwrap_or(usize::MAX))
                .copied()
            else {
                return;
            };
            sum = field_add_with_config(
                sum,
                montgomery_mul_with_config(value, vector_value, modulus, modulus_inverse_word),
                modulus,
            );
            entry = entry.saturating_add(1);
        }
        if let Some(destination) = output.get_mut(lane) {
            *destination = sum;
        }
    }

    /// Folds two equal-length Montgomery vectors per CUDA lane.
    #[kernel]
    #[launch_bounds(256)]
    pub fn fold_vectors(
        left: &[FieldLimbs],
        right: &[FieldLimbs],
        scalar: FieldLimbs,
        count: u32,
        modulus: FieldLimbs,
        modulus_inverse_word: u32,
        mut output: DisjointSlice<FieldLimbs>,
    ) {
        let lane = thread::index_1d();
        let index = lane.get();
        if index >= usize::try_from(count).unwrap_or(0) {
            return;
        }
        let Some(left_value) = left.get(index).copied() else {
            return;
        };
        let Some(right_value) = right.get(index).copied() else {
            return;
        };
        let folded = field_add_with_config(
            left_value,
            montgomery_mul_with_config(scalar, right_value, modulus, modulus_inverse_word),
            modulus,
        );
        if let Some(destination) = output.get_mut(lane) {
            *destination = folded;
        }
    }

    /// Computes the Nova R1CS cross-term per CUDA lane.
    #[kernel]
    #[launch_bounds(256)]
    pub fn cross_term(
        az: &[FieldLimbs],
        bz: &[FieldLimbs],
        cz: &[FieldLimbs],
        error: &[FieldLimbs],
        u: FieldLimbs,
        count: u32,
        modulus: FieldLimbs,
        modulus_inverse_word: u32,
        mut output: DisjointSlice<FieldLimbs>,
    ) {
        let lane = thread::index_1d();
        let index = lane.get();
        if index >= usize::try_from(count).unwrap_or(0) {
            return;
        }
        let (Some(az_value), Some(bz_value), Some(cz_value), Some(error_value)) = (
            az.get(index).copied(),
            bz.get(index).copied(),
            cz.get(index).copied(),
            error.get(index).copied(),
        ) else {
            return;
        };
        let product = montgomery_mul_with_config(az_value, bz_value, modulus, modulus_inverse_word);
        let scaled_c = montgomery_mul_with_config(u, cz_value, modulus, modulus_inverse_word);
        let value = field_sub_with_config(
            field_sub_with_config(product, scaled_c, modulus),
            error_value,
            modulus,
        );
        if let Some(destination) = output.get_mut(lane) {
            *destination = value;
        }
    }

    /// Checks one encoded transition before evaluating its field permutation.
    fn validate_poseidon_transition(
        encoded: &[u8],
        base: usize,
        lane_index: usize,
        step_size: usize,
        ark: &[FieldLimbs],
        mds: &[FieldLimbs],
        full_rounds: u32,
        partial_rounds: u32,
        domain: FieldLimbs,
    ) -> (bool, FieldLimbs) {
        let expected_index = u64::try_from(lane_index).unwrap_or(u64::MAX);
        let actual_index = read_u64_le(encoded, base + INDEX_OFFSET).unwrap_or(u64::MAX);
        let mut is_valid = actual_index == expected_index;
        is_valid = is_valid && has_expected_previous(encoded, base, lane_index, step_size);
        let Some(data_len) = encoded.get(base + DATA_LEN_OFFSET).copied() else {
            return (false, [0_u64; 4]);
        };
        is_valid = is_valid && usize::from(data_len) <= topology_commitment::POSEIDON_RATE;
        let mut field = usize::from(data_len);
        while field < topology_commitment::POSEIDON_RATE {
            let Some(field_offset) = field.checked_mul(32) else {
                return (false, [0_u64; 4]);
            };
            is_valid = is_valid && bytes_are_zero(encoded, base + DATA_OFFSET + field_offset, 32);
            field = field.saturating_add(1);
        }
        is_valid = is_valid && bytes_are_zero(encoded, base + DATA_LEN_OFFSET + 1, 7);
        if !is_valid {
            return (false, [0_u64; 4]);
        }

        let Some(previous) = read_field(encoded, base + PREVIOUS_OFFSET) else {
            return (false, [0_u64; 4]);
        };
        let mut state = [[0_u64; 4]; 13];
        state[0] = domain;
        state[1] = to_montgomery(previous);
        let mut position = 0_usize;
        while position < POSEIDON_RATE_DEVICE {
            let Some(field_offset) = position.checked_mul(32) else {
                return (false, [0_u64; 4]);
            };
            let Some(value) = read_field(encoded, base + DATA_OFFSET + field_offset) else {
                return (false, [0_u64; 4]);
            };
            state[position + 2] = to_montgomery(value);
            position = position.saturating_add(1);
        }
        if !poseidon_permutation(&mut state, ark, mds, full_rounds, partial_rounds) {
            return (false, [0_u64; 4]);
        }
        let digest = from_montgomery(state[0]);
        (
            field_equals_bytes(encoded, base + NEXT_OFFSET, digest),
            digest,
        )
    }

    /// Applies the fixed Circom Poseidon round schedule to one state.
    pub(super) fn poseidon_permutation(
        state: &mut [[u64; 4]; 13],
        ark: &[FieldLimbs],
        mds: &[FieldLimbs],
        full_rounds: u32,
        partial_rounds: u32,
    ) -> bool {
        let round_count = match full_rounds.checked_add(partial_rounds) {
            Some(value) => value,
            None => return false,
        };
        let half_full = full_rounds / 2;
        let mut round = 0_u32;
        while round < round_count {
            let full_sbox = round < half_full || round >= half_full + partial_rounds;
            if !apply_poseidon_round(state, ark, mds, round, full_sbox) {
                return false;
            }
            round = round.saturating_add(1);
        }
        true
    }

    /// Applies round constants, the x^5 S-box, and the MDS matrix.
    fn apply_poseidon_round(
        state: &mut [[u64; 4]; 13],
        ark: &[FieldLimbs],
        mds: &[FieldLimbs],
        round: u32,
        full_sbox: bool,
    ) -> bool {
        let Some(round_start) = usize::try_from(round)
            .ok()
            .and_then(|value| value.checked_mul(POSEIDON_WIDTH))
        else {
            return false;
        };
        let mut sboxed = [[0_u64; 4]; 13];
        let mut position = 0_usize;
        while position < POSEIDON_WIDTH {
            let Some(constant) = ark.get(round_start + position).copied() else {
                return false;
            };
            let added = field_add(state[position], constant);
            sboxed[position] = if full_sbox || position == 0 {
                field_pow5(added)
            } else {
                added
            };
            position = position.saturating_add(1);
        }
        let mut next = [[0_u64; 4]; 13];
        let mut row = 0_usize;
        while row < POSEIDON_WIDTH {
            let Some(row_start) = row.checked_mul(POSEIDON_WIDTH) else {
                return false;
            };
            let mut sum = [0_u64; 4];
            let mut column = 0_usize;
            while column < POSEIDON_WIDTH {
                let Some(coefficient) = mds.get(row_start + column).copied() else {
                    return false;
                };
                sum = field_add(sum, montgomery_mul(coefficient, sboxed[column]));
                column = column.saturating_add(1);
            }
            next[row] = sum;
            row = row.saturating_add(1);
        }
        *state = next;
        true
    }

    /// Computes x^5 using two squarings and one multiplication.
    fn field_pow5(value: FieldLimbs) -> FieldLimbs {
        let square = montgomery_mul(value, value);
        let fourth = montgomery_mul(square, square);
        montgomery_mul(fourth, value)
    }

    /// Adds two Montgomery-encoded BN254 residues modulo the scalar modulus.
    fn field_add(left: FieldLimbs, right: FieldLimbs) -> FieldLimbs {
        field_add_with_config(left, right, device_field_modulus())
    }

    /// Adds two Montgomery residues under an explicit modulus.
    fn field_add_with_config(
        left: FieldLimbs,
        right: FieldLimbs,
        modulus: FieldLimbs,
    ) -> FieldLimbs {
        let mut result = [0_u64; 4];
        let mut carry = 0_u64;
        let mut limb = 0_usize;
        while limb < FIELD_LIMBS {
            let first = left[limb].wrapping_add(right[limb]);
            let first_carry = u64::from(first < left[limb]);
            let second = first.wrapping_add(carry);
            let second_carry = u64::from(second < first);
            result[limb] = second;
            carry = u64::from(first_carry != 0 || second_carry != 0);
            limb = limb.saturating_add(1);
        }
        if carry != 0 || !field_less_than(result, modulus) {
            subtract_modulus(result, modulus)
        } else {
            result
        }
    }

    /// Subtracts two Montgomery residues under an explicit modulus.
    fn field_sub_with_config(
        left: FieldLimbs,
        right: FieldLimbs,
        modulus: FieldLimbs,
    ) -> FieldLimbs {
        let mut result = [0_u64; 4];
        let mut borrow = 0_u64;
        let mut limb = 0_usize;
        while limb < FIELD_LIMBS {
            let subtrahend = right[limb].wrapping_add(borrow);
            result[limb] = left[limb].wrapping_sub(subtrahend);
            borrow = u64::from(left[limb] < subtrahend);
            limb = limb.saturating_add(1);
        }
        if borrow != 0 {
            let mut corrected = [0_u64; 4];
            let mut carry = 0_u64;
            let mut index = 0_usize;
            while index < FIELD_LIMBS {
                let first = result[index].wrapping_add(modulus[index]);
                let first_carry = u64::from(first < result[index]);
                let second = first.wrapping_add(carry);
                corrected[index] = second;
                carry = u64::from(first_carry != 0 || second < first);
                index = index.saturating_add(1);
            }
            corrected
        } else {
            result
        }
    }

    /// Subtracts the modulus from a value known to be at least the modulus.
    fn subtract_modulus(value: FieldLimbs, modulus: FieldLimbs) -> FieldLimbs {
        let mut result = [0_u64; 4];
        let mut borrow = 0_u64;
        let mut limb = 0_usize;
        while limb < FIELD_LIMBS {
            let subtrahend = modulus[limb].wrapping_add(borrow);
            let difference = value[limb].wrapping_sub(subtrahend);
            result[limb] = difference;
            borrow = u64::from(value[limb] < subtrahend);
            limb = limb.saturating_add(1);
        }
        result
    }

    /// Returns whether one four-limb value is strictly below another.
    fn field_less_than(left: FieldLimbs, right: FieldLimbs) -> bool {
        let mut limb = FIELD_LIMBS;
        while limb > 0 {
            limb = limb.saturating_sub(1);
            if left[limb] != right[limb] {
                return left[limb] < right[limb];
            }
        }
        false
    }

    /// Decodes one canonical ordinary field element from the transcript.
    fn read_field(encoded: &[u8], offset: usize) -> Option<FieldLimbs> {
        let mut value = [0_u64; 4];
        let mut limb = 0_usize;
        while limb < FIELD_LIMBS {
            value[limb] = read_u64_le(encoded, offset.checked_add(limb.checked_mul(8)?)?)?;
            limb = limb.saturating_add(1);
        }
        field_less_than(value, device_field_modulus()).then_some(value)
    }

    /// Compares one ordinary field element with its little-endian bytes.
    fn field_equals_bytes(encoded: &[u8], offset: usize, value: FieldLimbs) -> bool {
        let mut limb = 0_usize;
        while limb < FIELD_LIMBS {
            let Some(encoded_limb) = read_u64_le(
                encoded,
                offset
                    .checked_add(limb.checked_mul(8).unwrap_or(usize::MAX))
                    .unwrap_or(usize::MAX),
            ) else {
                return false;
            };
            if encoded_limb != value[limb] {
                return false;
            }
            limb = limb.saturating_add(1);
        }
        true
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
        context.bind_to_thread()?;
        let stream = context.default_stream();
        let parameters = prepare_poseidon_parameters()?;
        let poseidon_ark = DeviceBuffer::from_host(&stream, &parameters.ark)?;
        let poseidon_mds = DeviceBuffer::from_host(&stream, &parameters.mds)?;
        stream.synchronize()?;
        Ok(Self {
            context,
            module,
            poseidon_ark,
            poseidon_mds,
            poseidon_full_rounds: parameters.full_rounds,
            poseidon_partial_rounds: parameters.partial_rounds,
            poseidon_domain: parameters.domain,
            execution_workspace: Mutex::new(CudaWorkspace::default()),
            spmv_calls: AtomicU64::new(0),
            spmv_batches: AtomicU64::new(0),
            fold_calls: AtomicU64::new(0),
            cross_term_calls: AtomicU64::new(0),
            spmv_microseconds: AtomicU64::new(0),
            fold_microseconds: AtomicU64::new(0),
            cross_term_microseconds: AtomicU64::new(0),
            csr_cache_hits: AtomicU64::new(0),
            csr_cache_misses: AtomicU64::new(0),
            output_cache_hits: AtomicU64::new(0),
            output_cache_misses: AtomicU64::new(0),
            device_buffer_allocations: AtomicU64::new(0),
            stream_synchronizations: AtomicU64::new(0),
        })
    }

    /// Returns successful arithmetic work and cache activity since creation.
    pub(crate) fn backend_stats(&self) -> GpuBackendStats {
        GpuBackendStats {
            spmv_calls: self.spmv_calls.load(Ordering::Relaxed),
            spmv_batches: self.spmv_batches.load(Ordering::Relaxed),
            fold_calls: self.fold_calls.load(Ordering::Relaxed),
            cross_term_calls: self.cross_term_calls.load(Ordering::Relaxed),
            spmv_microseconds: self.spmv_microseconds.load(Ordering::Relaxed),
            fold_microseconds: self.fold_microseconds.load(Ordering::Relaxed),
            cross_term_microseconds: self.cross_term_microseconds.load(Ordering::Relaxed),
            csr_cache_hits: self.csr_cache_hits.load(Ordering::Relaxed),
            csr_cache_misses: self.csr_cache_misses.load(Ordering::Relaxed),
            output_cache_hits: self.output_cache_hits.load(Ordering::Relaxed),
            output_cache_misses: self.output_cache_misses.load(Ordering::Relaxed),
            device_buffer_allocations: self.device_buffer_allocations.load(Ordering::Relaxed),
            stream_synchronizations: self.stream_synchronizations.load(Ordering::Relaxed),
        }
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
        let _workspace = self
            .execution_workspace
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
        if let Some(index) = first_invalid_lane(&valid) {
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

    /// Uploads one transcript and validates its complete Poseidon transition.
    pub(crate) fn validate_poseidon_steps(
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
        let _workspace = self
            .execution_workspace
            .lock()
            .map_err(|_| CudaNovaError::RuntimeStatePoisoned)?;
        self.context.bind_to_thread()?;
        let stream = self.context.default_stream();
        let upload_start = Instant::now();
        let encoded_device = DeviceBuffer::from_host(&stream, &encoded)?;
        let mut valid_device = DeviceBuffer::<u32>::zeroed(&stream, steps.len())?;
        let mut digests_device = DeviceBuffer::<FieldLimbs>::zeroed(&stream, steps.len())?;
        stream.synchronize()?;
        let upload_microseconds = duration_microseconds(upload_start.elapsed());
        let launch = launch_config(steps.len())?;
        let kernel_start = Instant::now();
        self.module.validate_poseidon_steps(
            &stream,
            launch,
            &encoded_device,
            step_size,
            step_count,
            &self.poseidon_ark,
            &self.poseidon_mds,
            self.poseidon_full_rounds,
            self.poseidon_partial_rounds,
            self.poseidon_domain,
            &mut valid_device,
            &mut digests_device,
        )?;
        stream.synchronize()?;
        let kernel_microseconds = duration_microseconds(kernel_start.elapsed());
        let download_start = Instant::now();
        let valid = valid_device.to_host_vec(&stream)?;
        let digests = digests_device.to_host_vec(&stream)?;
        let download_microseconds = duration_microseconds(download_start.elapsed());
        if let Some(index) = first_invalid_lane(&valid) {
            if let Some(mismatch) = find_poseidon_mismatch(&valid, &digests, steps) {
                return Err(mismatch);
            }
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

    /// Uploads one transcript, launches its validator repeatedly, and
    /// downloads one result vector after the final launch.
    pub(crate) fn benchmark_steps(
        &self,
        steps: &[TopologyFoldStep],
        iterations: usize,
    ) -> Result<ResidentValidationReport, CudaNovaError> {
        if iterations == 0 {
            return Err(CudaNovaError::InvalidTranscript {
                proposition: "benchmark iterations are positive",
            });
        }
        let end_to_end_start = Instant::now();
        let encoded = encode_steps(steps)?;
        let step_count = u32::try_from(steps.len()).map_err(|_| CudaNovaError::SizeOverflow {
            target: "CUDA step count",
        })?;
        let step_size =
            u32::try_from(ENCODED_STEP_BYTES).map_err(|_| CudaNovaError::SizeOverflow {
                target: "CUDA encoded step size",
            })?;
        let _workspace = self
            .execution_workspace
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
        for _ in 0..iterations {
            self.module.validate_topology_steps(
                &stream,
                launch,
                &encoded_device,
                step_size,
                step_count,
                &mut valid_device,
            )?;
        }
        stream.synchronize()?;
        let kernel_microseconds = duration_microseconds(kernel_start.elapsed());
        let download_start = Instant::now();
        let valid = valid_device.to_host_vec(&stream)?;
        let download_microseconds = duration_microseconds(download_start.elapsed());
        if let Some(index) = first_invalid_lane(&valid) {
            return Err(CudaNovaError::GpuValidationFailed { index });
        }
        Ok(ResidentValidationReport {
            steps: steps.len(),
            iterations,
            encoded_bytes: encoded.len(),
            upload_microseconds,
            kernel_microseconds,
            download_microseconds,
            end_to_end_microseconds: duration_microseconds(end_to_end_start.elapsed()),
        })
    }

    /// Evaluates one CSR matrix-vector product on the selected device.
    pub(crate) fn spmv(
        &self,
        row_offsets: &[usize],
        column_indices: &[usize],
        values: &[u8],
        vector: &[u8],
        field_width: usize,
        modulus: &[u8],
    ) -> Option<Vec<u8>> {
        let start = Instant::now();
        let result = self
            .run_spmv(
                row_offsets,
                column_indices,
                values,
                vector,
                field_width,
                modulus,
            )
            .ok()?;
        self.record_arithmetic_run(
            &result,
            duration_microseconds(start.elapsed()),
            &self.spmv_calls,
            &self.spmv_microseconds,
            1,
        );
        if result.launched {
            self.spmv_batches.fetch_add(1, Ordering::Relaxed);
        }
        Some(result.output)
    }

    /// Evaluates the Nova A, B, and C products in one device batch.
    pub(crate) fn spmv_three(
        &self,
        row_offsets: [&[usize]; 3],
        column_indices: [&[usize]; 3],
        values: [&[u8]; 3],
        vector: &[u8],
        field_width: usize,
        modulus: &[u8],
    ) -> Option<[Vec<u8>; 3]> {
        let start = Instant::now();
        let result = self
            .run_spmv_three(
                row_offsets,
                column_indices,
                values,
                vector,
                field_width,
                modulus,
            )
            .ok()?;
        self.record_arithmetic_run(
            &result,
            duration_microseconds(start.elapsed()),
            &self.spmv_calls,
            &self.spmv_microseconds,
            3,
        );
        if result.launched {
            self.spmv_batches.fetch_add(1, Ordering::Relaxed);
        }
        Some(result.output)
    }

    /// Folds two field vectors on the selected device.
    pub(crate) fn vector_linear_combination(
        &self,
        left: &[u8],
        right: &[u8],
        scalar: &[u8],
        field_width: usize,
        modulus: &[u8],
    ) -> Option<Vec<u8>> {
        let start = Instant::now();
        let result = self
            .run_vector_linear_combination(left, right, scalar, field_width, modulus)
            .ok()?;
        self.record_arithmetic_run(
            &result,
            duration_microseconds(start.elapsed()),
            &self.fold_calls,
            &self.fold_microseconds,
            1,
        );
        Some(result.output)
    }

    /// Evaluates the Nova cross-term on the selected device.
    pub(crate) fn cross_term(
        &self,
        az: &[u8],
        bz: &[u8],
        cz: &[u8],
        error: &[u8],
        u: &[u8],
        field_width: usize,
        modulus: &[u8],
    ) -> Option<Vec<u8>> {
        let start = Instant::now();
        let result = self
            .run_cross_term(az, bz, cz, error, u, field_width, modulus)
            .ok()?;
        self.record_arithmetic_run(
            &result,
            duration_microseconds(start.elapsed()),
            &self.cross_term_calls,
            &self.cross_term_microseconds,
            1,
        );
        Some(result.output)
    }

    /// Adds one successful launch's time, cache, allocation, and sync counters.
    fn record_arithmetic_run<T>(
        &self,
        run: &ArithmeticRun<T>,
        microseconds: u64,
        call_counter: &AtomicU64,
        time_counter: &AtomicU64,
        logical_calls: u64,
    ) {
        if !run.launched {
            return;
        }
        call_counter.fetch_add(logical_calls, Ordering::Relaxed);
        time_counter.fetch_add(microseconds, Ordering::Relaxed);
        self.device_buffer_allocations
            .fetch_add(run.device_buffer_allocations, Ordering::Relaxed);
        self.stream_synchronizations
            .fetch_add(run.stream_synchronizations, Ordering::Relaxed);
        if let Some(hit) = run.csr_cache_hit {
            let counter = if hit {
                &self.csr_cache_hits
            } else {
                &self.csr_cache_misses
            };
            counter.fetch_add(run.csr_cache_operations, Ordering::Relaxed);
        }
        if let Some(hit) = run.output_cache_hit {
            let counter = if hit {
                &self.output_cache_hits
            } else {
                &self.output_cache_misses
            };
            counter.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Executes the GPU CSR matrix-vector kernel and converts its result back.
    fn run_spmv(
        &self,
        row_offsets: &[usize],
        column_indices: &[usize],
        values: &[u8],
        vector: &[u8],
        field_width: usize,
        modulus: &[u8],
    ) -> Result<ArithmeticRun<Vec<u8>>, CudaNovaError> {
        let config = field_config(modulus, field_width)?;
        let packed_vector = packed_to_montgomery(vector, field_width, &config)?;
        let input = CsrMatrixInput {
            row_offsets,
            column_indices,
            values,
        };
        let rows = validate_csr_input(input, field_width, packed_vector.len())?;
        if rows == 0 {
            return Ok(ArithmeticRun::empty(Vec::new()));
        }
        let rows_u32 = u32::try_from(rows).map_err(|_| CudaNovaError::SizeOverflow {
            target: "CUDA CSR row count",
        })?;
        let mut workspace = self
            .execution_workspace
            .lock()
            .map_err(|_| CudaNovaError::RuntimeStatePoisoned)?;
        self.context.bind_to_thread()?;
        let stream = self.context.default_stream();
        let vector_device = DeviceBuffer::from_host(&stream, &packed_vector)?;
        let cache = workspace.csr_matrix(
            &stream,
            row_offsets,
            column_indices,
            values,
            field_width,
            modulus,
            rows,
            || packed_to_montgomery(values, field_width, &config),
        )?;
        let cache_hit = cache.hit;
        let device_buffer_allocations = cache.device_buffer_allocations + 1;
        let matrix = cache.value;
        let launch = launch_config(rows)?;
        self.module.spmv_rows(
            &stream,
            launch,
            &matrix.row_offsets_device,
            &matrix.column_indices_device,
            &matrix.values_device,
            &vector_device,
            rows_u32,
            config.modulus,
            config.modulus_inverse_word,
            &mut matrix.output_device,
        )?;
        let output = matrix.output_device.to_host_vec(&stream)?;
        Ok(ArithmeticRun {
            output: montgomery_to_packed(&output, &config)?,
            launched: true,
            device_buffer_allocations,
            csr_cache_hit: Some(cache_hit),
            csr_cache_operations: 1,
            output_cache_hit: None,
            stream_synchronizations: 1,
        })
    }

    /// Executes three same-height CSR products with one vector transfer and launch.
    fn run_spmv_three(
        &self,
        row_offsets: [&[usize]; 3],
        column_indices: [&[usize]; 3],
        values: [&[u8]; 3],
        vector: &[u8],
        field_width: usize,
        modulus: &[u8],
    ) -> Result<ArithmeticRun<[Vec<u8>; 3]>, CudaNovaError> {
        let config = field_config(modulus, field_width)?;
        let packed_vector = packed_to_montgomery(vector, field_width, &config)?;
        let [a_rows, b_rows, c_rows] = row_offsets;
        let [a_columns, b_columns, c_columns] = column_indices;
        let [a_values, b_values, c_values] = values;
        let inputs = [
            CsrMatrixInput {
                row_offsets: a_rows,
                column_indices: a_columns,
                values: a_values,
            },
            CsrMatrixInput {
                row_offsets: b_rows,
                column_indices: b_columns,
                values: b_values,
            },
            CsrMatrixInput {
                row_offsets: c_rows,
                column_indices: c_columns,
                values: c_values,
            },
        ];
        let [a_input, b_input, c_input] = inputs;
        let a_row_count = validate_csr_input(a_input, field_width, packed_vector.len())?;
        let b_row_count = validate_csr_input(b_input, field_width, packed_vector.len())?;
        let c_row_count = validate_csr_input(c_input, field_width, packed_vector.len())?;
        if a_row_count != b_row_count || a_row_count != c_row_count {
            return Err(CudaNovaError::InvalidTranscript {
                proposition: "the batched CSR matrices have equal row counts",
            });
        }
        if a_row_count == 0 {
            return Ok(ArithmeticRun {
                output: [Vec::new(), Vec::new(), Vec::new()],
                launched: false,
                device_buffer_allocations: 0,
                csr_cache_hit: None,
                csr_cache_operations: 0,
                output_cache_hit: None,
                stream_synchronizations: 0,
            });
        }
        let total_rows = a_row_count
            .checked_mul(3)
            .ok_or(CudaNovaError::SizeOverflow {
                target: "three-matrix CUDA launch size",
            })?;
        let total_rows_u32 =
            u32::try_from(total_rows).map_err(|_| CudaNovaError::SizeOverflow {
                target: "CUDA batched CSR row count",
            })?;
        let mut workspace = self
            .execution_workspace
            .lock()
            .map_err(|_| CudaNovaError::RuntimeStatePoisoned)?;
        self.context.bind_to_thread()?;
        let stream = self.context.default_stream();
        let vector_device = DeviceBuffer::from_host(&stream, &packed_vector)?;
        let cache =
            workspace.csr_batch(&stream, inputs, field_width, modulus, a_row_count, || {
                Ok([
                    packed_to_montgomery(a_values, field_width, &config)?,
                    packed_to_montgomery(b_values, field_width, &config)?,
                    packed_to_montgomery(c_values, field_width, &config)?,
                ])
            })?;
        let cache_hit = cache.hit;
        let device_buffer_allocations = cache.device_buffer_allocations + 1;
        let batch = cache.value;
        self.module.spmv_rows(
            &stream,
            launch_config(total_rows)?,
            &batch.row_offsets_device,
            &batch.column_indices_device,
            &batch.values_device,
            &vector_device,
            total_rows_u32,
            config.modulus,
            config.modulus_inverse_word,
            &mut batch.output_device,
        )?;
        let output = batch.output_device.to_host_vec(&stream)?;
        let mut chunks = output.chunks_exact(a_row_count);
        let packed_a = chunks.next().ok_or(CudaNovaError::InvalidTranscript {
            proposition: "the fused CSR output contains the A rows",
        })?;
        let packed_b = chunks.next().ok_or(CudaNovaError::InvalidTranscript {
            proposition: "the fused CSR output contains the B rows",
        })?;
        let packed_c = chunks.next().ok_or(CudaNovaError::InvalidTranscript {
            proposition: "the fused CSR output contains the C rows",
        })?;
        if !chunks.remainder().is_empty() || chunks.next().is_some() {
            return Err(CudaNovaError::InvalidTranscript {
                proposition: "the fused CSR output contains exactly three row ranges",
            });
        }
        Ok(ArithmeticRun {
            output: [
                montgomery_to_packed(packed_a, &config)?,
                montgomery_to_packed(packed_b, &config)?,
                montgomery_to_packed(packed_c, &config)?,
            ],
            launched: true,
            device_buffer_allocations,
            csr_cache_hit: Some(cache_hit),
            csr_cache_operations: 3,
            output_cache_hit: None,
            stream_synchronizations: 1,
        })
    }

    /// Executes the GPU vector-fold kernel and converts its result back.
    fn run_vector_linear_combination(
        &self,
        left: &[u8],
        right: &[u8],
        scalar: &[u8],
        field_width: usize,
        modulus: &[u8],
    ) -> Result<ArithmeticRun<Vec<u8>>, CudaNovaError> {
        let config = field_config(modulus, field_width)?;
        let left = packed_to_montgomery(left, field_width, &config)?;
        let right = packed_to_montgomery(right, field_width, &config)?;
        if left.len() != right.len() {
            return Err(CudaNovaError::InvalidTranscript {
                proposition: "fold vectors have equal length",
            });
        }
        let scalar = packed_to_montgomery(scalar, field_width, &config)?;
        let scalar = scalar.first().copied().ok_or(CudaNovaError::EmptyTrace)?;
        if left.is_empty() {
            return Ok(ArithmeticRun::empty(Vec::new()));
        }
        let count = u32::try_from(left.len()).map_err(|_| CudaNovaError::SizeOverflow {
            target: "CUDA fold vector length",
        })?;
        let (output, output_cache_hit, device_buffer_allocations) =
            self.run_fold_kernel(&left, &right, scalar, count, config)?;
        Ok(ArithmeticRun {
            output: montgomery_to_packed(&output, &config)?,
            launched: true,
            device_buffer_allocations,
            csr_cache_hit: None,
            csr_cache_operations: 0,
            output_cache_hit: Some(output_cache_hit),
            stream_synchronizations: 1,
        })
    }

    /// Executes the GPU cross-term kernel and converts its result back.
    fn run_cross_term(
        &self,
        az: &[u8],
        bz: &[u8],
        cz: &[u8],
        error: &[u8],
        u: &[u8],
        field_width: usize,
        modulus: &[u8],
    ) -> Result<ArithmeticRun<Vec<u8>>, CudaNovaError> {
        let config = field_config(modulus, field_width)?;
        let az = packed_to_montgomery(az, field_width, &config)?;
        let bz = packed_to_montgomery(bz, field_width, &config)?;
        let cz = packed_to_montgomery(cz, field_width, &config)?;
        let error = packed_to_montgomery(error, field_width, &config)?;
        let scalar = packed_to_montgomery(u, field_width, &config)?;
        let scalar = scalar.first().copied().ok_or(CudaNovaError::EmptyTrace)?;
        if az.len() != bz.len() || az.len() != cz.len() || az.len() != error.len() {
            return Err(CudaNovaError::InvalidTranscript {
                proposition: "cross-term vectors have equal length",
            });
        }
        if az.is_empty() {
            return Ok(ArithmeticRun::empty(Vec::new()));
        }
        let count = u32::try_from(az.len()).map_err(|_| CudaNovaError::SizeOverflow {
            target: "CUDA cross-term vector length",
        })?;
        let mut workspace = self
            .execution_workspace
            .lock()
            .map_err(|_| CudaNovaError::RuntimeStatePoisoned)?;
        self.context.bind_to_thread()?;
        let stream = self.context.default_stream();
        let az_device = DeviceBuffer::from_host(&stream, &az)?;
        let bz_device = DeviceBuffer::from_host(&stream, &bz)?;
        let cz_device = DeviceBuffer::from_host(&stream, &cz)?;
        let error_device = DeviceBuffer::from_host(&stream, &error)?;
        let output = workspace.output(&stream, az.len())?;
        let output_cache_hit = output.hit;
        let device_buffer_allocations = output.device_buffer_allocations + 4;
        let output_device = output.value;
        self.module.cross_term(
            &stream,
            launch_config(az.len())?,
            &az_device,
            &bz_device,
            &cz_device,
            &error_device,
            scalar,
            count,
            config.modulus,
            config.modulus_inverse_word,
            output_device,
        )?;
        let output = output_device.to_host_vec(&stream)?;
        Ok(ArithmeticRun {
            output: montgomery_to_packed(&output, &config)?,
            launched: true,
            device_buffer_allocations,
            csr_cache_hit: None,
            csr_cache_operations: 0,
            output_cache_hit: Some(output_cache_hit),
            stream_synchronizations: 1,
        })
    }

    /// Runs one vector-fold kernel while the caller owns the runtime lock.
    fn run_fold_kernel(
        &self,
        left: &[FieldLimbs],
        right: &[FieldLimbs],
        scalar: FieldLimbs,
        count: u32,
        config: DeviceFieldConfig,
    ) -> Result<(Vec<FieldLimbs>, bool, u64), CudaNovaError> {
        if left.is_empty() {
            return Ok((Vec::new(), false, 0));
        }
        let mut workspace = self
            .execution_workspace
            .lock()
            .map_err(|_| CudaNovaError::RuntimeStatePoisoned)?;
        self.context.bind_to_thread()?;
        let stream = self.context.default_stream();
        let left_device = DeviceBuffer::from_host(&stream, left)?;
        let right_device = DeviceBuffer::from_host(&stream, right)?;
        let output = workspace.output(&stream, left.len())?;
        let output_cache_hit = output.hit;
        let device_buffer_allocations = output.device_buffer_allocations + 2;
        let output_device = output.value;
        self.module.fold_vectors(
            &stream,
            launch_config(left.len())?,
            &left_device,
            &right_device,
            scalar,
            count,
            config.modulus,
            config.modulus_inverse_word,
            output_device,
        )?;
        Ok((
            output_device.to_host_vec(&stream)?,
            output_cache_hit,
            device_buffer_allocations,
        ))
    }
}

/// Validates one canonical CSR input and returns its number of rows.
fn validate_csr_input(
    input: CsrMatrixInput<'_>,
    field_width: usize,
    vector_len: usize,
) -> Result<usize, CudaNovaError> {
    let rows = input
        .row_offsets
        .len()
        .checked_sub(1)
        .ok_or(CudaNovaError::SizeOverflow {
            target: "CSR row count",
        })?;
    let values_count = input.values.len() / field_width;
    if input.values.len() % field_width != 0
        || values_count != input.column_indices.len()
        || input.row_offsets.last().copied() != Some(values_count)
    {
        return Err(CudaNovaError::InvalidTranscript {
            proposition: "CSR matrix buffers have compatible lengths",
        });
    }
    let offsets_are_canonical = input.row_offsets.first().copied() == Some(0)
        && input.row_offsets.windows(2).all(|pair| {
            let [left, right] = pair else {
                return false;
            };
            left <= right && *right <= values_count
        });
    let columns_are_in_bounds = input
        .column_indices
        .iter()
        .all(|column| *column < vector_len);
    if !offsets_are_canonical || !columns_are_in_bounds {
        return Err(CudaNovaError::InvalidTranscript {
            proposition: "CSR rows are canonical and columns address the dense vector",
        });
    }
    Ok(rows)
}

/// Converts the canonical light-Poseidon table into device-resident limbs.
fn prepare_poseidon_parameters() -> Result<PoseidonHostParameters, CudaNovaError> {
    let width =
        u8::try_from(POSEIDON_INPUTS + 1).map_err(|_| CudaNovaError::PoseidonParameters {
            message: "Poseidon width does not fit the parameter API".to_owned(),
        })?;
    let raw = get_poseidon_parameters::<Fr>(width).map_err(|error| {
        CudaNovaError::PoseidonParameters {
            message: error.to_string(),
        }
    })?;
    let expected_width = POSEIDON_INPUTS + 1;
    if raw.ark.len() % expected_width != 0 {
        return Err(CudaNovaError::PoseidonParameters {
            message: "round constants are not divisible by the state width".to_owned(),
        });
    }
    let ark = raw
        .ark
        .iter()
        .copied()
        .map(field_to_montgomery)
        .collect::<Result<Vec<_>, _>>()?;
    let mds_rows_are_valid = raw.mds.iter().all(|row| row.len() == expected_width);
    if raw.mds.len() != expected_width || !mds_rows_are_valid {
        return Err(CudaNovaError::PoseidonParameters {
            message: "MDS matrix does not have width-thirteen shape".to_owned(),
        });
    }
    let mds = raw
        .mds
        .iter()
        .flat_map(|row| row.iter().copied())
        .map(field_to_montgomery)
        .collect::<Result<Vec<_>, _>>()?;
    let full_rounds =
        u32::try_from(raw.full_rounds).map_err(|_| CudaNovaError::PoseidonParameters {
            message: "full round count does not fit the device ABI".to_owned(),
        })?;
    let partial_rounds =
        u32::try_from(raw.partial_rounds).map_err(|_| CudaNovaError::PoseidonParameters {
            message: "partial round count does not fit the device ABI".to_owned(),
        })?;
    let domain = field_to_montgomery(Fr::from_le_bytes_mod_order(TOPOLOGY_DOMAIN))?;
    Ok(PoseidonHostParameters {
        ark,
        mds,
        full_rounds,
        partial_rounds,
        domain,
    })
}

/// Selects one supported Nova cycle field from its canonical modulus bytes.
fn field_config(modulus: &[u8], field_width: usize) -> Result<DeviceFieldConfig, CudaNovaError> {
    if field_width != COMMITMENT_BYTES || modulus.len() != COMMITMENT_BYTES {
        return Err(CudaNovaError::InvalidTranscript {
            proposition: "the CUDA backend receives a 32-byte Nova field",
        });
    }
    let candidate = bytes_to_limbs(modulus).ok_or(CudaNovaError::InvalidTranscript {
        proposition: "the Nova field modulus has four 64-bit limbs",
    })?;
    if candidate == PRIMARY_FIELD_CONFIG.modulus {
        Ok(PRIMARY_FIELD_CONFIG)
    } else if candidate == SECONDARY_FIELD_CONFIG.modulus {
        Ok(SECONDARY_FIELD_CONFIG)
    } else {
        Err(CudaNovaError::InvalidTranscript {
            proposition: "the CUDA backend receives a BN254/Grumpkin cycle field",
        })
    }
}

/// Converts one packed byte slice into Montgomery limbs for a device kernel.
fn packed_to_montgomery(
    packed: &[u8],
    field_width: usize,
    config: &DeviceFieldConfig,
) -> Result<Vec<FieldLimbs>, CudaNovaError> {
    if field_width != COMMITMENT_BYTES || packed.len() % field_width != 0 {
        return Err(CudaNovaError::InvalidTranscript {
            proposition: "packed CUDA field vectors have fixed-width elements",
        });
    }
    let mut values = Vec::with_capacity(packed.len() / field_width);
    for chunk in packed.chunks_exact(field_width) {
        let value = bytes_to_limbs(chunk).ok_or(CudaNovaError::InvalidTranscript {
            proposition: "packed CUDA field vectors have complete limbs",
        })?;
        if !limbs_less_than(value, config.modulus) {
            return Err(CudaNovaError::InvalidTranscript {
                proposition: "CUDA field vectors use canonical residues",
            });
        }
        values.push(kernels::to_montgomery_with_config(
            value,
            config.modulus,
            config.modulus_inverse_word,
            config.montgomery_r2,
        ));
    }
    Ok(values)
}

/// Converts Montgomery limbs returned by a device kernel to packed bytes.
fn montgomery_to_packed(
    values: &[FieldLimbs],
    config: &DeviceFieldConfig,
) -> Result<Vec<u8>, CudaNovaError> {
    let mut packed = Vec::with_capacity(values.len().checked_mul(COMMITMENT_BYTES).ok_or(
        CudaNovaError::SizeOverflow {
            target: "packed CUDA field output size",
        },
    )?);
    for value in values {
        let canonical = kernels::from_montgomery_with_config(
            *value,
            config.modulus,
            config.modulus_inverse_word,
        );
        packed.extend_from_slice(&limbs_to_bytes(canonical));
    }
    Ok(packed)
}

/// Decodes a little-endian byte slice into four 64-bit limbs.
fn bytes_to_limbs(bytes: &[u8]) -> Option<FieldLimbs> {
    if bytes.len() != COMMITMENT_BYTES {
        return None;
    }
    let mut limbs = [0_u64; FIELD_LIMBS];
    for (limb, chunk) in limbs.iter_mut().zip(bytes.chunks_exact(8)) {
        let chunk = <[u8; 8]>::try_from(chunk).ok()?;
        *limb = u64::from_le_bytes(chunk);
    }
    Some(limbs)
}

/// Returns whether one four-limb value is strictly below another.
fn limbs_less_than(left: FieldLimbs, right: FieldLimbs) -> bool {
    let mut limb = FIELD_LIMBS;
    while limb > 0 {
        limb = limb.saturating_sub(1);
        if left[limb] != right[limb] {
            return left[limb] < right[limb];
        }
    }
    false
}

/// Serializes four little-endian limbs into the canonical 32-byte layout.
fn limbs_to_bytes(value: FieldLimbs) -> [u8; COMMITMENT_BYTES] {
    let mut bytes = [0_u8; COMMITMENT_BYTES];
    for (limb, chunk) in value.iter().zip(bytes.chunks_exact_mut(8)) {
        chunk.copy_from_slice(&limb.to_le_bytes());
    }
    bytes
}

/// Serializes an arkworks scalar into four little-endian device limbs.
fn field_to_limbs(value: Fr) -> Result<FieldLimbs, CudaNovaError> {
    let bytes = value.into_bigint().to_bytes_le();
    let bytes = <[u8; COMMITMENT_BYTES]>::try_from(bytes.as_slice()).map_err(|_| {
        CudaNovaError::PoseidonParameters {
            message: "arkworks field serialization is not 32 bytes".to_owned(),
        }
    })?;
    let mut limbs = [0_u64; FIELD_LIMBS];
    for (limb, chunk) in limbs.iter_mut().zip(bytes.chunks_exact(8)) {
        let chunk = <[u8; 8]>::try_from(chunk).map_err(|_| CudaNovaError::PoseidonParameters {
            message: "field limb serialization is not eight bytes".to_owned(),
        })?;
        *limb = u64::from_le_bytes(chunk);
    }
    Ok(limbs)
}

/// Converts an arkworks scalar into the GPU's Montgomery limb form.
fn field_to_montgomery(value: Fr) -> Result<FieldLimbs, CudaNovaError> {
    let r = Fr::from_le_bytes_mod_order(&limbs_to_bytes([
        MONTGOMERY_ONE_0,
        MONTGOMERY_ONE_1,
        MONTGOMERY_ONE_2,
        MONTGOMERY_ONE_3,
    ]));
    field_to_limbs(value * r)
}

/// Returns the first device lane that did not accept the transcript.
fn first_invalid_lane(valid: &[u32]) -> Option<usize> {
    valid
        .iter()
        .enumerate()
        .find_map(|(index, value)| (*value != 1_u32).then_some(index))
}

/// Returns a typed digest mismatch when a device lane computed a non-zero
/// digest that differs from the transcript claim.
fn find_poseidon_mismatch(
    valid: &[u32],
    digests: &[FieldLimbs],
    steps: &[TopologyFoldStep],
) -> Option<CudaNovaError> {
    for (index, value) in valid.iter().enumerate() {
        if *value == 1_u32 {
            continue;
        }
        let Some(expected) = steps.get(index).map(|step| step.next) else {
            continue;
        };
        let Some(actual) = digests.get(index).copied().map(limbs_to_bytes) else {
            continue;
        };
        if actual != [0_u8; COMMITMENT_BYTES] && actual != expected {
            return Some(CudaNovaError::GpuPoseidonMismatch {
                index,
                expected,
                actual,
            });
        }
    }
    None
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

#[cfg(test)]
mod tests {
    use super::*;
    use topology_commitment::commit_topology_with_trace;

    /// Checks that the device Montgomery product agrees with arkworks on
    /// non-trivial and high-limb field elements.
    #[test]
    fn field_product_matches_arkworks() {
        let cases = [
            (Fr::from(17_u64), Fr::from(29_u64)),
            (
                Fr::from_le_bytes_mod_order(&[0xff_u8; COMMITMENT_BYTES]),
                Fr::from_le_bytes_mod_order(&[0xa5_u8; COMMITMENT_BYTES]),
            ),
        ];
        for (left, right) in cases {
            let left_montgomery = field_to_montgomery(left).unwrap_or([0; 4]);
            let right_montgomery = field_to_montgomery(right).unwrap_or([0; 4]);
            let product_montgomery = kernels::montgomery_mul(left_montgomery, right_montgomery);
            let product = kernels::montgomery_mul(product_montgomery, [1_u64, 0, 0, 0]);
            let expected = field_to_limbs(left * right).unwrap_or([0; 4]);
            assert_eq!(product, expected);
        }
    }

    /// Checks the host execution of the exact device permutation against the
    /// canonical commitment trace before PTX code generation is involved.
    #[test]
    fn device_poseidon_permutation_matches_trace_on_host() {
        let mut steps = Vec::new();
        let result = commit_topology_with_trace(2, &[0, 1, 2], &[0, 1], |step| {
            steps.push(step);
        });
        assert!(result.is_ok());
        let Some(step) = steps.first() else {
            return;
        };
        let parameters = prepare_poseidon_parameters();
        assert!(parameters.is_ok());
        let Ok(parameters) = parameters else {
            return;
        };
        let mut state = [[0_u64; 4]; 13];
        state[0] = parameters.domain;
        let previous = Fr::from_le_bytes_mod_order(&step.previous);
        state[1] = field_to_montgomery(previous).unwrap_or([0; 4]);
        for (position, bytes) in step.data.iter().enumerate() {
            let value = Fr::from_le_bytes_mod_order(bytes);
            state[position + 2] = field_to_montgomery(value).unwrap_or([0; 4]);
        }
        let computed = kernels::poseidon_permutation(
            &mut state,
            &parameters.ark,
            &parameters.mds,
            parameters.full_rounds,
            parameters.partial_rounds,
        );
        assert!(computed);
        let digest = kernels::montgomery_mul(state[0], [1_u64, 0, 0, 0]);
        assert_eq!(limbs_to_bytes(digest), step.next);
    }
}

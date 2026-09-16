//! Adapter from the cuda-nova runtime to Nova's arithmetic backend hooks.

use std::sync::Arc;

use nova_snark::provider::gpu::GpuBackend;

use crate::cuda_runtime::CudaRuntime;

/// Synchronous Nova backend that dispatches field operations to one CUDA
/// context. The runtime serializes launches because cuda-oxide's typed module
/// owns the stream and device buffers for this MVP.
pub(crate) struct CudaGpuBackend {
    /// Device runtime shared with the public engine handle.
    runtime: Arc<CudaRuntime>,
}

impl CudaGpuBackend {
    /// Creates an adapter around an initialized CUDA runtime.
    pub(crate) const fn new(runtime: Arc<CudaRuntime>) -> Self {
        Self { runtime }
    }
}

impl GpuBackend for CudaGpuBackend {
    fn spmv(
        &self,
        row_offsets: &[usize],
        column_indices: &[usize],
        values: &[u8],
        vector: &[u8],
        field_width: usize,
        modulus: &[u8],
    ) -> Option<Vec<u8>> {
        self.runtime.spmv(
            row_offsets,
            column_indices,
            values,
            vector,
            field_width,
            modulus,
        )
    }

    fn spmv_three(
        &self,
        row_offsets: [&[usize]; 3],
        column_indices: [&[usize]; 3],
        values: [&[u8]; 3],
        vector: &[u8],
        field_width: usize,
        modulus: &[u8],
    ) -> Option<[Vec<u8>; 3]> {
        self.runtime.spmv_three(
            row_offsets,
            column_indices,
            values,
            vector,
            field_width,
            modulus,
        )
    }

    fn vector_linear_combination(
        &self,
        left: &[u8],
        right: &[u8],
        scalar: &[u8],
        field_width: usize,
        modulus: &[u8],
    ) -> Option<Vec<u8>> {
        self.runtime
            .vector_linear_combination(left, right, scalar, field_width, modulus)
    }

    fn cross_term(
        &self,
        az: &[u8],
        bz: &[u8],
        cz: &[u8],
        error: &[u8],
        u: &[u8],
        field_width: usize,
        modulus: &[u8],
    ) -> Option<Vec<u8>> {
        self.runtime
            .cross_term(az, bz, cz, error, u, field_width, modulus)
    }
}

/// Exercises the official Nova Blitzar path used by BN254 commitments.
#[cfg(all(
    feature = "cuda",
    feature = "gpu-msm",
    target_os = "linux",
    target_arch = "x86_64"
))]
pub(crate) fn msm_self_test() -> bool {
    use halo2curves::{
        bn256::{Fr, G1, G1Affine},
        group::Group,
    };

    let scalars = vec![Fr::from(3_u64), Fr::from(5_u64), Fr::from(7_u64)];
    let bases = vec![
        G1Affine::from(G1::generator()),
        G1Affine::from(G1::generator() * Fr::from(11_u64)),
        G1Affine::from(G1::generator() * Fr::from(13_u64)),
    ];
    let expected = bases
        .iter()
        .zip(scalars.iter())
        .fold(G1::identity(), |acc, (base, scalar)| acc + *base * *scalar);
    let actual = nova_snark::provider::blitzar::vartime_multiscalar_mul(&scalars, &bases);
    actual == expected
}

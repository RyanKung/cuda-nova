# cuda-nova

This crate is the CUDA sidecar for the official `nova-snark` topology prover.
It is kept inside `zkfly` while the API is measured, but its dependency and
runtime boundary is designed for extraction into a standalone public
repository.

The current engine has three explicit stages:

1. encode `TopologyFoldStep` values into a stable 432-byte record;
2. upload and preflight canonical indices, accumulator links, padding, and the
   BN254/Circom Poseidon transition on a CUDA device with cuda-oxide;
   `benchmark_steps` keeps the transcript buffers resident while repeating the
   shape kernel to expose steady-state cost;
3. install Nova's arithmetic backend and call `zkfly-nova`, which uses the
   official `nova-snark` R1CS synthesis and recursive folding implementation.
   The engine also exposes `prove_weighted_forward` for the bounded CSR
   weighted-forward fixture: topology positions are fixed, while weights and
   vectors stay private and input/output Poseidon commitments are public.

Bellpepper constraint construction, Nova transcript control, and MSM remain
host-orchestrated. During the official Nova proof, the patched backend sends
the arithmetic-heavy A/B/C CSR SpMV, NIFS cross-term, and relaxed-witness vector
folds to CUDA. MSM uses Nova's normal CPU implementation by default; passing
the explicit `gpu-msm` feature enables the official Linux Blitzar provider,
with the CPU implementation still available as a fallback. No custom MSM
protocol is introduced. The GPU Poseidon path uses an exact eight-word,
32-bit-radix CIOS Montgomery product.

On the V100, generate the PTX and run the two-step smoke fixture with:

```sh
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_TOOLKIT_PATH=/usr/local/cuda
CUDA_OXIDE_TARGET=sm_70 cargo oxide run \
  --features cuda,gpu-msm --arch sm_70 --bin cuda-nova -- --prove
```

The output includes upload, repeated-kernel, download, and end-to-end
microsecond measurements, plus a Poseidon preflight timing. With `--prove` it
also verifies the complete official Nova proof and reports the CUDA arithmetic
launch counters. The fixed two-step topology and weighted-forward runs are
smoke baselines, not a full MaleCNS proof benchmark. The weighted-forward
fixture currently accepts at most eleven neurons because its vector commitment
fits one Poseidon rate.

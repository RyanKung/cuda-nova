# cuda-nova

This crate is the CUDA sidecar for the official `nova-snark` topology prover.
Its dependency and runtime boundary is designed for extraction into a
standalone public repository.

The current engine has three explicit stages:

1. encode `TopologyFoldStep` values into a stable 432-byte record;
2. upload and preflight canonical indices, accumulator links, padding, and the
   BN254/Circom Poseidon transition on a CUDA device with cuda-oxide;
   `benchmark_steps` keeps the transcript buffers resident while repeating the
   shape kernel to expose steady-state cost;
3. install Nova's arithmetic backend and call the topology proof adapter, which
   uses the official `nova-snark` R1CS synthesis and recursive folding
   implementation.
   The engine also exposes `prove_weighted_forward` for the bounded CSR
weighted-forward fixture: topology positions are fixed, while weights and
vectors stay private and input/output Poseidon commitments are public.
`setup_weighted_forward` plus
`prove_weighted_forward_with_parameters` lets callers reuse Nova setup across
multiple proofs with the same topology and circuit shape.

The default build uses Nova's BN254 Pedersen/IPA primary commitment engine.
Enable the explicit `hyperkzg` feature to select Nova's official BN254
`HyperKZG` engine. A production HyperKZG run must pass `--ptau-dir` (or call
the corresponding `*_with_ptau_dir` API) with a directory containing trusted
pruned Powers-of-Tau files such as `ppot_pruned_XX.ptau`. The regular setup
entry points reject production HyperKZG setup without that directory. The
feature's unit tests use Nova `test-utils` random parameters only for regression
tests and must not be reused in production.

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

With HyperKZG and a trusted SRS directory:

```sh
CUDA_OXIDE_TARGET=sm_70 cargo oxide run \
  --features cuda,hyperkzg --arch sm_70 --bin cuda-nova -- \
  --prove --ptau-dir ./ptau_files
```

The HyperKZG feature selects Nova's primary commitment backend; it does not
change Nova's recursive protocol or turn this sidecar into HyperNova. CUDA
still handles the registered arithmetic-heavy SpMV, cross-term, and fold
paths, while Bellpepper synthesis, transcript orchestration, and MSM follow the
documented host boundary.

For callers that need commitments outside the recursive proof, the topology
proof adapter exposes `VectorCommitmentBackend` and
`HyperKzgVectorParameters`. The API supports dense vectors, canonical sparse
matrix rows, and batch multilinear openings at a common point. It pads vectors
to a power-of-two capacity and checks opening witnesses against their
commitments. This standalone evaluation path is CPU-hosted in the current
version; it is intentionally separate from the CUDA arithmetic kernels.

The output includes upload, repeated-kernel, download, and end-to-end
microsecond measurements, plus a Poseidon preflight timing. With `--prove` it
also verifies the complete official Nova proof and reports the CUDA arithmetic
launch counters. The fixed two-step topology and weighted-forward runs are
smoke baselines, not a full MaleCNS proof benchmark. Weighted-forward input and
output commitments are folded in fixed Poseidon-rate chunks; the runner's
fixture remains intentionally small for repeatable timing.

For a reusable-parameter timing curve, run the same command with
`--profile-forward` instead of `--prove`. It performs one setup and measures
1, 2, 8, and 16 recursive weighted-forward steps using that parameter set,
then proves a 12-neuron two-chunk commitment case.

## License

This crate is licensed under the GNU General Public License, version 3.0
only. See [LICENSE](LICENSE) for the complete text.

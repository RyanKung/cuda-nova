# cuda-nova

This crate connects CUDA Rust kernels compiled by
[NVIDIA Research cuda-oxide](https://github.com/NVlabs/cuda-oxide) to
[Microsoft Research Nova](https://github.com/microsoft/Nova), consumed as the
`nova-snark` crate. It is a GPU arithmetic sidecar for Nova, not a new proof
system or a HyperNova implementation. The repository is a standalone Cargo
workspace: its zkfly relation crates and patched Nova source are exact-revision
Git dependencies, so consumers do not need a Git submodule.

## Upstream and responsibility boundary

| Layer | Implementation | Responsibility |
| --- | --- | --- |
| proof protocol | Microsoft Nova / `nova-snark` 0.76 | R1CS synthesis, recursive folding, transcript, verification |
| GPU programming | CUDA Rust via cuda-oxide | PTX, CUDA context/stream, checked device buffers and kernel launches |
| polynomial commitment | Nova IPA or HyperKZG | primary Nova commitment engine |
| MSM | Nova CPU path or Nova's Blitzar provider | BN254 multi-scalar multiplication |
| application | `zkfly-commitment` and `zkfly-nova` | topology root, vector commitments, weighted CSR relation |

The workspace vendors Nova only to add a narrow arithmetic backend and
read-only Blitzar telemetry. The Nova relation, Fiat-Shamir transcript,
recursive proof, and verifier stay in Microsoft Nova. CUDA receives serialized
field arithmetic requests and returns their outputs to Nova.

## Implemented pipeline

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

## Feature matrix

| Feature | Effect |
| --- | --- |
| default | portable CPU Nova with BN254 Pedersen/IPA |
| `cuda` | compile and register cuda-oxide arithmetic and Poseidon kernels |
| `hyperkzg` | select Nova's BN254 HyperKZG engine and trusted PTau loader |
| `gpu-msm` | select Nova's optional Linux Blitzar MSM provider |

Features are orthogonal. The measured V100 configuration uses
`cuda,hyperkzg,gpu-msm`; omitting `gpu-msm` keeps MSM on Nova's normal CPU
provider, and omitting `hyperkzg` keeps the default IPA engine.

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

## Requirements

- Rust nightly `2026-04-03` with `rust-src`, `rustc-dev`,
  `llvm-tools-preview`, `rustfmt`, and `clippy`.
- Linux and an NVIDIA CUDA toolkit for the `cuda` feature. The recorded V100
  build uses CUDA 12.6 and `sm_70`.
- `cargo oxide` from the pinned cuda-oxide revision for CUDA Rust PTX builds.
- A trusted, sufficiently large pruned Powers-of-Tau directory for production
  `hyperkzg` setup.
- Blitzar's supported Linux/CUDA environment when `gpu-msm` is enabled.

The default feature-free and `hyperkzg` regression builds remain available on
non-Linux hosts. CUDA execution itself is Linux/NVIDIA-only.

## Run

On the V100, generate the PTX and run the two-step smoke fixture with:

```sh
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_TOOLKIT_PATH=/usr/local/cuda
CUDA_OXIDE_TARGET=sm_70 cargo oxide run \
  --features cuda,gpu-msm --arch sm_70 --bin cuda-nova -- --prove
```

With HyperKZG, Blitzar GPU MSM, and a trusted SRS directory on supported Linux
hosts:

```sh
CUDA_OXIDE_TARGET=sm_70 cargo oxide run \
  --features cuda,hyperkzg,gpu-msm --arch sm_70 --bin cuda-nova -- \
  --prove --ptau-dir ./ptau_files
```

The HyperKZG feature selects Nova's primary commitment backend, and `gpu-msm`
selects Nova's optional Linux Blitzar MSM provider. This does not change Nova's
recursive protocol or turn this sidecar into HyperNova. CUDA still handles the
registered arithmetic-heavy SpMV, cross-term, and fold paths, while Bellpepper
synthesis and transcript orchestration follow the documented host boundary.
Omit `gpu-msm` only when the normal CPU MSM provider is intended.

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

## Profiling

For the audited reusable-parameter timing curve, run the same command with
`--profile-forward` instead of `--prove`. The standard plan performs one setup
and measures 1, 8, 64, and 1,024 recursive weighted-forward steps using that
parameter set, then proves a 12-neuron two-chunk commitment case. The previous
1, 2, 8, and 16 sequence remains available as the bounded
`--profile-forward=smoke` mode. A custom positive, strictly increasing plan can
be supplied with `--profile-steps=1,4,32`; the runner rejects lengths above the
audited 1,024-step bound before allocating witnesses.

Use `--profile-json` to create a machine-readable receipt and
`--profile-revision` to bind it to the current `cuda-nova` commit:

```sh
CUDA_NOVA_REVISION=$(git rev-parse HEAD)
CUDA_OXIDE_TARGET=sm_70 cargo oxide run \
  --features cuda,hyperkzg,gpu-msm --arch sm_70 --bin cuda-nova -- \
  --profile-forward --ptau-dir ./ptau_files \
  --profile-revision "$CUDA_NOVA_REVISION" \
  --profile-json receipts/cuda-nova-forward-profile.json
```

The destination is create-new and is never overwritten. The versioned JSON
schema records the selected commitment backend, CUDA/MSM feature state, PTX and
device selection, optional PTau directory, setup time and circuit size, prover
initialization/recursive-fold/finalization wall time, verification time,
per-phase CUDA call/time/cache/allocation/synchronization deltas, Blitzar MSM
call/scalar/time deltas, and the separate chunked-commitment result. CUDA and
MSM times are accumulated provider-call durations; parallel callers can make
their sum exceed phase wall time. Without `--profile-json`, the same receipt is
written to standard output. Schema v3 distinguishes logical `spmv_calls` from
physical `spmv_batches`; the fused A/B/C path should report exactly three
logical products per batch.

The arithmetic runtime keeps a bounded exact-key cache of static CSR device
buffers and completely overwritten output buffers. Dynamic witness vectors
are still uploaded per call because cuda-oxide does not expose a safe pageable
in-place refill API under this crate's `unsafe_code = forbid` policy. One final
device-to-host copy synchronizes each call; redundant pre-kernel and
post-kernel synchronizations are omitted because all work uses one ordered
stream. Nova's A/B/C matrices share the same dense vector, so their CSR rows
are vertically concatenated under an exact three-matrix cache key. One vector
upload, one existing checked SpMV kernel launch, and one result download then
produce the same three logical products without changing the proof protocol.

## Validation

Portable and HyperKZG regression checks from the workspace root:

```sh
cargo test -p cuda-nova
cargo test -p cuda-nova --features hyperkzg
cargo clippy -p cuda-nova --all-targets --features hyperkzg -- -D warnings
```

The CUDA/HyperKZG/GPU-MSM path must additionally be run through `cargo oxide`
on a supported Linux GPU. A successful profile is accepted only when every
recursive case and the two-chunk regression report `verified: true`.

## Scope limits

- The standard receipt measures a fixed 3-neuron, 4-edge, 14,826-constraint
  circuit for 1, 8, 64, and 1,024 recursive steps.
- The 166,700-neuron graph execution benchmark is a numeric CUDA forward pass,
  not a Nova proving benchmark.
- Full-graph proof constraints and witness bytes are capacity lower bounds;
  no full MaleCNS proof time is claimed.
- Nova recursion is sequential here. No distributed proof aggregation protocol
  is implemented.

## License

This crate is licensed under the GNU General Public License, version 3.0
only. See [LICENSE](LICENSE) for the complete text.

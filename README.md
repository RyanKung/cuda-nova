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
3. delegate R1CS synthesis and recursive folding to `zkfly-nova`, which uses
   the official `nova-snark` crate.

The third stage is still host-side. The current GPU Poseidon path is a
correctness baseline using canonical-limb double-and-add multiplication; its
timings are not a field-arithmetic performance claim. A future GPU field/MSM
engine can replace that implementation without changing the transcript ABI or
the public `CudaNovaEngine` boundary.

On the V100, generate the PTX and run the two-step smoke fixture with:

```sh
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_TOOLKIT_PATH=/usr/local/cuda
CUDA_OXIDE_TARGET=sm_70 cargo oxide run \
  --features cuda --arch sm_70 --bin cuda-nova -- --prove
```

The output includes upload, repeated-kernel, download, and end-to-end
microsecond measurements, plus a Poseidon preflight timing. The numbers are a
smoke baseline, not a full MaleCNS proof benchmark; the Poseidon timing is
especially a correctness baseline until the wide-product kernel is validated.

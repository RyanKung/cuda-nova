# cuda-nova

This crate is the CUDA sidecar for the official `nova-snark` topology prover.
It is kept inside `zkfly` while the API is measured, but its dependency and
runtime boundary is designed for extraction into a standalone public
repository.

The current engine has three explicit stages:

1. encode `TopologyFoldStep` values into a stable 432-byte record;
2. upload and preflight canonical indices, accumulator links, and padding on a
   CUDA device with cuda-oxide;
3. delegate R1CS synthesis and recursive folding to `zkfly-nova`, which uses
   the official `nova-snark` crate.

The third stage is still host-side. A future GPU field/MSM engine can replace
that implementation without changing the transcript ABI or the public
`CudaNovaEngine` boundary.

On the V100, generate the PTX and run the two-step smoke fixture with:

```sh
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_TOOLKIT_PATH=/usr/local/cuda
CUDA_OXIDE_TARGET=sm_70 cargo oxide run \
  --features cuda --arch sm_70 --bin cuda-nova -- --prove
```

The output includes upload, kernel, download, and end-to-end microsecond
measurements. The numbers are a smoke baseline, not a full MaleCNS proof
benchmark.

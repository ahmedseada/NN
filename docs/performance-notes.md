# Performance notes: where the optimization work stopped

Measured on an RTX 5070 Ti (70 SMs, 16 GB). Resume from here when returning to optimization.

## Before relying on the current state

1. **Pretrained check against transformers.** Split k with atomic additions, the fused decoding kernels and the new
   split heuristics all change the order of floating-point sums. The unit tests pass (272), but the end-to-end check on
   a real model has not been run since:

   ```
   python tools/pytorch/pretrained_reference.py --model Qwen/Qwen3-0.6B --out qwen3.json
   dotnet run -c Release --project samples/NeuralSharp.Samples.Pretrained -- check qwen3.json --cuda
   dotnet run -c Release --project samples/NeuralSharp.Samples.Pretrained -- check qwen3.json --cuda --int8 --kv16
   ```

   The first compares float32 products with transformers exactly as before. The second goes through the packed-weight
   decoding kernels (int8 GEMVs with the residual and norm, gate/up with the activation, the fused head layout and cache
   writes); int8 weights move logits a little, so compare its greedy text and the size of the differences, not
   exact equality.

2. **FP8 against bfloat16 loss curves**, before training for real with `--fp8`. Two runs from the same seed and
   schedule, then the comparison (short runs: `--no-save` skips the checkpoints, `--steps` sets the schedule):

   ```
   $common = "--bin D:\TF.NET\POC\TinyCharTransformer\python\data\final_corpus.bin --vocab-file D:\TF.NET\POC\TinyCharTransformer\python\data\final_corpus.vocab --block 512 --batch 8 --dmodel 2048 --heads 16 --layers 24 --steps 3000 --warmup 300 --eval-every 250 --eval-steps 20 --grad-checkpoint --optim8bit --no-save --log-every 250".Split(' ')
   dotnet run -c Release --project samples/NeuralSharp.Samples.CharGpt -- @common --out checkpoints/cmp-bf16 --loss-log bf16.csv
   dotnet run -c Release --project samples/NeuralSharp.Samples.CharGpt -- @common --out checkpoints/cmp-fp8 --loss-log fp8.csv --fp8
   dotnet run -c Release --project samples/NeuralSharp.Samples.CharGpt -- compare bf16.csv fp8.csv --window 250
   ```

   FP8 is usable when its training and validation losses stay within about 1% of bfloat16 through the run. Delayed
   scaling gets the same test (a third run with `$env:NEURALSHARP_FP8_DELAYED=1`) before it can be turned on by default.

## Results so far

| | Start | Now |
|---|---|---|
| Qwen3-0.6B int8, bf16 KV cache: greedy decoding | 391 tok/s | 474 tok/s |
| Chat sampling (temperature 0.6, top-k 20, top-p 0.95) | 227 tok/s | 249 tok/s |
| Kernel launches per decoded token | 345 | 177 |
| 180-token prompt pass | 13.6 ms | 9.0 ms |
| Decoding attention, 4000 cached positions | 76 µs | 38 µs |
| 1.25B char model training step, FP8 | 592 ms | 560 ms (548 with delayed scaling) |
| 1.25B char model training step, bfloat16 | 736 ms | 712 ms |
| Weight-gradient products, small output and long k (768×768×12288 tn) | 29 TFLOPS | 59 TFLOPS |

What changed:

- Decoding: q/k RMS norm, rotary embedding, head layout and KV-cache writes in one kernel (`norm_rope_heads_f32`);
  split decoding attention merged by its last block; packed GEMVs with the residual addition and next RMSNorm
  (`*_gemv_addnorm_f32`) and with the gated activation (`*_gemv_multi_act_f32`); GEMV k splits ≈ 2 blocks per SM
  (power of two); decoding attention ≈ 5 blocks per SM over the cache.
- Prompts: packed tensor-core products split k up to ≈ 4 blocks per SM.
- Training: plain tensor-core products with fewer than 4 output tiles per SM split k (≈ 16 blocks per SM, ≤ 8 splits,
  ≥ 1024 k each). Delayed FP8 column scaling (kept per-column maxima with 2× headroom, recorded in the same pass,
  correction pass for columns that more than doubled) behind `NEURALSHARP_FP8_DELAYED=1`.
- Tools: `--bench-gemm` (with a split sweep for long-k shapes), `--bench-gemv` (decoding products, decoding attention
  and 180-row products with forced splits), `--bench-fp8` (column quantizers and delayed-scaling error), the
  per-kernel GPU-time tables of the Pretrained `profile` command and CharGpt `--profile` (CUDA events; small kernels
  read about 5 µs high, so compare ratios).

## Next steps, in order of expected gain

1. **Delayed FP8 scaling: identify operands by call order, not address.** In training it saved only 12 ms of the
   expected ~30 (`quant_cols` 73 → 61 ms against ~41 predicted by `--bench-fp8`). The likely cause: with
   `--grad-checkpoint` the same buffer address holds different tensors in the forward pass, the recompute and the
   backward pass, so kept maxima belong to another tensor and the correction fires. Key the kept maxima by the
   position of the quantization within the step (reset at each optimizer step) plus the shape, and count corrections
   to confirm. Then the loss-curve test above, then consider turning it on by default.
2. **FP8 quantization fused into its producers** (the larger payoff of delayed scaling): with scales known before a
   tensor exists, layer norm, GELU and the product epilogues can write FP8 directly, removing most of the remaining
   ~100 ms of `quant_rows` / `quant_cols` per 1.25B step.
3. **Decoding GEMVs:** 400–580 GB/s on 4–6 MB of weights, about 2–3 µs of fixed cost per launch; the next step is fewer,
   larger kernels (several layers' products in one persistent kernel) rather than more split tuning. Decoding is at
   ~31% of the memory-bandwidth ceiling (~1.5k tok/s for this model).
4. **INT8 prefill on int8 tensor cores:** prompt products currently expand int8 weights to bfloat16 tiles; the int8
   tensor-core path needs a second, k-major copy of the weights (memory cost: the int8 weights once more).
5. **Training step outside the products** (1.25B, FP8): `adam8` 34 ms, flash-attention backward 32 ms, `sumsq` for
   gradient clipping 9 ms, `fill` (zeroing gradients) 8 ms, dropout 10 ms. Candidates: the first gradient write with
   beta 0 instead of zeroing, clipping's norm fused into the backward pass.
6. **Decoding attention at short contexts:** the ≈5 blocks per SM rule is ~1 µs slower than 16 splits at 200 positions;
   a length-aware split would need the graph-replay constraint handled (splits must not depend on the cache length).

## Measured facts worth keeping

- One-launch column quantization (maxima and conversion in one kernel through L2) was built and measured: only 13% faster
  on 67 MB tensors and no gain elsewhere, so it was removed.
- FP8 relative precision does not suffer from a 2× larger scale (e4m3 is floating point); the drift test in
  `--bench-fp8` gives the same error with and without the headroom.
- Split-K weight gradients make bfloat16 training sums order-dependent (atomic additions): runs are no longer
  bit-for-bit reproducible.
- The event-timed profile tables overstate kernels of a few microseconds; `--bench-gemv` replays graphs and gives the
  real per-call times.

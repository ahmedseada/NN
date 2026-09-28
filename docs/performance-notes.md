# Performance notes: where the optimization work stopped

Measured on an RTX 5070 Ti (70 SMs, 16 GB). Resume from here when returning to optimization.

## Checks done (2026-09-28)

1. **Pretrained check against transformers** (Qwen3-0.6B, `qwen3.json` from `tools/pytorch/pretrained_reference.py`):
   - float32 (`check qwen3.json --cuda`): everything matches; max |Δlogit| 4e-5 to 1.4e-4, greedy output identical.
   - int8 weights, bfloat16 KV cache (`--int8 --kv16`): top-1 the same everywhere, greedy identical on 3 of 4 prompts
     (one C# continuation diverges after 12 characters into another plausible one); max |Δlogit| 0.5 to 1.4, as int8
     weights give. Decoding 486–575 tok/s in the check.
2. **FP8 against bfloat16 loss curves**, quick version (100.9M-parameter char model: d_model 768, 12 layers, batch 32,
   block 384; 600 steps, same seed and schedule; `--loss-log` and `compare`):

   | | bf16 | fp8 | fp8 + delayed scaling |
   |---|---|---|---|
   | validation loss, step 600 | 2.7425 | 2.7388 (-0.13%) | 2.7250 (-0.64%) |
   | training loss, last fifth | 2.6703 | 2.6693 (-0.03%) | 2.6589 (-0.43%) |
   | largest gap at any validation step | | 0.43% (step 100) | 0.64% |
   | steps/s | 5.64 | 5.75 | 5.76 |

   Both FP8 variants stay within 1% of bfloat16 throughout. Not yet run: the long comparison on the 1.25B model (3000
   steps, about 36 + 28 minutes), advisable once before a long `--fp8` run of that size:

   ```
   $common = "--bin D:\TF.NET\POC\TinyCharTransformer\python\data\final_corpus.bin --vocab-file D:\TF.NET\POC\TinyCharTransformer\python\data\final_corpus.vocab --block 512 --batch 8 --dmodel 2048 --heads 16 --layers 24 --steps 3000 --warmup 300 --eval-every 250 --eval-steps 20 --grad-checkpoint --optim8bit --no-save --log-every 250".Split(' ')
   dotnet run -c Release --project samples/NeuralSharp.Samples.CharGpt -- @common --out checkpoints/cmp-bf16 --loss-log bf16.csv
   dotnet run -c Release --project samples/NeuralSharp.Samples.CharGpt -- @common --out checkpoints/cmp-fp8 --loss-log fp8.csv --fp8
   dotnet run -c Release --project samples/NeuralSharp.Samples.CharGpt -- compare bf16.csv fp8.csv --window 250
   ```

   At this model size FP8 is barely faster than bfloat16 (2%): the quantization passes cost about what the 8-bit
   products save. The gain grows with the model (1.25B: 560 against 712 ms per step).

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
   to confirm. The quick loss-curve test already passes with it (within 0.64% of bfloat16); after the fix, rerun it
   and the 1.25B comparison, then consider turning it on by default.
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

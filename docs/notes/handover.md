# Handover (start of the rebrand)

## Where things are
- Repository `ahmedseada/NN` (NeuralSharp: C# / .NET 10 deep-learning library with hand-written PTX CUDA kernels).
- Working branch: **`rebrand`** (created from `agentic-coding-tooling` at `c5b00c9`; everything before it is merged
  there). Push with `git push -u origin HEAD`.
- The user works on Windows, PowerShell, `D:\Projects\NN`, one GPU: **RTX 5070 Ti 16 GB** (the RTX 5050 8 GB and
  RTX 3060 6 GB machines are a later priority). The cloud container has no GPU: GPU work is checked by the user.
- Rules are in `CLAUDE.md` (library-first, one fine-tuning tool, model-agnostic, any CUDA GPU, commands start with
  `git pull origin rebrand`, say what was only checked on the CPU).
- Commit trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and the session link; no model names in
  code or commits.

## Next task: the rebrand
Not specified yet: ask for the new name and scope (namespaces `NeuralSharp.*`, projects and folders under `src/`,
`tests/NeuralSharp.Tests`, tools `nstune` / `nsdata`, file formats such as `neuralsharp-tuning.json` and `.nsm`,
docs, READMEs, samples, apps). Map every occurrence before changing anything; keep file formats readable (old names
still load) unless told otherwise.

## Library state (all tested; GPU tests: 334 passed on the 5070 Ti)
- Fine-tuning (`src/NeuralSharp.Pretrained/FineTuning.cs`, tool `src/NeuralSharp.FineTuning.Cli` = nstune): LoRA /
  QLoRA, packing, CUDA graphs, fused optimizer, released / recomputed / bfloat16 activations, out-of-memory fallback
  that measures the lighter setting against checkpointing, first gradients written not accumulated, shared input
  gradient buffers, `BalanceAnswers` (nstune `--balance N`), batches that mix sequences of one length at random.
  Speed on the 5070 Ti: Qwen2.5-0.5B, 4k-token steps: 17.4k tok/s (chat data), 25.5k tok/s (Qasd); 16k steps 13.9k.
- Scoring: `Losses.TokenLogProbabilities`, `AnswerScorer` (log p of given answers; each prompt once through a
  key/value cache; bfloat16 tensor cores by default; `PositionsPerPass` bounds memory); nstune `evaluate --choices`.
- Generation: `TextGenerator.StreamBatch`, `ChatGenerator.StreamBatch(Async)`; cached forward pass frees each layer.
- Reverted after GPU measurement (slower): thin rank-16 CUDA-core kernels, the gated activation fused into the up
  projection. Kernel work on the tuner is stopped by the user's decision (big products run at 68-73 TFLOPS).

## Qasd (apps/Qasd*: intent classifier + tuned LLM; paused for the rebrand)
- Tuned model results on 61,057 held-out messages: `tuned-speed` 85.4 % accuracy, macro F1 0.779 (function_call
  recall 57.5 %); `tuned-balanced` (`--balance 3`) 84.5 %, 0.766 (function_call recall 69.1 %, precision 63.2 %).
  Tune 11 min, evaluate 2 min on the 5070 Ti.
- `qasd audit`: 32.6 % of rows are texts with more than one label; ceiling for a message-only model 97.3 %;
  function_call is 8.4 % of the data and 44 % of its messages are ≤ 3 words (context-dependent replies).
- Open decisions for the user: which adapter to serve; label questions ("عايز احجز ميعاد" mostly labelled retrieve,
  "where are you located?" labelled identity); a per-intent score offset + `qasd calibrate` — **in the app only**
  (the user ruled it out of the library); sending the bot's previous question with the message.
- On hold: `docs/notes/qasd-orchestration-mcp.md` (Qasd → orchestrator → planner with tools over MCP; what the
  library has and lacks).

## How the user tests
```powershell
cd D:\Projects\NN
git pull origin rebrand
dotnet run -c Release --project tests/NeuralSharp.Tests
```
In the container: `NS_DEVICES=cpu dotnet run -c Release --project tests/NeuralSharp.Tests` (167 CPU tests pass);
`NS_FILTER=text` runs matching tests. PTX can be assembled with ptxas from the pip package `nvidia-cuda-nvcc-cu12`.

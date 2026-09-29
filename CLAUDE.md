# Working on this repository

- **Fixes and general features go in the library.** An application (`apps/`, `samples/`) is a user of NeuralSharp and
  a test of it: when an application hits a bug or a missing capability that another application could hit too (memory,
  speed, precision, data handling, fine-tuning, generation, scoring, progress reporting), fix or add it in `src/`, with a
  test in `tests/NeuralSharp.Tests`, and have the application call it. Only logic specific to that application stays in
  the application.
- **Fine-tuning goes through one tool.** Tuning a pretrained model is done with the general fine-tuning console, not a
  console per project. A project that needs its own training (like the Qasd hashed-feature classifier) keeps one.
- **GPU work is checked on a GPU.** This container has no GPU: say which parts were only checked on the CPU, and give
  the exact commands to run the GPU tests (`dotnet run -c Release --project tests/NeuralSharp.Tests`).
- **Commands for the user** (Windows, PowerShell, `D:\Projects\NN`) always start with
  `git pull origin agentic-coding-tooling`.
- **Fine-tuning is model-agnostic.** The tuner, the tools and the applications work with any model the library loads,
  through its own tokenizer and chat template; a model used for testing (Qwen today) is only an example. Nothing outside
  the library's architecture support may depend on a particular model family.

# nsdata

Inspect, download and assemble datasets from the command line. The output is JSON Lines (one JSON object per line):
conversations as `{"messages": [...], "tools": [...]}` and plain text as `{"text": ...}`, the layout NeuralSharp's
fine-tuning, Hugging Face's `datasets` and most training tools read.

## Install

As a .NET tool (needs the .NET SDK):

```
dotnet pack src/NeuralSharp.Datasets.Cli -c Release -o artifacts
dotnet tool install --global --add-source artifacts NeuralSharp.Datasets.Cli
nsdata --help
```

Update after a `git pull`: pack again, then `dotnet tool update --global --add-source artifacts NeuralSharp.Datasets.Cli`
(each pack gets a new version number, so the update always installs it).

As one executable that runs without .NET (Native AOT; `win-x64`, `linux-x64` or `osx-arm64`):

```
dotnet publish src/NeuralSharp.Datasets.Cli -c Release -r win-x64 -p:PublishAot=true -p:PackAsTool=false -o artifacts/nsdata
```

Or without installing: `dotnet run --project src/NeuralSharp.Datasets.Cli -- <command>`.

## Examples

```
nsdata show "hf:openai/gsm8k?config=main"
nsdata count "hf:openai/gsm8k?config=main" "hf:openai/gsm8k?config=main&split=test"
nsdata build "hf:openai/gsm8k?config=main&user={question}&assistant={answer}" --out gsm8k.jsonl --eval-fraction 0.02
nsdata show "github:owner/repo?files=src/**/*.cs"
nsdata build recipe.json --out train.jsonl
```

A recipe mixes sources:

```json
{
  "sources": [
    {"source": "hf:openai/gsm8k", "config": "main", "user": "{question}", "assistant": "{answer}", "weight": 1, "take": 2000},
    {"source": "hf:yahma/alpaca-cleaned", "weight": 2, "take": 4000},
    "my-examples.jsonl?weight=0.5"
  ],
  "system": "You are a helpful assistant.",
  "seed": 1, "min_chars": 20, "eval_fraction": 0.02
}
```

Credentials come from the environment: `HF_TOKEN` (or `huggingface-cli login`), `GITHUB_TOKEN`, `KAGGLE_USERNAME` and
`KAGGLE_KEY` (or `~/.kaggle/kaggle.json`), `ZENODO_TOKEN`. Downloads are cached under `NEURALSHARP_CACHE` or
`~/.cache/neuralsharp`, in `downloads/` laid out by source:

```
downloads/huggingface/datasets/<owner>/<name>/<commit>/<path in the repository>
downloads/huggingface/datasets/<owner>/<name>/parquet/<config>/<split>/00000.parquet   (the Hub's Parquet copy)
downloads/github/<owner>/<repo>/<commit>.tar.gz                                     (repository snapshots)
downloads/github/<owner>/<repo>/<commit>/<path>                                     (single files)
downloads/github/<owner>/<repo>/releases/<tag>/<asset>
downloads/kaggle/<owner>/<dataset>/<latest | vN>/<dataset>.zip
downloads/zenodo/<record>/<file>
downloads/urls/<host>/<path>
```

Branches and tags are resolved to their commit first, so new commits are downloaded again rather than read stale.
`nsdata cache` shows the size per source; `nsdata cache --clear` empties it. `nsdata --help` lists
every option.

The same features are available from code in the `NeuralSharp.Datasets` library (`Dataset`, `HuggingFace`, `GitHub`,
`Kaggle`, `Zenodo`, `ChatRows`, `DatasetSpec`, `DatasetRecipe`).

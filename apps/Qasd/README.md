# Qasd (قصد): intent classifier

An application built on NeuralSharp (it is not part of the library): it classifies a user's message into an intent
(`retrieve`, `function_call`, `direct_reply`, `identity`, or whatever labels the training data has). It trains on a GPU
or CPU from a CSV of labeled messages and serves predictions from one model file.

| Project | What it is |
|---|---|
| `Qasd.Core` | the model: training, evaluation, prediction, one-file save / load |
| `Qasd` | the command line (`qasd`): train, evaluate, predict, info |
| `Qasd.Api` | the HTTP service (ASP.NET Core) serving a trained model, API reference with Scalar |

`apps/Qasd.slnx` opens all three.

How it works: each message becomes a hashed bag of words, word pairs and character 2-4-grams (TF-IDF weighted; Arabic
normalized: diacritics, alef / yaa / taa marbuta forms, Arabic-Indic digits), and a small network built with NeuralSharp
maps it to the intents.

## Train

```
dotnet run -c Release --project apps/Qasd -- train plan-queries.csv --out models/intents.nsm --cuda
```

The columns default to `raw_question` (text) and `intent` (label); `--text` / `--label` pick others. 20% of the distinct
messages are held out (all copies of a message stay on one side, so the score is on messages the model never saw):
accuracy, per-intent precision / recall / F1 and the confusion matrix are printed. For the model you deploy, train on
everything with `--test-fraction 0`. Other options: `--epochs`, `--batch-size`, `--buckets`, `--hidden`, `--lr`, `--seed`.

## Use

```
dotnet run -c Release --project apps/Qasd -- predict models/intents.nsm "Book me with Dr. Heba on Tuesday" "هلا"
dotnet run -c Release --project apps/Qasd -- predict models/intents.nsm --json --min-confidence 0.6 < messages.txt
dotnet run -c Release --project apps/Qasd -- evaluate models/intents.nsm new-labeled.csv
```

## Serve (Qasd.Api)

```
dotnet run -c Release --project apps/Qasd.Api
```

Then open http://localhost:5080 (the Scalar API reference, where every endpoint can be tried). The model file and
settings come from `appsettings.json` (section `IntentModel`), or environment variables such as
`IntentModel__Path=D:\models\intents.nsm`:

| Setting | Default | |
|---|---|---|
| `Path` | `models/intents.nsm` | the model file from `qasd train` |
| `Device` | `cpu` | `auto`, `cpu`, `cuda` or `cuda:N` (the CPU is fast enough for single messages) |
| `MinConfidence` | 0.6 | below it a result has `accepted: false`: send the message to your fallback (the LLM planner) |
| `MaxBatch` | 256 | most messages per batch request |
| `MaxTextLength` | 4000 | longest message in characters |
| `AdminKey` | empty | the `X-Admin-Key` for `POST /v1/model/reload`; empty disables reloading |

| Endpoint | |
|---|---|
| `POST /v1/classify` | `{"text": "..."}` → label, confidence, accepted, every intent's probability |
| `POST /v1/classify/batch` | `{"texts": ["...", "..."]}` → the results in order (one pass through the model) |
| `GET /v1/model` | the model file, its intents, device and load time |
| `POST /v1/model/reload` | after retraining: loads the file again without dropping requests in flight |
| `GET /health` | healthy while a model is loaded |
| `GET /openapi/v1.json` | the OpenAPI document |

The model is loaded at startup (a missing or broken file stops the service with a clear message, not the first
request). Invalid input gets a 400 with ProblemDetails.

From your own .NET code, reference `Qasd.Core` and load the model once:

```csharp
using var classifier = Qasd.TextClassifier.Load("models/intents.nsm", NeuralSharp.Device.Cpu);
var p = classifier.Predict(message);          // p.Label, p.Confidence, p.Probabilities; thread-safe
```

## Publish

```
dotnet publish apps/Qasd.Api -c Release -o artifacts/intent-api        (then copy the model next to it)
dotnet publish apps/Qasd -c Release -r win-x64 -o artifacts/qasd
```

`Qasd.Core` references the NeuralSharp projects in this repository; when the library is published as
packages, replace its two `ProjectReference`s with `PackageReference`s and the application can live in its own repository.

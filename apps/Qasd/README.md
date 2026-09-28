# Qasd (قصد): intent classifier

An application built on NeuralSharp (it is not part of the library): it classifies a user's message into an intent
(`retrieve`, `function_call`, `direct_reply`, `identity`, or whatever labels the training data has). It trains on a GPU
or CPU from a CSV of labeled messages and serves predictions from one model file.

| Project | What it is |
|---|---|
| `Qasd.Core` | the model: training, evaluation, prediction, one-file save / load |
| `Qasd` | the command line (`qasd`): train, evaluate, predict, info |
| `Qasd.Tuned` | the command line (`qasd-tuned`) that tunes a pretrained chat model (LoRA) on the same data: train, evaluate, predict (streamed), benchmark |
| `Qasd.Api` | the HTTP service (ASP.NET Core) serving both models, as one JSON response or streamed; test page and Scalar reference |

`apps/Qasd.slnx` opens all four.

| | classifier (`qasd`) | tuned model (`qasd-tuned`) |
|---|---|---|
| what it is | hashed n-grams and a small network | a pretrained chat model (default Qwen/Qwen2.5-0.5B-Instruct) taught to answer with the intent |
| trains in | seconds on a CPU | minutes on a GPU (hours on a CPU) |
| predicts in | about 0.1 ms per message on a CPU | tens of milliseconds per message |
| result | intent, confidence, every intent's probability | the same (each intent scored as the model's answer), plus its answer streamed token by token |

How it works: each message becomes a hashed bag of words, word pairs and character 2-4-grams (TF-IDF weighted; Arabic
normalized: diacritics, alef / yaa / taa marbuta forms, Arabic-Indic digits), and a small network built with NeuralSharp
maps it to the intents.

## Train

```
dotnet run -c Release --project apps/Qasd -- train apps/Qasd/data/plan-queries.csv --out models/intents.nsm
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

## Devices

Training (`qasd train`, `qasd-tuned train`) runs on the GPU when there is one (else the CPU, with a notice); `--cpu`
forces the CPU. Inference runs on the CPU unless asked: `predict` and `evaluate` take `--cuda`, and the service takes
`IntentModel__Device=cuda` (classifier) and `IntentModel__TunedDevice=cuda` (tuned model). The classifier is fastest on
the CPU (about 0.1 ms per message); the tuned model is a language model and gains most from the GPU. The benchmarks
measure both devices when there is a GPU (`--devices cpu` or `--devices cuda` for one).

## Tune a language model (Qasd.Tuned)

```
dotnet run -c Release --project apps/Qasd.Tuned -- train apps/Qasd/data/plan-queries.csv --out models/qasd-tuned
dotnet run -c Release --project apps/Qasd.Tuned -- predict models/qasd-tuned "Book me with Dr. Heba on Tuesday" --stream
```

It downloads the base model once (`--model` picks another: a Hugging Face id, a folder, a .gguf file or `ollama:name`),
tunes LoRA adapters so each message is answered with its intent, and holds out the same messages as `qasd train`, so the
two scores compare directly. The output folder holds the adapter and `qasd-tuned.json` (base model, intents,
instruction). Predictions score every intent as the model's answer, so the label is always one of the trained intents.

Compare both models on a device:

```
dotnet run -c Release --project apps/Qasd.Tuned -- benchmark apps/Qasd/data/plan-queries.csv                 (CPU)
dotnet run -c Release --project apps/Qasd.Tuned -- benchmark apps/Qasd/data/plan-queries.csv --devices cpu,cuda
dotnet run -c Release --project apps/Qasd.Tuned -- benchmark apps/Qasd/data/plan-queries.csv --sample 400      (quick CPU run)
```

## Serve (Qasd.Api)

```
dotnet run -c Release --project apps/Qasd.Api
```

Then open http://localhost:5080: a test page (messages on the left, one per line; each message's intent, confidence
and probabilities with the round-trip and model times on the right). The Scalar API reference, where every endpoint
can be tried, is at http://localhost:5080/scalar.

![The test page](docs/qasd-light.png)

On the CPU, inference skips the dense product: a message sets a few hundred of the 16,384 features, so the first layer
adds up those rows of its weights (about 0.1 ms per message, 40,000 messages per second in batches on 4 cores). The model file and
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
| `TunedPath` | empty | the folder from `qasd-tuned train`; empty: only the classifier is served |
| `TunedDevice` | `cpu` | where the tuned model runs (`cuda` recommended) |
| `MaxTunedBatch` | 32 | most messages per batch request to the tuned model |
| `DefaultModel` | `classifier` | the model requests use when they name none |

| Endpoint | |
|---|---|
| `POST /v1/classify` | `{"text": "...", "model": "classifier" \| "tuned", "stream": false}` → label, confidence, accepted, every intent's probability |
| `POST /v1/classify/batch` | `{"texts": ["...", "..."], "model": ..., "stream": false}` → the results in order |
| `GET /v1/models` | both models: available, default, intents, device |
| `GET /v1/model` | the model file, its intents, device and load time |
| `POST /v1/model/reload` | after retraining: loads the file again without dropping requests in flight |
| `GET /health` | healthy while a model is loaded |
| `GET /openapi/v1.json` | the OpenAPI document |

With `"stream": true` both classify endpoints answer with Server-Sent Events (`text/event-stream`) instead of one
JSON response:

```
event: token      data: {"index":0,"token":"function"}                    (tuned model: its answer as it is generated)
event: result     data: {"index":0,"result":{"label":"function_call","confidence":0.97,...}}   (one per message)
event: done       data: {"count":1,"elapsedMilliseconds":41.2,"firstTokenMilliseconds":18.5,"model":"tuned"}
```

The test page switches between the two models; with the tuned model selected, the Stream switch shows its answer as it
is generated, with the time to the first token.

![The tuned model on the test page](docs/qasd-tuned.png)

The model is loaded at startup (a missing or broken file stops the service with a clear message, not the first
request). Invalid input gets a 400 with ProblemDetails.

From your own .NET code, reference `Qasd.Core` and load the model once:

```csharp
using var classifier = Qasd.TextClassifier.Load("models/intents.nsm", NeuralSharp.Device.Cpu);
var p = classifier.Predict(message);          // p.Label, p.Confidence, p.Probabilities; thread-safe
```

## Training data

`apps/Qasd/data/plan-queries.csv`: 5,177 labeled messages (1,872 distinct; Arabic, English and mixed) with the columns
`raw_question` (the text), `semantic_query`, `intent` (the label: `retrieve` 2,983, `function_call` 1,096,
`direct_reply` 973, `identity` 125) and `confidence`. Retrain after adding rows; later rows of the same message count
as more examples of it.

## Postman

`apps/Qasd.Api/Qasd.postman_collection.json`: every endpoint with English and Arabic examples, the validation errors and
the admin reload, each with tests (status, labels, probabilities summing to 1, `accepted` against MinConfidence).
Import it in Postman (variables `baseUrl`, default http://localhost:5080, and `adminKey`), or run it from the command line:

```
npx newman run apps/Qasd.Api/Qasd.postman_collection.json --env-var adminKey=<the service's IntentModel:AdminKey>
```

## Publish

```
dotnet publish apps/Qasd.Api -c Release -o artifacts/intent-api        (then copy the model next to it)
dotnet publish apps/Qasd -c Release -r win-x64 -o artifacts/qasd
```

`Qasd.Core` references the NeuralSharp projects in this repository; when the library is published as
packages, replace its two `ProjectReference`s with `PackageReference`s and the application can live in its own repository.

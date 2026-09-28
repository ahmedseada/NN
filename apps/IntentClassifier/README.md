# Intent classifier

An application built on NeuralSharp (it is not part of the library): it classifies a user's message into an intent
(`retrieve`, `function_call`, `direct_reply`, `identity`, or whatever labels the training data has). It trains on a GPU
or CPU from a CSV of labeled messages and serves predictions from one model file.

How it works: each message becomes a hashed bag of words, word pairs and character 2-4-grams (TF-IDF weighted; Arabic
normalized: diacritics, alef / yaa / taa marbuta forms, Arabic-Indic digits), and a small network built with NeuralSharp
maps it to the intents.

## Train

```
dotnet run -c Release --project apps/IntentClassifier -- train plan-queries.csv --out models/intents.nsm --cuda
```

The columns default to `raw_question` (text) and `intent` (label); `--text` / `--label` pick others. 20% of the distinct
messages are held out (all copies of a message stay on one side, so the score is on messages the model never saw):
accuracy, per-intent precision / recall / F1 and the confusion matrix are printed. For the model you deploy, train on
everything with `--test-fraction 0`. Other options: `--epochs`, `--batch-size`, `--buckets`, `--hidden`, `--lr`, `--seed`.

## Use

```
dotnet run -c Release --project apps/IntentClassifier -- predict models/intents.nsm "Book me with Dr. Heba on Tuesday" "هلا"
dotnet run -c Release --project apps/IntentClassifier -- predict models/intents.nsm --json --min-confidence 0.6 < messages.txt
dotnet run -c Release --project apps/IntentClassifier -- evaluate models/intents.nsm new-labeled.csv
```

From a service, reference this project (or copy `TextClassifier.cs`) and load the model once:

```csharp
using var classifier = IntentClassifier.TextClassifier.Load("models/intents.nsm", NeuralSharp.Device.Cpu);
var p = classifier.Predict(message);          // p.Label, p.Confidence, p.Probabilities; thread-safe
```

Messages below a confidence (say 0.6) are better sent to a fallback (the LLM planner) than trusted.

## Publish

```
dotnet publish apps/IntentClassifier -c Release -r win-x64 -o artifacts/intent-classifier
```

It references the NeuralSharp projects in this repository; when the library is published as packages, replace the two
`ProjectReference`s in `IntentClassifier.csproj` with `PackageReference`s and the application can live in its own repository.

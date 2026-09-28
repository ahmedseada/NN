# nsclassify

Train, evaluate and run text classifiers (intents, topics, routing) from the command line, on a GPU or CPU. Texts in
any language; Arabic and Latin text are normalized (diacritics, letter forms, digits). The library behind it is
`NeuralSharp.Classification` (`TextClassifier`), for use from your own application.

## Install

As a .NET tool (needs the .NET SDK):

```
dotnet pack src/NeuralSharp.Classification.Cli -c Release -o artifacts
dotnet tool install --global --add-source artifacts NeuralSharp.Classification.Cli
nsclassify --help
```

Update after a `git pull`: pack again, then `dotnet tool update --global --add-source artifacts NeuralSharp.Classification.Cli`.

Or without installing: `dotnet run -c Release --project src/NeuralSharp.Classification.Cli -- <command>`.

## Examples

```
nsclassify train queries.csv --text raw_question --label intent --out intents.nsm --cuda
nsclassify evaluate intents.nsm new-queries.csv
nsclassify predict intents.nsm "Book me with Dr. Heba on Tuesday" "هلا"
type questions.txt | nsclassify predict intents.nsm --json --min-confidence 0.6
nsclassify info intents.nsm
```

`train` holds out 20% of the distinct texts (every copy of a text stays on one side, so the score is on texts the model
never saw), keeps the epoch that scored best on a validation part of the rest, prints accuracy, per-label precision,
recall and F1 and the confusion matrix, and saves one file. Train on everything for the final model with
`--test-fraction 0`.

Data can be a CSV, JSON Lines, JSON or Parquet file or folder, a URL or a dataset spec as `nsdata` reads it
(`hf:owner/name?split=train`); several sources are combined.

## In your application

```csharp
using NeuralSharp;
using NeuralSharp.Classification;

using var classifier = TextClassifier.Load("intents.nsm", Device.Cpu);   // once, at startup; share the instance
var prediction = classifier.Predict("عايز احجز مع د. هبة يوم الخميس");
// prediction.Label, prediction.Confidence, prediction.Probabilities
```

`Predict` is safe to call from several threads; `Predict(IReadOnlyList<string>)` classifies many texts in one pass.
Training from code: `TextClassifier.Read`, `TextClassifier.Split`, `TextClassifier.Train`, `Evaluate`, `Save`.

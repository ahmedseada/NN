using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json.Nodes;
using Idrak;
using Qasd;
using Idrak.Datasets;
using Idrak.LanguageModels;

// qasd: trains and serves the intent model (an application built on Idrak).
const string Usage = """
    qasd: train, evaluate and run text classifiers (intents, topics, routing; any language)

      qasd train <data…> [--out F]         train on labeled texts, report the score on held-out texts, save the model
                                          (default apps/Qasd/models/intents.qasd)
      qasd evaluate [model] <data…>        score a model on labeled texts (accuracy, per-label F1, confusion matrix)
      qasd predict [model] [text…]         classify texts (arguments, else one per line from standard input)
      qasd info [model]                    labels and settings of a model
      qasd audit <data…>                   the labels' consistency: texts labelled differently in different rows, and
                                          short messages per label (often a reply that only the conversation explains);
                                          --out F.csv writes the conflicting texts with their label counts
    (model: a .qasd file; default apps/Qasd/models/intents.qasd)
      qasd split <data…>                   write the train / test split the classifier uses (train.csv, test.csv in
                                          apps/Qasd/data/split, or --out DIR), so a model tuned with idrak-tune on train.csv
                                          is scored on the same held-out messages (qasd evaluate --tuned …/test.csv)
      qasd benchmark <data…>               train and measure on each device (CPU, and CUDA when present): training time,
                                          held-out accuracy and F1, single-message latency and batch throughput
                                          (--devices cpu,cuda to choose)

    Data: a CSV, JSON Lines, JSON or Parquet file or folder, a URL, or a dataset spec as nsdata reads it
    (hf:owner/name?split=train, github:…); several sources are combined.

    Options:
      --text C            column with the text (default raw_question if present, else text)
      --label C           column with the label (default intent if present, else label)
      --device D          cpu | cuda | cuda:N | auto; --cpu / --cuda for short (default: train on the GPU when there is one,
                          evaluate and predict on the CPU; --cuda runs them on the GPU)
      --test-fraction F   train: distinct texts held out for the test (default 0.2; 0 = train on everything)
      --epochs N, --batch-size N, --buckets N (16384), --hidden N (256), --lr F (0.002), --seed N, --patience N (8)
      --tuned             evaluate, predict, info: the tuned chat model (apps/Qasd/models/tuned, or --folder DIR) instead of the
                          classifier; benchmark: measure it too (inference: accuracy, latency, first streamed token, throughput)
      --labels a,b,…      predict --tuned: the intents (default: the classifier's)
      --stream            predict --tuned: print the model's answer as it is generated
      --json              predict: one JSON object per text (label, confidence, probabilities)
      --min-confidence F  predict: print "unknown" below this confidence
      --errors F.csv      evaluate: write the texts classified wrong (label, prediction, confidence, every probability),
                          the most confident mistakes first
    """;

var positional = new List<string>();
string? output = null, textColumn = null, labelColumn = null, deviceName = null, deviceList = null;
double testFraction = 0.2, minConfidence = 0;
bool json = false, tuned = false, stream = false;
string? tunedFolder = null, labelList = null, errorsPath = null;
var options = new TextClassifierOptions();
try
{
    for (int i = 0; i < args.Length; i++)
    {
        string Next() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"{args[i]} needs a value.");
        int NextInt() => int.Parse(Next(), CultureInfo.InvariantCulture);
        switch (args[i])
        {
            case "--out" or "-o": output = Next(); break;
            case "--text": textColumn = Next(); break;
            case "--label": labelColumn = Next(); break;
            case "--device": deviceName = Next(); break;
            case "--devices": deviceList = Next(); break;
            case "--cpu": deviceName = "cpu"; break;
            case "--cuda" or "--gpu": deviceName = "cuda"; break;
            case "--test-fraction": testFraction = double.Parse(Next(), CultureInfo.InvariantCulture); break;
            case "--epochs": options = options with { Epochs = NextInt() }; break;
            case "--batch-size": options = options with { BatchSize = NextInt() }; break;
            case "--buckets": options = options with { Buckets = NextInt() }; break;
            case "--hidden": options = options with { Hidden = NextInt() }; break;
            case "--seed": options = options with { Seed = NextInt() }; break;
            case "--patience": options = options with { Patience = NextInt() }; break;
            case "--lr": options = options with { LearningRate = float.Parse(Next(), CultureInfo.InvariantCulture) }; break;
            case "--json": json = true; break;
            case "--tuned": tuned = true; break;
            case "--folder": tunedFolder = Next(); tuned = true; break;
            case "--labels": labelList = Next(); break;
            case "--stream": stream = true; break;
            case "--errors": errorsPath = Next(); break;
            case "--min-confidence": minConfidence = double.Parse(Next(), CultureInfo.InvariantCulture); break;
            case "-h" or "--help" or "help": Console.WriteLine(Usage); return 0;
            case ['-', '-', ..]: throw new ArgumentException($"Unknown option {args[i]}.");
            default: positional.Add(args[i]); break;
        }
    }
}
catch (Exception ex) when (ex is ArgumentException or FormatException)
{
    Console.Error.WriteLine(ex.Message);
    return 1;
}

string command = positional.Count > 0 ? positional[0] : "";
bool valid = command switch
{
    "train" or "split" or "audit" => positional.Count >= 2,
    "benchmark" => positional.Count >= 2,
    "evaluate" => positional.Count >= 2,
    "predict" or "info" => positional.Count >= 1,
    _ => false,
};
if (!valid)
{
    Console.WriteLine(Usage);
    return 1;
}

Console.OutputEncoding = Encoding.UTF8;
output ??= command switch
{
    "split" => Path.Combine(QasdPaths.Data, "split"),
    "audit" => null,                                                    // only when asked: never over the model file
    _ => QasdPaths.Classifier,
};
tunedFolder ??= QasdPaths.Tuned;

// evaluate / predict / info: the model file first when given (a .qasd file, or .nsm from before), else the default one.
bool modelGiven = positional.Count > 1 && (positional[1].EndsWith(".qasd", StringComparison.OrdinalIgnoreCase) || positional[1].EndsWith(".nsm", StringComparison.OrdinalIgnoreCase));
string modelPath = modelGiven ? positional[1] : QasdPaths.Classifier;
var rest = positional.Skip(modelGiven ? 2 : 1).ToList();
// JSON with Arabic and other text as written, not \u escapes (for people and pipelines, not HTML).
var jsonOutput = new System.Text.Json.JsonSerializerOptions { Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping };
try
{
    // Training on the GPU when there is one; evaluation and prediction on the CPU (fast enough for this model) unless asked.
    var device = ParseDevice(deviceName ?? (command == "train" ? "auto" : "cpu"));
    if (command == "train" && deviceName is null && device.Type == DeviceType.Cpu)
    {
        Console.WriteLine("no CUDA GPU found: training on the CPU");
    }

    switch (command)
    {
        case "train":
        {
            var examples = ReadAll(positional.Skip(1));
            var (train, test) = testFraction > 0 ? TextClassifier.Split(examples, testFraction, options.Seed + 7) : (examples, []);
            Console.WriteLine($"{examples.Count:N0} texts ({examples.Select(e => e.Text.Trim().ToLowerInvariant()).Distinct().Count():N0} distinct), labels: "
                              + string.Join(", ", examples.GroupBy(e => e.Label.Trim()).OrderBy(g => g.Key, StringComparer.Ordinal).Select(g => $"{g.Key} {g.Count():N0}")));
            Console.WriteLine($"training on {train.Count:N0} on {device.Name}{(test.Count > 0 ? $", testing on {test.Count:N0} (texts not in training)" : "")}");
            var clock = Stopwatch.StartNew();
            using var classifier = ConsoleTraining.Classifier(train, options with { Device = device });
            Console.WriteLine($"trained in {clock.Elapsed.TotalSeconds:F1} s (the best validation epoch, *, is kept)");
            if (test.Count > 0)
            {
                Console.WriteLine();
                Console.WriteLine(classifier.Evaluate(test));
            }

            classifier.Save(output!);
            Console.WriteLine($"saved {output} ({new FileInfo(output!).Length / 1048576.0:F1} MB)");
            return 0;
        }

        case "split":
        {
            var examples = ReadAll(positional.Skip(1));
            var (train, test) = TextClassifier.Split(examples, testFraction > 0 ? testFraction : 0.2, options.Seed + 7);
            Directory.CreateDirectory(output!);
            string text = textColumn ?? "raw_question", label = labelColumn ?? "intent";
            WriteCsv(Path.Combine(output!, "train.csv"), text, label, train);
            WriteCsv(Path.Combine(output!, "test.csv"), text, label, test);
            Console.WriteLine($"{examples.Count:N0} texts: {train.Count:N0} to train on, {test.Count:N0} held out (distinct texts, the split qasd train uses)");
            Console.WriteLine($"wrote {Path.GetFullPath(Path.Combine(output!, "train.csv"))} and test.csv ({text}, {label})");
            return 0;
        }

        case "evaluate" when tuned:
        {
            var examples = ReadAll(rest);
            using var model = TunedClassifier.Load(tunedFolder, [.. examples.Select(e => e.Label)], device);
            Console.WriteLine($"scoring {examples.Count:N0} messages with {model.BaseModel} (tuned) on {device.Name}…");
            var report = model.Evaluate(examples, ConsoleTraining.Scoring());
            Console.WriteLine(report);
            WriteErrors(report);
            return 0;
        }

        case "info" when tuned:
        {
            var manifest = TuningManifest.Read(tunedFolder) ?? throw new InvalidDataException($"{tunedFolder}: no tuned model.");
            Console.WriteLine($"{tunedFolder}: tuned from {manifest.BaseModel}, max length {manifest.MaxLength}, system prompt: {manifest.System ?? "(none)"}");
            return 0;
        }

        case "predict" when tuned:
        {
            IReadOnlyList<string> labels = labelList is not null
                ? labelList.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                : File.Exists(QasdPaths.Classifier) ? TextClassifier.Load(QasdPaths.Classifier, Device.Cpu) is var c ? Take(c) : []
                : throw new ArgumentException("predict --tuned needs the intents: --labels a,b,… (or a trained classifier to take them from).");
            using var model = TunedClassifier.Load(tunedFolder, labels, device);
            var texts = rest.Count > 0 ? rest : ReadLines().ToList();
            foreach (string text in texts)
            {
                TextPrediction prediction;
                if (stream)
                {
                    Console.Write($"{text}\n  answer: ");
                    TextPrediction? final = null;
                    await foreach (var item in model.StreamAsync(text))
                    {
                        if (item.Token is { } token)
                        {
                            Console.Write(token);
                        }

                        final = item.Prediction ?? final;
                    }

                    Console.WriteLine();
                    prediction = final!;
                }
                else
                {
                    prediction = model.Predict(text);
                }

                Print(text, prediction);
            }

            return 0;
        }

        case "evaluate":
        {
            using var classifier = TextClassifier.Load(modelPath, device);
            var examples = ReadAll(rest);
            var report = classifier.Evaluate(examples);
            Console.WriteLine(report);
            WriteErrors(report);
            return 0;
        }

        case "audit":
        {
            Audit(ReadAll(positional.Skip(1)));
            return 0;
        }

        case "benchmark":
        {
            var examples = ReadAll(positional.Skip(1));
            var (train, test) = TextClassifier.Split(examples, testFraction > 0 ? testFraction : 0.2, options.Seed + 7);
            var devices = (deviceList ?? (Device.IsCudaAvailable ? "cpu,cuda" : "cpu")).Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .Select(ParseDevice).ToList();
            Console.WriteLine($"{examples.Count:N0} texts: training on {train.Count:N0}, measuring on {test.Count:N0} held-out texts; same data, settings and seed on every device\n");
            var rows = new List<string[]>();
            var tunedRows = new List<string[]>();
            if (tuned && !TunedClassifier.IsTunedFolder(tunedFolder))
            {
                throw new InvalidDataException($"--tuned: no tuned model in {Path.GetFullPath(tunedFolder)} (tune one with idrak-tune, see the README).");
            }

            foreach (var target in devices)
            {
                Console.WriteLine($"{target.Name}:");
                var clock = Stopwatch.StartNew();
                int epochs = 0;
                using var classifier = ConsoleTraining.Classifier(train, options with { Device = target }, epochLines: false, label: "  training",
                    onEpoch: epoch => epochs = epoch.Epoch);
                double trainSeconds = clock.Elapsed.TotalSeconds;
                var report = classifier.Evaluate(test);

                // Single messages, one call each (a service answering one request at a time), after a warm-up.
                var texts = test.Select(e => e.Text).ToList();
                foreach (string text in texts.Take(20))
                {
                    classifier.Predict(text);
                }

                int singles = Math.Min(500, texts.Count);
                var latencies = new double[singles];
                for (int i = 0; i < singles; i++)
                {
                    var one = Stopwatch.StartNew();
                    classifier.Predict(texts[i % texts.Count]);
                    latencies[i] = one.Elapsed.TotalMilliseconds;
                }

                Array.Sort(latencies);

                // Batches of 256 (the batch endpoint), over all held-out texts, three times.
                classifier.Predict(texts.Take(256).ToList());
                var batch = Stopwatch.StartNew();
                int classified = 0;
                for (int round = 0; round < 3; round++)
                {
                    for (int first = 0; first < texts.Count; first += 256)
                    {
                        var chunk = texts.Skip(first).Take(256).ToList();
                        classifier.Predict(chunk);
                        classified += chunk.Count;
                    }
                }

                double throughput = classified / batch.Elapsed.TotalSeconds;
                double p50 = latencies[singles / 2], p95 = latencies[(int)(singles * 0.95)];
                Console.WriteLine($"  trained {epochs} epochs in {trainSeconds:F1} s; accuracy {report.Accuracy:P1}, macro F1 {report.MacroF1:F3}; "
                                  + $"one message p50 {p50:F2} ms, p95 {p95:F2} ms; batches {throughput:N0} messages/s");
                rows.Add([target.Name.Length > 34 ? target.Name[..34] : target.Name, $"{trainSeconds:F1} s", $"{epochs}",
                    $"{train.Count * (double)epochs / trainSeconds:N0}/s", $"{report.Accuracy:P1}", $"{report.MacroF1:F3}",
                    $"{p50:F2} ms", $"{p95:F2} ms", $"{throughput:N0}/s"]);
                if (tuned)
                {
                    tunedRows.Add(MeasureTuned(target, [.. test.Take(200)], classifier.Labels));
                }
            }

            string[] header = ["device", "train", "epochs", "train speed", "accuracy", "macro F1", "latency p50", "latency p95", "batch speed"];
            var widths = header.Select((h, c) => Math.Max(h.Length, rows.Max(r => r[c].Length)) + 2).ToArray();
            Console.WriteLine();
            Console.WriteLine(string.Concat(header.Select((h, c) => c == 0 ? h.PadRight(widths[c]) : h.PadLeft(widths[c]))));
            foreach (var row in rows)
            {
                Console.WriteLine(string.Concat(row.Select((v, c) => c == 0 ? v.PadRight(widths[c]) : v.PadLeft(widths[c]))));
            }

            Console.WriteLine("\ntrain speed: training texts per second (texts × epochs / time); latency: one Predict call; batch speed: Predict on 256 texts at a time.");
            if (tunedRows.Count > 0)
            {
                string[] tunedHeader = ["tuned model on", "accuracy", "macro F1", "latency p50", "latency p95", "first token", "batch speed"];
                var tunedWidths = tunedHeader.Select((h, c) => Math.Max(h.Length, tunedRows.Max(r => r[c].Length)) + 2).ToArray();
                Console.WriteLine();
                Console.WriteLine(string.Concat(tunedHeader.Select((h, c) => c == 0 ? h.PadRight(tunedWidths[c]) : h.PadLeft(tunedWidths[c]))));
                foreach (var row in tunedRows)
                {
                    Console.WriteLine(string.Concat(row.Select((v, c) => c == 0 ? v.PadRight(tunedWidths[c]) : v.PadLeft(tunedWidths[c]))));
                }

                Console.WriteLine("tuned: measured on up to 200 of the held-out texts (only if it was tuned on 'qasd split' train.csv are they unseen); "
                                  + "first token: until the first streamed token of its answer.");
            }

            return 0;
        }

        case "info":
        {
            using var classifier = TextClassifier.Load(modelPath, Device.Cpu);
            Console.WriteLine($"{modelPath}: {classifier.Labels.Count} labels: {string.Join(", ", classifier.Labels)}");
            return 0;
        }

        default:
        {
            using var classifier = TextClassifier.Load(modelPath, device);
            var texts = rest.Count > 0 ? rest : ReadLines().ToList();
            foreach (var (text, prediction) in texts.Zip(classifier.Predict(texts)))
            {
                Print(text, prediction);
            }

            return 0;
        }
    }
}
catch (Exception ex) when (ex is IOException or ArgumentException or InvalidDataException or HttpRequestException or InvalidOperationException
                               or NotSupportedException or FormatException or System.Text.Json.JsonException)
{
    Console.Error.WriteLine($"error: {ex.Message}");
    return 2;
}

// The tuned model's inference on the device: accuracy, latency of single messages, time to the first streamed token and
// batch throughput.
string[] MeasureTuned(Device target, List<LabeledText> probe, IReadOnlyList<string> labels)
{
    using var model = TunedClassifier.Load(tunedFolder, labels, target);
    var texts = probe.Select(e => e.Text).ToList();
    model.Predict([.. texts.Take(3)]);
    var report = model.Evaluate(probe);
    int singles = Math.Min(30, texts.Count);
    var latencies = new double[singles];
    for (int i = 0; i < singles; i++)
    {
        var one = Stopwatch.StartNew();
        model.Predict(texts[i]);
        latencies[i] = one.Elapsed.TotalMilliseconds;
    }

    Array.Sort(latencies);
    var firsts = new List<double>();
    foreach (string text in texts.Take(10))
    {
        var one = Stopwatch.StartNew();
        var enumerator = model.StreamAsync(text).GetAsyncEnumerator();
        try
        {
            if (enumerator.MoveNextAsync().AsTask().GetAwaiter().GetResult())
            {
                firsts.Add(one.Elapsed.TotalMilliseconds);
            }

            while (enumerator.MoveNextAsync().AsTask().GetAwaiter().GetResult())
            {
            }
        }
        finally
        {
            enumerator.DisposeAsync().AsTask().GetAwaiter().GetResult();
        }
    }

    firsts.Sort();
    var batch = Stopwatch.StartNew();
    model.Predict(texts);
    double throughput = texts.Count / batch.Elapsed.TotalSeconds;
    Console.WriteLine($"  tuned: accuracy {report.Accuracy:P1}, macro F1 {report.MacroF1:F3} on {probe.Count} texts; one message p50 {latencies[singles / 2]:F0} ms; {throughput:N1} messages/s");
    return [target.Name.Length > 34 ? target.Name[..34] : target.Name, $"{report.Accuracy:P1}", $"{report.MacroF1:F3}", $"{latencies[singles / 2]:F0} ms",
        $"{latencies[(int)(singles * 0.95)]:F0} ms", firsts.Count > 0 ? $"{firsts[firsts.Count / 2]:F0} ms" : "–", $"{throughput:N1}/s"];
}

// One prediction: a line, or a JSON object with --json; "unknown" below --min-confidence.
void Print(string text, TextPrediction prediction)
{
    string label = prediction.Confidence < minConfidence ? "unknown" : prediction.Label;
    if (json)
    {
        var probabilities = new JsonObject();
        foreach (var (name, probability) in prediction.Probabilities)
        {
            probabilities[name] = Math.Round(probability, 4);
        }

        Console.WriteLine(new JsonObject
        {
            ["text"] = text, ["label"] = label, ["confidence"] = Math.Round(prediction.Confidence, 4), ["probabilities"] = probabilities,
        }.ToJsonString(jsonOutput));
    }
    else
    {
        Console.WriteLine($"{label,-16}{prediction.Confidence,7:P1}  {text}");
    }
}

static string[] Take(TextClassifier classifier)
{
    using (classifier)
    {
        return [.. classifier.Labels];
    }
}

// Labeled texts as CSV (a header, then one quoted row per text).
// evaluate --errors: the texts classified wrong, with every probability, the most confident mistakes first.
void WriteErrors(TextClassifierReport report)
{
    if (errorsPath is null)
    {
        return;
    }

    static string Quote(string value) => "\"" + value.Replace("\"", "\"\"", StringComparison.Ordinal) + "\"";
    var errors = report.Errors().ToList();
    using (var writer = new StreamWriter(errorsPath, false, new UTF8Encoding(false)))
    {
        writer.WriteLine(string.Join(",", new[] { "text", "label", "predicted", "confidence" }.Concat(report.Labels).Select(Quote)));
        foreach (var (example, prediction) in errors)
        {
            var probabilities = report.Labels.Select(l => prediction.Probabilities.FirstOrDefault(p => p.Label == l).Probability.ToString("F4", CultureInfo.InvariantCulture));
            writer.WriteLine(string.Join(",", new[] { Quote(example.Text), Quote(example.Label.Trim()), Quote(prediction.Label),
                prediction.Confidence.ToString("F4", CultureInfo.InvariantCulture) }.Concat(probabilities)));
        }
    }

    Console.WriteLine($"{errors.Count:N0} texts classified wrong written to {Path.GetFullPath(errorsPath)} (the most confident first)");
}

// audit: labels given differently to the same text (after trimming, lower-casing and joining spaces), and the share of
// short messages per label: a two-word reply ("Tuesday", "yes", a phone number) is often only one intent in context. The
// ceiling is the accuracy a model that sees only the message could reach at best: each text's most common label.
void Audit(List<LabeledText> examples)
{
    static string Key(string text) => string.Join(' ', text.Trim().ToLowerInvariant().Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));
    var rows = examples.Select(e => (Key: Key(e.Text), Label: e.Label.Trim(), e.Text)).ToList();         // each key computed once
    var texts = rows.GroupBy(r => r.Key).Select(g => (g.Key, Rows: g.ToList(), Labels: g.GroupBy(r => r.Label).ToDictionary(l => l.Key, l => l.Count()))).ToList();
    var conflicts = texts.Where(t => t.Labels.Count > 1).OrderByDescending(t => t.Rows.Count).ToList();
    var conflicting = conflicts.Select(t => t.Key).ToHashSet(StringComparer.Ordinal);
    var majority = texts.ToDictionary(t => t.Key, t => t.Labels.MaxBy(l => l.Value).Key, StringComparer.Ordinal);
    var labels = rows.GroupBy(r => r.Label).OrderByDescending(g => g.Count()).ToList();
    int conflictRows = conflicts.Sum(t => t.Rows.Count);
    int reachable = rows.Count(r => majority[r.Key] == r.Label);
    Console.WriteLine($"{rows.Count:N0} rows, {texts.Count:N0} distinct texts, {labels.Count} labels");
    Console.WriteLine($"{conflicts.Count:N0} texts have more than one label ({conflictRows:N0} rows, {conflictRows / (double)Math.Max(1, rows.Count):P1}); "
                      + "the most frequent:");
    foreach (var t in conflicts.Take(20))
    {
        Console.WriteLine($"  {t.Key,-50} {string.Join(", ", t.Labels.OrderByDescending(l => l.Value).Select(l => $"{l.Key} {l.Value:N0}"))}");
    }

    Console.WriteLine();
    Console.WriteLine($"ceiling: a model that sees only the message is right on at most {reachable / (double)Math.Max(1, rows.Count):P1} of the rows "
                      + "(each text's most common label; the rest carry another label for the same text)");
    Console.WriteLine($"{"label",-16}{"rows",10}{"≤ 3 words",11}{"conflicting",13}{"max recall",12}   short examples");
    foreach (var label in labels)
    {
        int count = label.Count();
        int shortCount = label.Count(r => r.Key.Split(' ').Length <= 3);
        int inConflict = label.Count(r => conflicting.Contains(r.Key));
        int best = label.Count(r => majority[r.Key] == label.Key);
        string samples = string.Join(" | ", label.Where(r => r.Key.Split(' ').Length <= 3).Select(r => r.Text.Trim()).Distinct().Take(6));
        Console.WriteLine($"{label.Key,-16}{count,10:N0}{shortCount / (double)count,11:P1}{inConflict / (double)count,13:P1}{best / (double)count,12:P1}   {samples}");
    }

    if (output is not null)
    {
        static string Quote(string value) => "\"" + value.Replace("\"", "\"\"", StringComparison.Ordinal) + "\"";
        using var writer = new StreamWriter(output, false, new UTF8Encoding(false));
        writer.WriteLine(string.Join(",", new[] { "text", "rows" }.Concat(labels.Select(l => l.Key)).Select(Quote)));
        foreach (var t in conflicts)
        {
            writer.WriteLine(string.Join(",", new[] { Quote(t.Rows[0].Text.Trim()), t.Rows.Count.ToString(CultureInfo.InvariantCulture) }
                .Concat(labels.Select(l => t.Labels.GetValueOrDefault(l.Key).ToString(CultureInfo.InvariantCulture)))));
        }

        Console.WriteLine($"\n{conflicts.Count:N0} conflicting texts written to {Path.GetFullPath(output)}");
    }
}

static void WriteCsv(string path, string textColumn, string labelColumn, IEnumerable<LabeledText> rows)
{
    static string Quote(string value) => "\"" + value.Replace("\"", "\"\"", StringComparison.Ordinal) + "\"";
    using var writer = new StreamWriter(path, false, new UTF8Encoding(false));
    writer.WriteLine($"{Quote(textColumn)},{Quote(labelColumn)}");
    foreach (var row in rows)
    {
        writer.WriteLine($"{Quote(row.Text)},{Quote(row.Label)}");
    }
}

// Labeled texts from every source; the columns default to raw_question / intent when the data has them, else text / label.
List<LabeledText> ReadAll(IEnumerable<string> sources)
{
    var all = new List<LabeledText>();
    foreach (string source in sources)
    {
        var first = DatasetSpec.Parse(source).Open().FirstOrDefault()
                    ?? throw new InvalidDataException($"{source} has no rows.");
        string text = textColumn ?? (first.ContainsKey("raw_question") ? "raw_question" : "text");
        string label = labelColumn ?? (first.ContainsKey("intent") ? "intent" : "label");
        all.AddRange(TextClassifier.Read(source, text, label));
    }

    return all;
}

static Device ParseDevice(string name) => name.ToLowerInvariant() switch
{
    "auto" => Device.IsCudaAvailable ? Device.Cuda() : Device.Cpu,
    "cpu" => Device.Cpu,
    "cuda" or "gpu" => Device.Cuda(),
    ['c', 'u', 'd', 'a', ':', .. var index] => Device.Cuda(int.Parse(index, CultureInfo.InvariantCulture)),
    _ => throw new ArgumentException($"Unknown device '{name}' (auto, cpu, cuda or cuda:N)."),
};

static IEnumerable<string> ReadLines()
{
    while (Console.ReadLine() is { } line)
    {
        if (line.Trim().Length > 0)
        {
            yield return line;
        }
    }
}

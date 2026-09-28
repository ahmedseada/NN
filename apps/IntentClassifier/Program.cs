using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json.Nodes;
using NeuralSharp;
using IntentClassifier;
using NeuralSharp.Datasets;

// intent-classifier: trains and serves the intent model (an application built on NeuralSharp).
const string Usage = """
    intent-classifier: train, evaluate and run text classifiers (intents, topics, routing; any language)

      intent-classifier train <data…> --out model.nsm   train on labeled texts, report the score on held-out texts, save the model
      intent-classifier evaluate <model> <data…>        score a model on labeled texts (accuracy, per-label F1, confusion matrix)
      intent-classifier predict <model> [text…]         classify texts (arguments, else one per line from standard input)
      intent-classifier info <model>                    labels and settings of a model

    Data: a CSV, JSON Lines, JSON or Parquet file or folder, a URL, or a dataset spec as nsdata reads it
    (hf:owner/name?split=train, github:…); several sources are combined.

    Options:
      --text C            column with the text (default raw_question if present, else text)
      --label C           column with the label (default intent if present, else label)
      --device D          auto | cpu | cuda | cuda:N (default auto: the GPU when there is one)
      --test-fraction F   train: distinct texts held out for the test (default 0.2; 0 = train on everything)
      --epochs N, --batch-size N, --buckets N (16384), --hidden N (256), --lr F (0.002), --seed N, --patience N (8)
      --json              predict: one JSON object per text (label, confidence, probabilities)
      --min-confidence F  predict: print "unknown" below this confidence
    """;

var positional = new List<string>();
string? output = null, textColumn = null, labelColumn = null, deviceName = null;
double testFraction = 0.2, minConfidence = 0;
bool json = false;
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
    "train" => positional.Count >= 2 && output is not null,
    "evaluate" => positional.Count >= 3,
    "predict" or "info" => positional.Count >= 2,
    _ => false,
};
if (!valid)
{
    Console.WriteLine(Usage);
    return 1;
}

Console.OutputEncoding = Encoding.UTF8;
// JSON with Arabic and other text as written, not \u escapes (for people and pipelines, not HTML).
var jsonOutput = new System.Text.Json.JsonSerializerOptions { Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping };
try
{
    var device = deviceName switch
    {
        null or "auto" => Device.IsCudaAvailable ? Device.Cuda() : Device.Cpu,
        "cpu" => Device.Cpu,
        "cuda" or "gpu" => Device.Cuda(),
        ['c', 'u', 'd', 'a', ':', .. var index] => Device.Cuda(int.Parse(index, CultureInfo.InvariantCulture)),
        _ => throw new ArgumentException($"Unknown device '{deviceName}' (auto, cpu, cuda or cuda:N)."),
    };

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
            using var classifier = TextClassifier.Train(train, options with { Device = device },
                epoch => Console.WriteLine($"  epoch {epoch.Epoch,3}: loss {epoch.Loss:F4}"
                                           + (double.IsNaN(epoch.ValidationAccuracy) ? "" : $", validation accuracy {epoch.ValidationAccuracy:P1}") + (epoch.Best ? "  *" : "")));
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

        case "evaluate":
        {
            using var classifier = TextClassifier.Load(positional[1], device);
            var examples = ReadAll(positional.Skip(2));
            Console.WriteLine(classifier.Evaluate(examples));
            return 0;
        }

        case "info":
        {
            using var classifier = TextClassifier.Load(positional[1], Device.Cpu);
            Console.WriteLine($"{positional[1]}: {classifier.Labels.Count} labels: {string.Join(", ", classifier.Labels)}");
            return 0;
        }

        default:
        {
            using var classifier = TextClassifier.Load(positional[1], device);
            var texts = positional.Count > 2 ? positional.Skip(2).ToList() : ReadLines().ToList();
            foreach (var (text, prediction) in texts.Zip(classifier.Predict(texts)))
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

// Text classification with NeuralSharp.Text.TextClassifier: labeled texts in a CSV (any language: Arabic, English,
// mixed) → a model file you load in your application.
//
//   dotnet run -c Release --project samples/NeuralSharp.Samples.TextClassifier -- --data queries.csv --cuda
//        train (columns raw_question and intent by default), report the score on held-out texts, save the model
//   dotnet run -c Release --project samples/NeuralSharp.Samples.TextClassifier -- --predict --input "book me with Dr. Heba|what are your services"
//        classify texts (separated by |) with the saved model; without --input, one text per line from the console
//
// Options: --text-column raw_question, --label-column intent, --test-fraction 0.2, --buckets 16384, --hidden 256,
// --epochs 30, --batch-size 64, --model <file> (default models/text-classifier.nsm). Every copy of a text stays on one
// side of the split, so the test score is on texts the model never saw.

using System.Diagnostics;
using System.Globalization;
using System.Text;
using NeuralSharp;
using NeuralSharp.Samples;
using NeuralSharp.Text;

if (SampleOptions.Parse(args,
        ("text-column", "CSV column with the text (default raw_question)"),
        ("label-column", "CSV column with the label (default intent)"),
        ("test-fraction", "share of distinct texts held out for the test (default 0.2)"),
        ("buckets", "hashed feature buckets (default 16384)"),
        ("hidden", "hidden layer width (default 256)")) is not { } options)
{
    return 0;
}

Console.OutputEncoding = Encoding.UTF8;
var device = options.Device;
string modelPath = options.ModelPath("text-classifier.nsm");

if (options.PredictOnly)
{
    if (!options.RequireModel(modelPath))
    {
        return 1;
    }

    using var loaded = TextClassifier.Load(modelPath, device);
    Console.WriteLine($"Loaded {modelPath} on {device}: {string.Join(", ", loaded.Labels)}\n");
    var texts = options.Input is { } input
        ? input.Split('|', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
        : ReadLines().ToArray();
    foreach (var (text, prediction) in texts.Zip(loaded.Predict(texts)))
    {
        Console.WriteLine($"{prediction.Label,-16}{prediction.Confidence,7:P1}  {text}");
        Console.WriteLine($"{"",-23}{string.Join("  ", prediction.Probabilities.Select(p => $"{p.Label} {p.Probability:P0}"))}");
    }

    return 0;
}

if (options.DataFile is not { } dataFile || !File.Exists(dataFile))
{
    Console.Error.WriteLine("error: pass the labeled CSV with --data <file.csv>");
    return 1;
}

var examples = TextClassifier.ReadCsv(dataFile, options.Get("text-column", "raw_question")!, options.Get("label-column", "intent")!);
var (train, test) = TextClassifier.Split(examples, double.Parse(options.Get("test-fraction", "0.2")!, CultureInfo.InvariantCulture), seed: 7);
Console.WriteLine($"Device: {device} - {device.Name}");
Console.WriteLine($"{examples.Count} texts ({examples.Select(e => e.Text.Trim().ToLowerInvariant()).Distinct().Count()} distinct), labels: "
                  + string.Join(", ", examples.GroupBy(e => e.Label.Trim()).OrderBy(g => g.Key, StringComparer.Ordinal).Select(g => $"{g.Key} {g.Count()}")));
Console.WriteLine($"training on {train.Count}, testing on {test.Count} (texts not in training)\n");

var clock = Stopwatch.StartNew();
using var classifier = TextClassifier.Train(train, new TextClassifierOptions
{
    Device = device,
    Epochs = options.Epochs ?? 30,
    BatchSize = options.BatchSize ?? 64,
    Buckets = int.Parse(options.Get("buckets", "16384")!, CultureInfo.InvariantCulture),
    Hidden = int.Parse(options.Get("hidden", "256")!, CultureInfo.InvariantCulture),
    Seed = 1,
}, epoch => Console.WriteLine($"epoch {epoch.Epoch,3}: loss {epoch.Loss:F4}, validation accuracy {epoch.ValidationAccuracy:P1}{(epoch.Best ? "  *" : "")}"));
Console.WriteLine($"trained in {clock.Elapsed.TotalSeconds:F1} s (the best epoch, *, is kept)\n");

var report = classifier.Evaluate(test);
Console.WriteLine("Held-out test:");
Console.WriteLine(report);

var predictions = classifier.Predict([.. test.Select(e => e.Text)]);
var mistakes = test.Zip(predictions).Where(p => p.First.Label.Trim() != p.Second.Label).DistinctBy(p => p.First.Text).Take(10).ToList();
if (mistakes.Count > 0)
{
    Console.WriteLine("Some mistakes (actual → predicted):");
    foreach (var (example, prediction) in mistakes)
    {
        string text = example.Text.ReplaceLineEndings(" ");
        Console.WriteLine($"  {example.Label} → {prediction.Label} ({prediction.Confidence:P0}): {text[..Math.Min(90, text.Length)]}");
    }
}

classifier.Save(modelPath);
Console.WriteLine($"\nSaved {modelPath} ({new FileInfo(modelPath).Length / 1024.0 / 1024.0:F1} MB); try it with --predict");
return 0;

static IEnumerable<string> ReadLines()
{
    Console.WriteLine("Type a text per line (empty line to stop):");
    while (Console.ReadLine() is { Length: > 0 } line)
    {
        yield return line;
    }
}

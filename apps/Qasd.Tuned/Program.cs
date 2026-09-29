using System.Diagnostics;
using System.Globalization;
using System.Text;
using NeuralSharp;
using NeuralSharp.Datasets;
using Qasd;

// qasd-tuned: tunes a pretrained chat model to answer each message with its intent (LoRA), on the same data as qasd.
const string Usage = """
    qasd-tuned: tune a pretrained chat model to classify intents (the language-model counterpart of qasd)

      qasd-tuned train <data…> [--out F]        tune on labeled messages, report the score on held-out messages, save
                                                (default apps/Qasd.Tuned/models/qasd-tuned)
      qasd-tuned evaluate [folder] <data…>      score a tuned model on labeled messages
      qasd-tuned predict [folder] [text…]       classify messages (arguments, else one per line from standard input);
                                                --stream prints the model's answer as it is generated
      qasd-tuned benchmark <data…>              tune and measure on each device (--devices cpu,cuda; default: both when there is a GPU): tuning time,
                                                held-out accuracy and F1, latency, time to the first streamed token and
                                                throughput, next to the qasd classifier trained on the same split
                                                (--sample N: tune on N messages only, for a quick CPU run)

    Data: as qasd reads it (CSV, JSON Lines, Parquet, …); columns raw_question / intent (or text / label) by default.
    train holds out the same messages as 'qasd train' (same split and seed), so the two scores compare directly.

    Options:
      --model M           the pretrained chat model (default Qwen/Qwen2.5-0.5B-Instruct; a Hugging Face id, folder, .gguf or ollama:name)
      --cpu | --cuda      where to run (default: train on the GPU when there is one; evaluate and predict on the CPU, --cuda for the GPU)
      --text C, --label C columns with the message and the intent
      --test-fraction F   distinct messages held out (default 0.2; 0 = tune on everything)
      --held-out          evaluate: score only the messages train held out (same --test-fraction), not the whole data
      --epochs N (1), --rank N (16), --lr F (0.0002), --batch-tokens N (4096), --max-length N (256), --seed N
    """;

var positional = new List<string>();
string? output = null, textColumn = null, labelColumn = null, deviceList = null;
int sample = 0;
bool heldOut = false;
double testFraction = 0.2;
bool stream = false;
Device? device = null;
var options = new TunedOptions();
try
{
    for (int i = 0; i < args.Length; i++)
    {
        string Next() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"{args[i]} needs a value.");
        int NextInt() => int.Parse(Next(), CultureInfo.InvariantCulture);
        switch (args[i])
        {
            case "--out" or "-o": output = Next(); break;
            case "--model": options = options with { BaseModel = Next() }; break;
            case "--text": textColumn = Next(); break;
            case "--label": labelColumn = Next(); break;
            case "--cpu": device = Device.Cpu; break;
            case "--cuda" or "--gpu": device = Device.Cuda(); break;
            case "--test-fraction": testFraction = double.Parse(Next(), CultureInfo.InvariantCulture); break;
            case "--epochs": options = options with { Epochs = NextInt() }; break;
            case "--rank":
                int rank = NextInt();
                options = options with { Rank = rank, Alpha = 2f * rank };
                break;
            case "--lr": options = options with { LearningRate = float.Parse(Next(), CultureInfo.InvariantCulture) }; break;
            case "--batch-tokens": options = options with { BatchTokens = NextInt() }; break;
            case "--max-length": options = options with { MaxLength = NextInt() }; break;
            case "--seed": options = options with { Seed = NextInt() }; break;
            case "--stream": stream = true; break;
            case "--held-out": heldOut = true; break;
            case "--devices": deviceList = Next(); break;
            case "--sample": sample = NextInt(); break;
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
    "train" => positional.Count >= 2,
    "evaluate" => positional.Count >= 2,
    "predict" => positional.Count >= 1,
    "benchmark" => positional.Count >= 2,
    _ => false,
};
if (!valid)
{
    Console.WriteLine(Usage);
    return 1;
}

Console.OutputEncoding = Encoding.UTF8;

output ??= QasdPaths.Tuned;

// evaluate / predict: the tuned model's folder first when given, else the default one.
bool folderGiven = positional.Count > 1 && TunedClassifier.IsTunedFolder(positional[1]);
string folder = folderGiven ? positional[1] : QasdPaths.Tuned;
var rest = positional.Skip(folderGiven ? 2 : 1).ToList();

// Tuning on the GPU when there is one; evaluation and prediction on the CPU unless asked (--cuda).
if (device is null)
{
    device = command == "train" && Device.IsCudaAvailable ? Device.Cuda() : Device.Cpu;
    if (command == "train" && device.Type == DeviceType.Cpu)
    {
        Console.WriteLine("no CUDA GPU found: tuning on the CPU (slow for a language model)");
    }
}
try
{
    switch (command)
    {
        case "train":
        {
            var examples = ReadAll(positional.Skip(1));
            var (train, test) = testFraction > 0 ? TextClassifier.Split(examples, testFraction, 7) : (examples, []);
            Console.WriteLine($"{examples.Count:N0} messages: tuning on {train.Count:N0}{(test.Count > 0 ? $", testing on {test.Count:N0} held out (the same as qasd train)" : "")}");
            var clock = Stopwatch.StartNew();
            using var tuned = ConsoleTraining.Tuned(train, output!, options with { Device = device });
            Console.WriteLine($"tuned in {clock.Elapsed.TotalMinutes:F1} min; saved to {Path.GetFullPath(output!)}");
            if (test.Count > 0)
            {
                clock.Restart();
                Console.WriteLine($"scoring the {test.Count:N0} held-out messages…");
                var report = tuned.Evaluate(test, ConsoleTraining.Scoring());
                Console.WriteLine($"\n{report}");
                Console.WriteLine($"{test.Count / clock.Elapsed.TotalSeconds:F1} messages/s while scoring on {device.Name}");
            }

            return 0;
        }

        case "benchmark":
        {
            var examples = ReadAll(positional.Skip(1));
            var (train, test) = TextClassifier.Split(examples, testFraction > 0 ? testFraction : 0.2, 7);
            if (sample > 0 && sample < train.Count)
            {
                int step = train.Count / sample;
                train = [.. train.Where((_, i) => i % step == 0).Take(sample)];                  // spread over the data, the same every run
            }

            int measured = Math.Min(test.Count, 200);
            var probe = test.Take(measured).ToList();
            var devices = (deviceList ?? (Device.IsCudaAvailable ? "cpu,cuda" : "cpu")).Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .Select(d => d == "cpu" ? Device.Cpu : d is "cuda" or "gpu" ? Device.Cuda() : Device.Cuda(int.Parse(d[5..], CultureInfo.InvariantCulture))).ToList();
            Console.WriteLine($"{examples.Count:N0} messages; tuning on {train.Count:N0}, measuring on {probe.Count:N0} held-out messages (the split of qasd train)\n");
            var rows = new List<string[]>();
            foreach (var target in devices)
            {
                Console.WriteLine($"{target.Name}:");
                var clock = Stopwatch.StartNew();
                using var classifier = ConsoleTraining.Classifier(train, new TextClassifierOptions { Device = target, Seed = 1 }, epochLines: false,
                    label: "  training the classifier");
                double classifierTrain = clock.Elapsed.TotalSeconds;
                rows.Add(Measure("classifier", target, classifierTrain, classifier.Evaluate(probe), t => classifier.Predict(t), null, probe));

                string benchFolder = Path.Combine(Path.GetTempPath(), $"qasd-tuned-bench-{Guid.NewGuid():N}");
                try
                {
                    clock.Restart();
                    using var tuned = ConsoleTraining.Tuned(train, benchFolder, options with { Device = target }, "  tuning");
                    double tunedTrain = clock.Elapsed.TotalSeconds;
                    rows.Add(Measure("tuned", target, tunedTrain, tuned.Evaluate(probe), t => tuned.Predict(t), tuned, probe));
                }
                finally
                {
                    Directory.Delete(benchFolder, true);
                }
            }

            string[] header = ["model", "device", "train", "accuracy", "macro F1", "latency p50", "latency p95", "first token", "throughput"];
            var widths = header.Select((h, c) => Math.Max(h.Length, rows.Max(r => r[c].Length)) + 2).ToArray();
            Console.WriteLine();
            Console.WriteLine(string.Concat(header.Select((h, c) => c < 2 ? h.PadRight(widths[c]) : h.PadLeft(widths[c]))));
            foreach (var row in rows)
            {
                Console.WriteLine(string.Concat(row.Select((v, c) => c < 2 ? v.PadRight(widths[c]) : v.PadLeft(widths[c]))));
            }

            Console.WriteLine("\nlatency: one message per call; first token: until the first streamed token of the tuned model's answer; throughput: messages per second in batches.");
            return 0;
        }

        case "evaluate":
        {
            using var tuned = TunedClassifier.Load(folder, device);
            var examples = ReadAll(rest);
            if (heldOut)
            {
                // The messages train held out (same split and seed), for a score after tuning without scoring again there.
                examples = TextClassifier.Split(examples, testFraction > 0 ? testFraction : 0.2, 7).Test;
            }

            Console.WriteLine($"scoring {examples.Count:N0}{(heldOut ? " held-out" : "")} messages on {device.Name}…");
            Console.WriteLine(tuned.Evaluate(examples, ConsoleTraining.Scoring()));
            return 0;
        }

        default:
        {
            using var tuned = TunedClassifier.Load(folder, device);
            var texts = rest.Count > 0 ? rest : ReadLines().ToList();
            foreach (string text in texts)
            {
                if (stream)
                {
                    Console.Write($"{text}\n  answer: ");
                    await foreach (var item in tuned.StreamAsync(text))
                    {
                        if (item.Token is { } token)
                        {
                            Console.Write(token);
                        }
                        else if (item.Prediction is { } p)
                        {
                            Console.WriteLine($"\n  intent: {p.Label} {p.Confidence:P1}  ({string.Join(", ", p.Probabilities.Select(x => $"{x.Label} {x.Probability:P0}"))})");
                        }
                    }
                }
                else
                {
                    var p = tuned.Predict(text);
                    Console.WriteLine($"{p.Label,-16}{p.Confidence,7:P1}  {text}");
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

// Latency (one message per call, after a warm-up), time to the first streamed token (tuned) and batch throughput.
static string[] Measure(string name, Device device, double trainSeconds, TextClassifierReport report, Func<IReadOnlyList<string>, IReadOnlyList<TextPrediction>> predict,
    TunedClassifier? tuned, List<LabeledText> probe)
{
    var texts = probe.Select(e => e.Text).ToList();
    predict([.. texts.Take(3)]);
    int singles = Math.Min(tuned is null ? 300 : 30, texts.Count);
    var latencies = new double[singles];
    for (int i = 0; i < singles; i++)
    {
        var one = Stopwatch.StartNew();
        predict([texts[i]]);
        latencies[i] = one.Elapsed.TotalMilliseconds;
    }

    Array.Sort(latencies);
    string firstToken = "–";
    if (tuned is not null)
    {
        var firsts = new List<double>();
        foreach (string text in texts.Take(10))
        {
            var one = Stopwatch.StartNew();
            var enumerator = tuned.StreamAsync(text).GetAsyncEnumerator();
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
        firstToken = firsts.Count > 0 ? $"{firsts[firsts.Count / 2]:F0} ms" : "–";
    }

    var batch = Stopwatch.StartNew();
    predict(texts);
    double throughput = texts.Count / batch.Elapsed.TotalSeconds;
    Console.WriteLine($"  {name}: trained in {trainSeconds:F1} s; accuracy {report.Accuracy:P1}, macro F1 {report.MacroF1:F3}; latency p50 {latencies[singles / 2]:F2} ms");
    return [name, device.Name.Length > 30 ? device.Name[..30] : device.Name, trainSeconds >= 120 ? $"{trainSeconds / 60:F1} min" : $"{trainSeconds:F1} s",
        $"{report.Accuracy:P1}", $"{report.MacroF1:F3}", $"{latencies[singles / 2]:F2} ms", $"{latencies[(int)(singles * 0.95)]:F2} ms", firstToken, $"{throughput:N1}/s"];
}

List<LabeledText> ReadAll(IEnumerable<string> sources)
{
    var all = new List<LabeledText>();
    foreach (string source in sources)
    {
        var first = DatasetSpec.Parse(source).Open().FirstOrDefault() ?? throw new InvalidDataException($"{source} has no rows.");
        all.AddRange(TextClassifier.Read(source, textColumn ?? (first.ContainsKey("raw_question") ? "raw_question" : "text"),
            labelColumn ?? (first.ContainsKey("intent") ? "intent" : "label")));
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

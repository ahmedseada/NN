using System.Globalization;
using System.Text;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Datasets;
using Idrak.Inference;
using Idrak.Layers;
using Idrak.Optimizers;
using Idrak.Training;

namespace Qasd;

/// <summary>Settings of <see cref="TextClassifier.Train"/>.</summary>
public sealed record TextClassifierOptions
{
    /// <summary>Hashed feature buckets (word, word-pair and character n-grams share them). More: fewer collisions, a larger model.</summary>
    public int Buckets { get; init; } = 16384;

    /// <summary>Width of the hidden layer.</summary>
    public int Hidden { get; init; } = 256;

    /// <summary>Dropout after the hidden layer while training.</summary>
    public float Dropout { get; init; } = 0.3f;

    /// <summary>Most passes over the training texts (training stops earlier when validation stops improving).</summary>
    public int Epochs { get; init; } = 30;

    /// <summary>Epochs without a better validation score before training stops (0: never stop early).</summary>
    public int Patience { get; init; } = 8;

    /// <summary>Texts per optimizer step.</summary>
    public int BatchSize { get; init; } = 64;

    /// <summary>Peak learning rate (AdamW, cosine decay after a short warm-up).</summary>
    public float LearningRate { get; init; } = 2e-3f;

    /// <summary>AdamW weight decay.</summary>
    public float WeightDecay { get; init; } = 1e-4f;

    /// <summary>
    /// Share of the distinct texts kept aside to choose the best epoch (copies of a text stay together, so the score is
    /// on texts never trained on). 0: no validation, the last epoch is kept.
    /// </summary>
    public double ValidationFraction { get; init; } = 0.1;

    /// <summary>Weights each label by the square root of its inverse frequency, so rare labels are not drowned out.</summary>
    public bool BalanceLabels { get; init; } = true;

    /// <summary>Seed for the initial weights, the split and the batch order (the same data and seed give the same model).</summary>
    public int Seed { get; init; }

    /// <summary>Where the model trains (default: the GPU when there is one, else the CPU). <see cref="TextClassifier.Load"/> runs on the CPU unless told otherwise.</summary>
    public Device? Device { get; init; }
}

/// <summary>A labeled text.</summary>
public readonly record struct LabeledText(string Text, string Label);

/// <summary>A prediction: the most likely label, its probability, and every label's probability (most likely first).</summary>
public sealed record TextPrediction(string Label, float Confidence, IReadOnlyList<(string Label, float Probability)> Probabilities);

/// <summary>One epoch of training.</summary>
/// <param name="Epoch">1-based epoch.</param>
/// <param name="Loss">Mean training loss.</param>
/// <param name="ValidationAccuracy">Accuracy on the validation texts (NaN without validation).</param>
/// <param name="Best">Whether this epoch is the best so far (the one kept).</param>
public sealed record TextClassifierEpoch(int Epoch, double Loss, double ValidationAccuracy, bool Best);

/// <summary>One optimizer step while training a <see cref="TextClassifier"/>.</summary>
/// <param name="Step">1-based step across all epochs.</param>
/// <param name="TotalSteps">Steps if every epoch runs (training can stop earlier, see <see cref="TextClassifierOptions.Patience"/>).</param>
/// <param name="Epoch">1-based epoch.</param>
/// <param name="Loss">Mean training loss of the epoch so far.</param>
public sealed record TextClassifierStep(int Step, int TotalSteps, int Epoch, double Loss);

/// <summary>Precision, recall and F1 of one label.</summary>
public sealed record LabelScores(string Label, double Precision, double Recall, double F1, int Support);

/// <summary>What <see cref="TextClassifier.Evaluate"/> measured.</summary>
public sealed record TextClassifierReport(IReadOnlyList<string> Labels, int[,] Confusion, IReadOnlyList<LabelScores> Scores)
{
    /// <summary>Texts evaluated.</summary>
    public int Count => Scores.Sum(s => s.Support);

    /// <summary>Share classified right.</summary>
    public double Accuracy => Count == 0 ? 0 : Enumerable.Range(0, Labels.Count).Sum(i => Confusion[i, i]) / (double)Count;

    /// <summary>
    /// The report for <paramref name="predictions"/> of <paramref name="examples"/> (in the same order) over
    /// <paramref name="labels"/>; examples whose label is not one of them are left out.
    /// </summary>
    public static TextClassifierReport Create(IReadOnlyList<string> labels, IReadOnlyList<LabeledText> examples, IReadOnlyList<TextPrediction> predictions)
    {
        var index = labels.Select((l, i) => (l, i)).ToDictionary(p => p.l, p => p.i, StringComparer.Ordinal);
        int k = labels.Count;
        var confusion = new int[k, k];
        for (int i = 0; i < examples.Count; i++)
        {
            if (index.TryGetValue(examples[i].Label.Trim(), out int actual) && index.TryGetValue(predictions[i].Label, out int predicted))
            {
                confusion[actual, predicted]++;
            }
        }

        var scores = new List<LabelScores>();
        for (int c = 0; c < k; c++)
        {
            int predicted = Enumerable.Range(0, k).Sum(a => confusion[a, c]), actual = Enumerable.Range(0, k).Sum(p => confusion[c, p]);
            double precision = predicted == 0 ? 0 : confusion[c, c] / (double)predicted, recall = actual == 0 ? 0 : confusion[c, c] / (double)actual;
            scores.Add(new LabelScores(labels[c], precision, recall, precision + recall == 0 ? 0 : 2 * precision * recall / (precision + recall), actual));
        }

        return new TextClassifierReport(labels, confusion, scores) { Examples = examples, Predictions = predictions };
    }

    /// <summary>The evaluated texts, in order (empty for a report built otherwise).</summary>
    public IReadOnlyList<LabeledText> Examples { get; init; } = [];

    /// <summary>Their predictions, in the same order.</summary>
    public IReadOnlyList<TextPrediction> Predictions { get; init; } = [];

    /// <summary>The texts classified wrong, the most confident mistakes first (often labels worth a second look).</summary>
    public IEnumerable<(LabeledText Example, TextPrediction Prediction)> Errors() =>
        Examples.Zip(Predictions).Where(p => p.First.Label.Trim() != p.Second.Label).OrderByDescending(p => p.Second.Confidence);

    /// <summary>Mean F1 over the labels (each label counts the same, however rare).</summary>
    public double MacroF1 => Scores.Count == 0 ? 0 : Scores.Average(s => s.F1);

    /// <inheritdoc />
    public override string ToString()
    {
        int width = Math.Max(10, Labels.Max(l => l.Length) + 2);
        var sb = new StringBuilder();
        sb.AppendLine(CultureInfo.InvariantCulture, $"accuracy {Accuracy:P1} on {Count} texts, macro F1 {MacroF1:F3}");
        sb.AppendLine(CultureInfo.InvariantCulture, $"{"label".PadRight(width)}{"precision",10}{"recall",10}{"F1",8}{"texts",7}");
        foreach (var s in Scores)
        {
            sb.AppendLine(CultureInfo.InvariantCulture, $"{s.Label.PadRight(width)}{s.Precision,10:P1}{s.Recall,10:P1}{s.F1,8:F3}{s.Support,7}");
        }

        sb.AppendLine("confusion (rows: actual, columns: predicted)");
        sb.AppendLine("".PadRight(width) + string.Concat(Labels.Select(l => l.PadLeft(width))));
        for (int a = 0; a < Labels.Count; a++)
        {
            sb.AppendLine(Labels[a].PadRight(width) + string.Concat(Enumerable.Range(0, Labels.Count).Select(p => Confusion[a, p].ToString(CultureInfo.InvariantCulture).PadLeft(width))));
        }

        return sb.ToString();
    }
}

/// <summary>
/// A text classifier for short texts in any language (intents, topics, routing): each text becomes a hashed bag of
/// words, word pairs and character 2-4-grams (TF-IDF weighted; Arabic and Latin text normalized), and a small network
/// maps it to the labels. Trains in seconds to minutes on a GPU or CPU, from a few hundred labeled texts up; memory does
/// not grow with the data (features are built batch by batch). One file holds the whole model
/// (<see cref="Save"/> / <see cref="Load"/>, a <see cref="ModelPackage"/>). <see cref="Predict(string)"/> is safe to call from
/// several threads.
/// </summary>
/// <example>
/// <code>
/// var examples = TextClassifier.Read("queries.csv", textColumn: "raw_question", labelColumn: "intent");
/// var (train, test) = TextClassifier.Split(examples, 0.2);
/// using var classifier = TextClassifier.Train(train, new TextClassifierOptions { Device = Device.Cuda() });
/// Console.WriteLine(classifier.Evaluate(test));
/// classifier.Save("intents.qasd");
///
/// using var loaded = TextClassifier.Load("intents.qasd");
/// var prediction = loaded.Predict("book me with Dr. Heba on Tuesday");   // Label, Confidence, Probabilities
/// </code>
/// </example>
public sealed class TextClassifier : IDisposable
{
    private const string Format = "qasd-text-classifier/1";
    private readonly Sequential _model;
    private readonly TextFeatures _features;
    private readonly Lock _gate = new();
    private readonly int _hidden;
    private readonly float _dropout;

    private TextClassifier(Sequential model, TextFeatures features, string[] labels, int hidden, float dropout)
    {
        _model = model;
        _features = features;
        Labels = labels;
        _hidden = hidden;
        _dropout = dropout;
    }

    // A network trained elsewhere (LightTextClassifier) with the same layers: Linear, ReLU, Dropout, Linear.
    internal static TextClassifier FromTrained(Sequential model, TextFeatures features, string[] labels, int hidden, float dropout)
    {
        model.Eval();
        var classifier = new TextClassifier(model, features, labels, hidden, dropout);
        classifier.SnapshotWeights();
        return classifier;
    }

    /// <summary>The labels, in the order of <see cref="TextPrediction.Probabilities"/> before sorting.</summary>
    public IReadOnlyList<string> Labels { get; }

    /// <summary>The device the model runs on.</summary>
    public Device Device => _model.Parameters().First().Device;

    /// <summary>
    /// Trains a classifier on <paramref name="examples"/> (at least two labels). <paramref name="progress"/> receives
    /// each epoch and <paramref name="steps"/> each optimizer step. With validation (<see cref="TextClassifierOptions.ValidationFraction"/>), the best epoch's weights
    /// are kept.
    /// </summary>
    public static TextClassifier Train(IEnumerable<LabeledText> examples, TextClassifierOptions? options = null, Action<TextClassifierEpoch>? progress = null,
        Action<TextClassifierStep>? steps = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(examples);
        options ??= new TextClassifierOptions();
        var all = examples.Where(e => !string.IsNullOrWhiteSpace(e.Text) && !string.IsNullOrWhiteSpace(e.Label))
            .Select(e => new LabeledText(e.Text.Trim(), e.Label.Trim())).ToList();
        string[] labels = [.. all.Select(e => e.Label).Distinct().Order(StringComparer.Ordinal)];
        if (labels.Length < 2)
        {
            throw new ArgumentException($"Training needs texts of at least two labels; got {labels.Length}.", nameof(examples));
        }

        var device = options.Device ?? (Device.IsCudaAvailable ? Device.Cuda() : Device.Cpu);
        var (train, validation) = options.ValidationFraction > 0 ? Split(all, options.ValidationFraction, options.Seed + 1) : (all, []);
        if (train.Count == 0)
        {
            (train, validation) = (all, []);
        }

        var features = TextFeatures.Fit(train.Select(e => e.Text), options.Buckets);
        var model = Network(options.Buckets, options.Hidden, labels.Length, options.Dropout, options.Seed, device);
        var classifier = new TextClassifier(model, features, labels, options.Hidden, options.Dropout);
        try
        {
            classifier.Fit(train, validation, options, progress, steps, cancellationToken);
            return classifier;
        }
        catch
        {
            classifier.Dispose();
            throw;
        }
    }

    private void Fit(List<LabeledText> train, List<LabeledText> validation, TextClassifierOptions options, Action<TextClassifierEpoch>? progress,
        Action<TextClassifierStep>? steps, CancellationToken cancellationToken)
    {
        var index = Labels.Select((l, i) => (l, i)).ToDictionary(p => p.l, p => p.i, StringComparer.Ordinal);
        var weights = Labels.Select(l => options.BalanceLabels
            ? (float)Math.Sqrt(train.Count / (Labels.Count * (double)Math.Max(1, train.Count(e => e.Label == l))))
            : 1f).ToArray();
        int batchSize = Math.Max(1, options.BatchSize), buckets = _features.Buckets, classes = Labels.Count;
        int stepsPerEpoch = (train.Count + batchSize - 1) / batchSize;
        using var optimizer = new AdamW(_model.Parameters(), options.LearningRate, weightDecay: options.WeightDecay);
        var schedule = new CosineAnnealing(optimizer, Math.Max(1, options.Epochs * stepsPerEpoch), 0f, Math.Max(1, stepsPerEpoch));
        var random = new Random(options.Seed);
        var order = Enumerable.Range(0, train.Count).ToArray();
        var x = new float[batchSize * buckets];
        var y = new float[batchSize * classes];
        double bestScore = double.NegativeInfinity;
        byte[]? best = null;
        int sinceBest = 0, step = 0, totalSteps = Math.Max(1, options.Epochs) * stepsPerEpoch;
        for (int epoch = 1; epoch <= Math.Max(1, options.Epochs); epoch++)
        {
            random.Shuffle(order);
            _model.Train();
            double lossSum = 0;
            for (int first = 0; first < order.Length; first += batchSize)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int n = Math.Min(batchSize, order.Length - first);
                Array.Clear(y);
                for (int i = 0; i < n; i++)
                {
                    var example = train[order[first + i]];
                    _features.Write(example.Text, x.AsSpan(i * buckets, buckets));
                    int label = index[example.Label];
                    y[i * classes + label] = weights[label];
                }

                using var scope = new TensorScope();
                var inputs = Tensor.From(x.AsSpan(0, n * buckets).ToArray(), [n, buckets], Device);
                var targets = Tensor.From(y.AsSpan(0, n * classes).ToArray(), [n, classes], Device);
                optimizer.ZeroGrad();
                var loss = Losses.CrossEntropy(_model.Forward(inputs), targets);
                loss.Backward();
                optimizer.Step();
                schedule.Step();
                lossSum += loss.Item() * n;
                steps?.Invoke(new TextClassifierStep(++step, totalSteps, epoch, lossSum / (first + n)));
            }

            double accuracy = validation.Count > 0 ? Evaluate(validation).Accuracy : double.NaN;
            // Without validation every epoch is "best" (the last is kept); with it, ties keep the earlier epoch.
            double score = validation.Count > 0 ? accuracy : epoch;
            bool improved = score > bestScore;
            if (improved)
            {
                bestScore = score;
                sinceBest = 0;
                if (validation.Count > 0)
                {
                    using var copy = new MemoryStream();
                    _model.Save(copy);
                    best = copy.ToArray();
                }
            }
            else
            {
                sinceBest++;
            }

            progress?.Invoke(new TextClassifierEpoch(epoch, lossSum / Math.Max(1, train.Count), accuracy, improved));
            if (options.Patience > 0 && validation.Count > 0 && sinceBest >= options.Patience)
            {
                break;
            }
        }

        if (best is not null)
        {
            using var copy = new MemoryStream(best);
            _model.Load(copy);
        }

        _model.Eval();
        SnapshotWeights();
    }

    // CPU inference without the dense product: a text sets a few hundred of the buckets, so the first layer is the sum of
    // those rows of W1 (scaled by the feature values) instead of a mostly-zero [n, buckets] · [buckets, hidden] product.
    // Snapshot of the trained weights on the host; null while training (the weights still change) or on the GPU.
    private sealed record HostWeights(float[] W1, float[] B1, float[] W2, float[] B2, int Hidden, int Classes);

    private HostWeights? _host;

    private void SnapshotWeights()
    {
        _host = null;
        if (Device.Type != DeviceType.Cpu)
        {
            return;
        }

        var parameters = _model.Parameters().ToList();                    // W1 [buckets, hidden], b1, W2 [hidden, classes], b2
        _host = new HostWeights(parameters[0].ToArray(), parameters[1].ToArray(), parameters[2].ToArray(), parameters[3].ToArray(),
            parameters[0].Shape[1], parameters[2].Shape[1]);
    }

    private float[] HostProbabilities(HostWeights w, string text)
    {
        var hidden = (float[])w.B1.Clone();
        foreach (var (bucket, value) in _features.Sparse(text))
        {
            var row = w.W1.AsSpan(bucket * w.Hidden, w.Hidden);
            for (int j = 0; j < w.Hidden; j++)
            {
                hidden[j] += value * row[j];
            }
        }

        var logits = (float[])w.B2.Clone();
        for (int j = 0; j < w.Hidden; j++)
        {
            float h = MathF.Max(0f, hidden[j]);                          // ReLU (dropout is off at inference)
            if (h == 0f)
            {
                continue;
            }

            var row = w.W2.AsSpan(j * w.Classes, w.Classes);
            for (int c = 0; c < w.Classes; c++)
            {
                logits[c] += h * row[c];
            }
        }

        float max = logits.Max(), sum = 0f;
        for (int c = 0; c < logits.Length; c++)
        {
            logits[c] = MathF.Exp(logits[c] - max);
            sum += logits[c];
        }

        for (int c = 0; c < logits.Length; c++)
        {
            logits[c] /= sum;
        }

        return logits;
    }

    private TextPrediction Prediction(ReadOnlySpan<float> p)
    {
        var all = new List<(string Label, float Probability)>(p.Length);
        for (int c = 0; c < p.Length; c++)
        {
            all.Add((Labels[c], p[c]));
        }

        all.Sort((a, b) => b.Probability.CompareTo(a.Probability));
        return new TextPrediction(all[0].Label, all[0].Probability, all);
    }

    /// <summary>The label of <paramref name="text"/>, with probabilities.</summary>
    public TextPrediction Predict(string text) => Predict([text])[0];

    /// <summary>The labels of several texts at once (one pass through the model per 256 texts).</summary>
    public IReadOnlyList<TextPrediction> Predict(IReadOnlyList<string> texts)
    {
        ArgumentNullException.ThrowIfNull(texts);
        if (_host is { } host)
        {
            // The trained weights on the host: no lock needed, texts in parallel for larger batches.
            var predictions = new TextPrediction[texts.Count];
            if (texts.Count < 32)
            {
                for (int i = 0; i < texts.Count; i++)
                {
                    predictions[i] = Prediction(HostProbabilities(host, texts[i] ?? ""));
                }
            }
            else
            {
                Parallel.For(0, texts.Count, i => predictions[i] = Prediction(HostProbabilities(host, texts[i] ?? "")));
            }

            return predictions;
        }

        var results = new List<TextPrediction>(texts.Count);
        int buckets = _features.Buckets, classes = Labels.Count;
        const int Chunk = 256;
        lock (_gate)
        {
            _model.Eval();
            for (int first = 0; first < texts.Count; first += Chunk)
            {
                int n = Math.Min(Chunk, texts.Count - first);
                var x = new float[n * buckets];
                for (int i = 0; i < n; i++)
                {
                    _features.Write(texts[first + i] ?? "", x.AsSpan(i * buckets, buckets));
                }

                float[] p;
                using (Autograd.NoGrad())
                using (var scope = new TensorScope())
                {
                    p = _model.Forward(Tensor.From(x, [n, buckets], Device)).Softmax().ToArray();
                }

                for (int i = 0; i < n; i++)
                {
                    results.Add(Prediction(p.AsSpan(i * classes, classes)));
                }
            }
        }

        return results;
    }

    /// <summary>Accuracy, per-label precision / recall / F1 and the confusion matrix on labeled texts (labels the model does not know count as wrong).</summary>
    public TextClassifierReport Evaluate(IEnumerable<LabeledText> examples)
    {
        var list = examples.ToList();
        return TextClassifierReport.Create(Labels, list, Predict([.. list.Select(e => e.Text)]));
    }

    /// <summary>Writes the model (weights, labels, feature settings) to one file.</summary>
    public void Save(string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        var settings = new JsonObject
        {
            ["format"] = Format,
            ["labels"] = new JsonArray([.. Labels.Select(l => (JsonNode?)JsonValue.Create(l))]),
            ["buckets"] = _features.Buckets,
            ["hidden"] = _hidden,
            ["dropout"] = _dropout,
            ["idf"] = new JsonArray([.. _features.Idf.Select(v => (JsonNode?)JsonValue.Create(v))]),
        };
        lock (_gate)
        {
            ModelPackage.Create(path).Json("text-classifier", settings).Weights(_model).Save();
        }
    }

    /// <summary>Reads a model written by <see cref="Save"/>, onto <paramref name="device"/> (default: the CPU).</summary>
    public static TextClassifier Load(string path, Device? device = null)
    {
        using var package = ModelPackage.Open(path);
        var settings = package.Json("text-classifier").AsObject();
        if ((string?)settings["format"] != Format)
        {
            throw new InvalidDataException($"{path} is not a text classifier ({settings["format"]}).");
        }

        string[] labels = [.. settings["labels"]!.AsArray().Select(l => (string)l!)];
        int buckets = (int)settings["buckets"]!, hidden = (int)settings["hidden"]!;
        float dropout = (float)settings["dropout"]!;
        float[] idf = [.. settings["idf"]!.AsArray().Select(v => (float)v!)];
        var model = Network(buckets, hidden, labels.Length, dropout, 0, device ?? Device.Cpu);
        package.LoadWeights(model);
        model.Eval();
        var classifier = new TextClassifier(model, new TextFeatures(buckets, idf), labels, hidden, dropout);
        classifier.SnapshotWeights();
        return classifier;
    }

    /// <summary>
    /// Splits labeled texts so that every copy of a text (after normalization) lands on the same side: a test score on
    /// the second part is then a score on texts the model never saw. <paramref name="testFraction"/> of the distinct texts
    /// go to the second part.
    /// </summary>
    public static (List<LabeledText> Train, List<LabeledText> Test) Split(IEnumerable<LabeledText> examples, double testFraction, int seed = 0)
    {
        var random = new Random(seed);
        var groups = examples.GroupBy(e => TextFeatures.Normalize(e.Text)).OrderBy(g => g.Key, StringComparer.Ordinal).ToList();
        random.Shuffle(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(groups));
        int test = (int)Math.Round(groups.Count * Math.Clamp(testFraction, 0, 1));
        return ([.. groups.Skip(test).SelectMany(g => g)], [.. groups.Take(test).SelectMany(g => g)]);
    }

    /// <summary>
    /// Labeled texts from any dataset source Idrak.Datasets reads: a CSV, JSON Lines, JSON or Parquet file or folder,
    /// a URL, or a spec such as <c>hf:owner/name?split=train</c> (see <see cref="DatasetSpec"/>). Rows without a text or
    /// label are skipped; numbers and booleans are read as their text. Repeated rows (the same text, ignoring case and
    /// spacing, with the same label) are read once unless <paramref name="removeDuplicates"/> is false.
    /// </summary>
    public static List<LabeledText> Read(string source, string textColumn = "text", string labelColumn = "label", Downloader? downloader = null,
        bool removeDuplicates = true) =>
        Read([(source, textColumn, labelColumn)], removeDuplicates, out _, downloader);

    /// <summary>
    /// Labeled texts from several sources (each with its own text and label columns), read as one: with
    /// <paramref name="removeDuplicates"/> (the default) a row repeated in any of them is read once, by the library's
    /// <see cref="Idrak.Datasets.Dataset.Deduplicate"/> on the text and the label (ignoring case and spacing). A text with
    /// different labels keeps one row per label: that is a conflict to fix in the data (see qasd audit), not a repeat.
    /// <paramref name="duplicates"/> is how many rows were dropped.
    /// </summary>
    public static List<LabeledText> Read(IEnumerable<(string Source, string TextColumn, string LabelColumn)> sources, bool removeDuplicates,
        out int duplicates, Downloader? downloader = null)
    {
        int read = 0;
        var parts = sources.Select(s =>
        {
            bool checkedColumns = false;
            return DatasetSpec.Parse(s.Source).Open(downloader).Select(row =>
            {
                if (!checkedColumns)
                {
                    if (!row.ContainsKey(s.TextColumn) || !row.ContainsKey(s.LabelColumn))
                    {
                        throw new ArgumentException($"{s.Source} needs columns '{s.TextColumn}' and '{s.LabelColumn}'; its rows have {string.Join(", ", row.Select(p => p.Key))}.");
                    }

                    checkedColumns = true;
                }

                string? text = Cell(row[s.TextColumn]), label = Cell(row[s.LabelColumn]);
                if (string.IsNullOrWhiteSpace(text) || string.IsNullOrWhiteSpace(label))
                {
                    return null;
                }

                read++;
                return new JsonObject { ["text"] = text, ["label"] = label };
            });
        }).ToArray();

        var data = Idrak.Datasets.Dataset.Concat(parts);
        if (removeDuplicates)
        {
            data = data.Deduplicate(["text", "label"], normalize: true);
        }

        var examples = new List<LabeledText>();
        foreach (var row in data)
        {
            examples.Add(new LabeledText((string)row["text"]!, (string)row["label"]!));
        }

        duplicates = read - examples.Count;
        return examples;

        static string? Cell(JsonNode? node) => node switch
        {
            null => null,
            JsonValue value when value.TryGetValue<string>(out var text) => text,
            _ => node.ToJsonString(),
        };
    }

    /// <inheritdoc />
    public void Dispose() => _model.Dispose();

    private static Sequential Network(int buckets, int hidden, int classes, float dropout, int seed, Device device)
    {
        var init = new Random(seed + 3);
        return new Sequential
        {
            new Linear(buckets, hidden, device: device, random: init), new ReLU(), new Dropout(dropout, new Random(seed + 5)),
            new Linear(hidden, classes, device: device, random: init),
        };
    }
}

/// <summary>
/// Hashed n-gram features: words and word pairs, and character 2-4-grams of each word with its boundaries (spelling
/// variants, typos and Arabic prefixes still share most of them), plus the text's length in words. Counts are
/// log-scaled, weighted by inverse document frequency and L2-normalized. Hashing (FNV-1a) is the same on every machine.
/// </summary>
internal sealed class TextFeatures(int buckets, float[] idf)
{
    public int Buckets { get; } = buckets;

    public float[] Idf { get; } = idf.Length == buckets ? idf : throw new ArgumentException("One IDF weight per bucket.");

    public static TextFeatures Fit(IEnumerable<string> texts, int buckets)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(buckets, 16);
        var documents = new int[buckets];
        int count = 0;
        var seen = new HashSet<int>();
        foreach (string text in texts)
        {
            seen.Clear();
            foreach (int bucket in Hashes(text, buckets))
            {
                if (seen.Add(bucket))
                {
                    documents[bucket]++;
                }
            }

            count++;
        }

        return new TextFeatures(buckets, [.. documents.Select(d => (float)Math.Log((1.0 + count) / (1.0 + d)) + 1f)]);
    }

    /// <summary>The non-zero features of <paramref name="text"/> (bucket, value), with the values <see cref="Write"/> gives.</summary>
    public List<(int Bucket, float Value)> Sparse(string text)
    {
        var counts = new Dictionary<int, int>();
        foreach (int bucket in Hashes(text, Buckets))
        {
            counts[bucket] = counts.GetValueOrDefault(bucket) + 1;
        }

        var features = new List<(int Bucket, float Value)>(counts.Count);
        double norm = 0;
        foreach (var (bucket, count) in counts)
        {
            float value = (float)(Math.Log(1 + count) * Idf[bucket]);
            features.Add((bucket, value));
            norm += value * value;
        }

        float scale = norm > 0 ? (float)(1 / Math.Sqrt(norm)) : 0f;
        for (int i = 0; i < features.Count; i++)
        {
            features[i] = (features[i].Bucket, features[i].Value * scale);
        }

        return features;
    }

    public void Write(string text, Span<float> output)
    {
        output.Clear();
        foreach (int bucket in Hashes(text, Buckets))
        {
            output[bucket] += 1f;
        }

        double norm = 0;
        for (int i = 0; i < output.Length; i++)
        {
            if (output[i] != 0f)
            {
                output[i] = (float)(Math.Log(1 + output[i]) * Idf[i]);
                norm += output[i] * output[i];
            }
        }

        if (norm > 0)
        {
            float scale = (float)(1 / Math.Sqrt(norm));
            for (int i = 0; i < output.Length; i++)
            {
                output[i] *= scale;
            }
        }
    }

    /// <summary>
    /// Lower case; Arabic without diacritics and tatweel, with one form of alef, yaa and taa marbuta; Arabic-Indic digits
    /// as ASCII; punctuation as spaces; one space between words.
    /// </summary>
    public static string Normalize(string text)
    {
        var sb = new StringBuilder(text.Length);
        foreach (char raw in text.ToLowerInvariant())
        {
            char c = raw switch
            {
                'أ' or 'إ' or 'آ' or 'ٱ' => 'ا',
                'ى' => 'ي',
                'ة' => 'ه',
                >= '٠' and <= '٩' => (char)('0' + (raw - '٠')),
                >= '۰' and <= '۹' => (char)('0' + (raw - '۰')),
                _ => raw,
            };
            if (c is 'ـ' or (>= 'ً' and <= 'ٟ') or 'ٰ')
            {
                continue;
            }

            sb.Append(char.IsLetterOrDigit(c) ? c : ' ');
        }

        return string.Join(' ', sb.ToString().Split(' ', StringSplitOptions.RemoveEmptyEntries));
    }

    private static IEnumerable<int> Hashes(string text, int buckets)
    {
        var words = Normalize(text).Split(' ', StringSplitOptions.RemoveEmptyEntries);
        if (words.Length == 0)
        {
            yield return Bucket("<empty>", buckets);
            yield break;
        }

        for (int i = 0; i < words.Length; i++)
        {
            yield return Bucket("w:" + words[i], buckets);
            if (i + 1 < words.Length)
            {
                yield return Bucket("b:" + words[i] + " " + words[i + 1], buckets);
            }

            string padded = "<" + words[i] + ">";
            for (int n = 2; n <= 4; n++)
            {
                for (int start = 0; start + n <= padded.Length; start++)
                {
                    yield return Bucket("c:" + padded.Substring(start, n), buckets);
                }
            }
        }

        yield return Bucket("len:" + Math.Min(words.Length, 12).ToString(CultureInfo.InvariantCulture), buckets);
    }

    private static int Bucket(string feature, int buckets)
    {
        uint hash = 2166136261;
        foreach (char c in feature)
        {
            hash = (hash ^ c) * 16777619;
        }

        return (int)(hash % (uint)buckets);
    }
}

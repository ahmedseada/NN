using Idrak;
using Idrak.Data;
using Idrak.Diagnostics;
using Idrak.Layers;
using Idrak.Optimizers;
using Idrak.Training;

namespace Qasd;

/// <summary>
/// Settings of <see cref="LightTextClassifier.Train"/>: a much smaller network than <see cref="TextClassifierOptions"/>
/// (2,048 features and 64 hidden units instead of 16,384 and 256: about 130 thousand weights instead of 4.2 million), to
/// see how much of the large one's score is memorized.
/// </summary>
public sealed record LightTextClassifierOptions
{
    /// <summary>Hashed feature buckets (the input width). Every training text is held densely in memory: texts × buckets × 4 bytes.</summary>
    public int Buckets { get; init; } = 2048;

    /// <summary>Width of the hidden layer.</summary>
    public int Hidden { get; init; } = 64;

    /// <summary>Dropout after the hidden layer while training.</summary>
    public float Dropout { get; init; } = 0.3f;

    /// <summary>Most passes over the training texts.</summary>
    public int Epochs { get; init; } = 30;

    /// <summary>Epochs without a lower validation loss before training stops (the best epoch's weights are restored).</summary>
    public int Patience { get; init; } = 5;

    /// <summary>Texts per optimizer step.</summary>
    public int BatchSize { get; init; } = 256;

    /// <summary>AdamW learning rate.</summary>
    public float LearningRate { get; init; } = 2e-3f;

    /// <summary>AdamW weight decay.</summary>
    public float WeightDecay { get; init; } = 1e-4f;

    /// <summary>Share of the distinct training texts held out to pick the best epoch.</summary>
    public double ValidationFraction { get; init; } = 0.1;

    /// <summary>Weigh rare labels up (the square root of their inverse frequency), as the large trainer does.</summary>
    public bool BalanceLabels { get; init; } = true;

    /// <summary>Seed of the weights, the validation split and the batch order.</summary>
    public int Seed { get; init; }

    /// <summary>Where to train (null: the GPU when there is one, else the CPU).</summary>
    public Device? Device { get; init; }
}

/// <summary>
/// A second trainer for the same kind of model as <see cref="TextClassifier"/> (hashed n-grams, one hidden layer, softmax),
/// written with Idrak's simplified API: the network from the fluent builder, the training as one
/// <see cref="TrainingRun"/>. The result is a <see cref="TextClassifier"/>, saved in the same file format, so
/// <c>qasd predict</c>, <c>qasd evaluate</c> and the API load it as they load the large one.
/// </summary>
public static class LightTextClassifier
{
    /// <summary>
    /// Trains on <paramref name="examples"/>; <paramref name="onEpoch"/> receives each epoch (training and validation loss
    /// and accuracy: a widening gap between them is the network memorizing).
    /// </summary>
    public static TextClassifier Train(IReadOnlyList<LabeledText> examples, LightTextClassifierOptions? options = null,
        Action<EpochCompleted>? onEpoch = null, CancellationToken cancellationToken = default)
    {
        options ??= new LightTextClassifierOptions();
        var all = examples.Where(e => !string.IsNullOrWhiteSpace(e.Text) && !string.IsNullOrWhiteSpace(e.Label))
            .Select(e => new LabeledText(e.Text.Trim(), e.Label.Trim())).ToList();
        string[] labels = [.. all.Select(e => e.Label).Distinct().Order(StringComparer.Ordinal)];
        if (labels.Length < 2)
        {
            throw new ArgumentException($"Training needs texts of at least two labels; got {labels.Length}.", nameof(examples));
        }

        var device = options.Device ?? (Device.IsCudaAvailable ? Device.Cuda() : Device.Cpu);
        var (train, validation) = options.ValidationFraction > 0 ? TextClassifier.Split(all, options.ValidationFraction, options.Seed + 1) : (all, []);
        var features = TextFeatures.Fit(train.Select(e => e.Text), options.Buckets);
        float[] weights = [.. labels.Select(l => options.BalanceLabels
            ? (float)Math.Sqrt(train.Count / (labels.Length * (double)Math.Max(1, train.Count(e => e.Label == l))))
            : 1f)];

        var model = Network.Input(options.Buckets).OnDevice(device).Seed(options.Seed)
            .Linear(options.Hidden).ReLU().Dropout(options.Dropout)
            .Linear(labels.Length)
            .Build();
        try
        {
            new TrainingRun
            {
                Model = model, Loss = Losses.CrossEntropy, Optimizer = p => new AdamW(p, options.LearningRate, weightDecay: options.WeightDecay),
                Train = Encode(train, features, labels, weights).Batches(options.BatchSize, shuffle: true, device: device, seed: options.Seed),
                Validation = validation.Count > 0 ? Encode(validation, features, labels, weights).Batches(1024, device: device) : null,
                Epochs = Math.Max(1, options.Epochs), Metrics = [Metric.Accuracy],
                EarlyStoppingPatience = validation.Count > 0 && options.Patience > 0 ? options.Patience : null,
                OnEpoch = onEpoch,
            }.Fit(cancellationToken);
            return TextClassifier.FromTrained(model, features, labels, options.Hidden, options.Dropout);
        }
        catch
        {
            model.Dispose();
            throw;
        }
    }

    // The texts as dense feature rows and one-hot targets (the label's weight in place of 1, as the large trainer does).
    private static Dataset Encode(List<LabeledText> texts, TextFeatures features, string[] labels, float[] weights)
    {
        int buckets = features.Buckets, classes = labels.Length;
        if ((long)texts.Count * buckets > Array.MaxLength)
        {
            throw new InvalidOperationException($"{texts.Count:N0} texts × {buckets:N0} buckets do not fit in one array: use fewer --buckets.");
        }

        var index = labels.Select((l, i) => (l, i)).ToDictionary(p => p.l, p => p.i, StringComparer.Ordinal);
        var x = new float[texts.Count * buckets];
        var y = new float[texts.Count * classes];
        Parallel.For(0, texts.Count, i =>
        {
            features.Write(texts[i].Text, x.AsSpan(i * buckets, buckets));
            int label = index[texts[i].Label];
            y[i * classes + label] = weights[label];
        });
        return Dataset.FromFlat(x, y, texts.Count, [.. Enumerable.Range(0, buckets).Select(i => $"f{i}")], labels);
    }
}

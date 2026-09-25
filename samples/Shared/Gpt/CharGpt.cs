using System.Diagnostics;
using System.Text;
using System.Text.Json;
using NeuralSharp.Layers;

namespace NeuralSharp.Samples.Gpt;

/// <summary>Architecture and training record of a character-level GPT, saved next to its weights as JSON.</summary>
public sealed record GptConfig(string Vocabulary, int Context = 64, int Dim = 96, int Heads = 4, int Layers = 3)
{
    /// <summary>Epochs trained, final validation loss/accuracy and when (filled in after training).</summary>
    public int TrainedEpochs { get; init; }

    public double? ValidationLoss { get; init; }

    public double? ValidationAccuracy { get; init; }

    public DateTimeOffset? TrainedAt { get; init; }

    public string? Corpus { get; init; }

    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true };

    public static string ConfigPath(string weightsPath) => Path.ChangeExtension(weightsPath, ".json");

    public void Save(string weightsPath) => File.WriteAllText(ConfigPath(weightsPath), JsonSerializer.Serialize(this, Json));

    public static GptConfig Load(string weightsPath) =>
        JsonSerializer.Deserialize<GptConfig>(File.ReadAllText(ConfigPath(weightsPath)), Json)
        ?? throw new InvalidDataException($"Invalid model config {ConfigPath(weightsPath)}.");
}

/// <summary>Sampling settings for <see cref="CharGpt.Generate"/>.</summary>
/// <param name="Length">Characters to generate.</param>
/// <param name="Temperature">Softmax temperature: below 1 is more conservative, above 1 more random.</param>
/// <param name="TopK">Sample only among the k most likely characters (0 = all).</param>
/// <param name="Seed">Random seed for reproducible output (null = random).</param>
public sealed record GenerationSettings(int Length = 200, float Temperature = 0.7f, int TopK = 0, int? Seed = null);

/// <summary>A likely alternative for one position.</summary>
public sealed record Alternative(string Token, float Probability);

/// <summary>One generated character with its probability, the distribution's entropy (bits), the top alternatives and its latency.</summary>
public sealed record GeneratedToken(int Index, string Token, float Probability, float Entropy, IReadOnlyList<Alternative> Alternatives, double LatencyMs);

/// <summary>Aggregate inference metrics.</summary>
public sealed record GenerationMetrics(
    string Device,
    int PromptTokens,
    int GeneratedTokens,
    double TotalMs,
    double FirstTokenMs,
    double MsPerToken,
    double TokensPerSecond,
    double AverageProbability,
    double Perplexity,
    double AverageEntropyBits,
    long MemoryInUseBytes,
    long MemoryCachedBytes);

/// <summary>The output of a generation call.</summary>
public sealed record GenerationResult(string Prompt, string Text, IReadOnlyList<GeneratedToken> Tokens, GenerationMetrics Metrics);

/// <summary>
/// A decoder-only character-level transformer: embeddings + sinusoidal positions → causal transformer blocks →
/// per-position next-character logits. Shared by the Transformer console sample and the GPT Web API.
/// </summary>
public sealed class CharGpt : IDisposable
{
    private readonly Dictionary<char, int> _index;

    private CharGpt(GptConfig config, Sequential model, Device device)
    {
        Config = config;
        Model = model;
        Device = device;
        _index = config.Vocabulary.Select((c, i) => (c, i)).ToDictionary(p => p.c, p => p.i);
    }

    public GptConfig Config { get; private set; }

    public Sequential Model { get; }

    public Device Device { get; private set; }

    /// <summary>Builds an untrained model.</summary>
    public static CharGpt Create(GptConfig config, Device device, Random? random = null)
    {
        random ??= new Random(2);
        var model = new Sequential
        {
            new Embedding(config.Vocabulary.Length, config.Dim, device, random),
            new PositionalEncoding(config.Context, config.Dim, device),
        };
        for (int layer = 0; layer < config.Layers; layer++)
        {
            model.Add(new TransformerEncoderLayer(config.Dim, config.Heads, ffDim: 4 * config.Dim, dropout: 0.1f, causal: true, device: device, random: random));
        }

        model.Add(new LayerNorm(config.Dim, device: device));
        model.Add(new Linear(config.Dim, config.Vocabulary.Length, device: device, random: random));
        model.Name = "char-gpt";
        return new CharGpt(config, model, device);
    }

    /// <summary>Loads weights and their JSON config.</summary>
    public static CharGpt Load(string weightsPath, Device device)
    {
        var gpt = Create(GptConfig.Load(weightsPath), device);
        gpt.Model.Load(weightsPath);
        return gpt;
    }

    public void Save(string weightsPath, GptConfig? config = null)
    {
        Config = config ?? Config;
        Model.Save(weightsPath);
        Config.Save(weightsPath);
    }

    /// <summary>Moves the weights to another device.</summary>
    public void MoveTo(Device device)
    {
        if (device != Device)
        {
            Model.To(device);
            Device = device;
        }
    }

    /// <summary>Encodes text as token ids, dropping characters outside the vocabulary.</summary>
    public List<int> Encode(string text) => [.. text.Where(_index.ContainsKey).Select(c => _index[c])];

    /// <summary>
    /// Generates text autoregressively: each step feeds the last Context characters, turns the final position's
    /// logits into probabilities (temperature, optional top-k), samples one character and reports it.
    /// </summary>
    public GenerationResult Generate(string prompt, GenerationSettings settings, Action<GeneratedToken>? onToken = null, CancellationToken cancellationToken = default)
    {
        var random = settings.Seed is { } seed ? new Random(seed) : new Random();
        var ids = Encode(prompt);
        int promptTokens = ids.Count;
        if (ids.Count == 0)
        {
            ids.Add(_index.GetValueOrDefault(' '));
        }

        var tokens = new List<GeneratedToken>(settings.Length);
        var text = new StringBuilder();
        var total = Stopwatch.StartNew();
        double firstTokenMs = 0, logProbabilitySum = 0, probabilitySum = 0, entropySum = 0;
        float inverseTemperature = 1f / Math.Max(settings.Temperature, 1e-3f);
        for (int i = 0; i < settings.Length && !cancellationToken.IsCancellationRequested; i++)
        {
            long start = Stopwatch.GetTimestamp();
            float[] probabilities;
            using (var scope = new TensorScope())
            {
                var window = ids.Skip(Math.Max(0, ids.Count - Config.Context)).Select(id => (float)id).ToArray();
                var input = Tensor.From(window, [1, window.Length], Device);
                var logits = Model.Predict(input);
                probabilities = (logits.Narrow(1, window.Length - 1, 1).Reshape(-1) * inverseTemperature).Softmax().ToArray();
            }

            if (settings.TopK > 0 && settings.TopK < probabilities.Length)
            {
                float cutoff = probabilities.OrderDescending().ElementAt(settings.TopK - 1);
                float kept = 0;
                for (int c = 0; c < probabilities.Length; c++)
                {
                    probabilities[c] = probabilities[c] >= cutoff ? probabilities[c] : 0f;
                    kept += probabilities[c];
                }

                for (int c = 0; c < probabilities.Length; c++)
                {
                    probabilities[c] /= kept;
                }
            }

            int next = Sample(probabilities, random);
            ids.Add(next);
            double entropy = -probabilities.Where(p => p > 0).Sum(p => p * Math.Log2(p));
            var alternatives = probabilities.Select((p, c) => new Alternative(Config.Vocabulary[c].ToString(), p))
                .OrderByDescending(a => a.Probability).Take(5).ToList();
            double latency = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            if (i == 0)
            {
                firstTokenMs = latency;
            }

            var token = new GeneratedToken(i, Config.Vocabulary[next].ToString(), probabilities[next], (float)entropy, alternatives, latency);
            tokens.Add(token);
            text.Append(token.Token);
            logProbabilitySum += Math.Log(Math.Max(probabilities[next], 1e-12f));
            probabilitySum += probabilities[next];
            entropySum += entropy;
            onToken?.Invoke(token);
        }

        total.Stop();
        int n = Math.Max(tokens.Count, 1);
        var memory = ComputeResources.GetMemoryUsage(Device);
        var metrics = new GenerationMetrics(
            $"{Device} ({Device.Name})", promptTokens, tokens.Count, total.Elapsed.TotalMilliseconds, firstTokenMs,
            total.Elapsed.TotalMilliseconds / n, tokens.Count / Math.Max(total.Elapsed.TotalSeconds, 1e-9),
            probabilitySum / n, Math.Exp(-logProbabilitySum / n), entropySum / n, memory.InUse, memory.Cached);
        return new GenerationResult(prompt, text.ToString(), tokens, metrics);
    }

    private static int Sample(float[] probabilities, Random random)
    {
        double roll = random.NextDouble(), cumulative = 0;
        for (int c = 0; c < probabilities.Length; c++)
        {
            cumulative += probabilities[c];
            if (roll < cumulative)
            {
                return c;
            }
        }

        return Array.FindLastIndex(probabilities, p => p > 0);
    }

    public void Dispose() => Model.Dispose();
}

/// <summary>A tiny English-like grammar, so generated text can be checked word by word.</summary>
public static class Grammar
{
    private static readonly string[] Subjects = ["the little robot", "a curious cat", "the old wizard", "my neighbor", "the young pilot", "a quiet student", "the tired farmer", "our teacher"];
    private static readonly string[] Verbs = ["found", "built", "painted", "carried", "watched", "repaired", "opened", "forgot", "sold", "borrowed"];
    private static readonly string[] Objects = ["a shiny key", "the broken clock", "an old map", "a tiny garden", "the red door", "a wooden boat", "the heavy box", "a strange letter"];
    private static readonly string[] Places = ["near the river", "in the tower", "behind the market", "under the bridge", "at the station", "on the hill"];
    private static readonly string[] Times = ["every morning", "last night", "before dawn", "after lunch", "on sunday", "in the spring"];

    /// <summary>Every word the grammar can produce.</summary>
    public static readonly HashSet<string> Words =
        [.. new[] { Subjects, Verbs, Objects, Places, Times }.SelectMany(p => p).SelectMany(p => p.Split(' ')).Append("and").Append("then")];

    /// <summary>Random sentences, five per line.</summary>
    public static string Corpus(int sentences, int seed)
    {
        var random = new Random(seed);
        string Pick(string[] options) => options[random.Next(options.Length)];
        var text = new StringBuilder();
        for (int i = 0; i < sentences; i++)
        {
            text.Append(random.Next(4) switch
            {
                0 => $"{Pick(Times)}, {Pick(Subjects)} {Pick(Verbs)} {Pick(Objects)}.",
                1 => $"{Pick(Subjects)} {Pick(Verbs)} {Pick(Objects)} and then {Pick(Verbs)} {Pick(Objects)}.",
                _ => $"{Pick(Subjects)} {Pick(Verbs)} {Pick(Objects)} {Pick(Places)}.",
            });
            text.Append(i % 5 == 4 ? '\n' : ' ');
        }

        return text.ToString();
    }

    /// <summary>Fraction of complete generated words that are grammar words.</summary>
    public static (int Valid, int Total) Score(string generated)
    {
        var words = generated.Split([' ', '.', ',', '\n'], StringSplitOptions.RemoveEmptyEntries).SkipLast(1).ToList();
        return (words.Count(Words.Contains), words.Count);
    }
}

/// <summary>Builds training windows (inputs and next-character targets) and trains a <see cref="CharGpt"/>.</summary>
public static class GptTraining
{
    public static Data.Dataset Windows(string corpus, GptConfig config, int maxWindows, int seed)
    {
        var index = config.Vocabulary.Select((c, i) => (c, i)).ToDictionary(p => p.c, p => p.i);
        var random = new Random(seed);
        int context = config.Context;
        int windows = Math.Min(maxWindows, Math.Max(1, corpus.Length / 4));
        var features = new float[windows * context];
        var targets = new float[windows * context];
        for (int w = 0; w < windows; w++)
        {
            int start = random.Next(corpus.Length - context - 1);
            for (int t = 0; t < context; t++)
            {
                features[w * context + t] = index[corpus[start + t]];
                targets[w * context + t] = index[corpus[start + t + 1]];
            }
        }

        string[] positions = [.. Enumerable.Range(0, context).Select(t => $"t{t}")];
        return Data.Dataset.FromFlat(features, targets, windows, positions, positions);
    }

    /// <summary>Trains with AdamW, cosine schedule and gradient clipping; returns the config with training results.</summary>
    public static GptConfig Train(CharGpt gpt, string corpus, int epochs, int batchSize, string corpusName, CancellationToken cancellationToken = default)
    {
        var data = Windows(corpus, gpt.Config, maxWindows: 20_000, seed: 3);
        var (train, validation) = data.Split(0.95, seed: 4);
        using var optimizer = new Optimizers.AdamW(gpt.Model.Parameters(), learningRate: 0.002f, weightDecay: 0.01f);
        var trainer = new Training.Trainer(gpt.Model, optimizer, (logits, next) => Losses.SparseCrossEntropy(logits, next))
        {
            Metrics = { Training.Metric.SparseAccuracy },
            Scheduler = new Optimizers.CosineAnnealing(optimizer, epochs, minLearningRate: 2e-4f),
            MaxGradientNorm = 1f,
        };
        var history = trainer.Fit(new Data.DataLoader(train, batchSize, shuffle: true, device: gpt.Device, seed: 5), epochs,
            validation: new Data.DataLoader(validation, 256, device: gpt.Device), cancellationToken);
        var last = history.Epochs[^1];
        return gpt.Config with
        {
            TrainedEpochs = history.Epochs.Count,
            ValidationLoss = last.ValidationLoss,
            ValidationAccuracy = last.ValidationMetrics?["accuracy"],
            TrainedAt = DateTimeOffset.UtcNow,
            Corpus = corpusName,
        };
    }
}

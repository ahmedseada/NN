using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;
using NeuralSharp;
using NeuralSharp.Generation;
using NeuralSharp.Layers;
using NeuralSharp.Pretrained;

namespace Qasd;

/// <summary>Settings of <see cref="TunedClassifier.Train"/>.</summary>
public sealed record TunedOptions
{
    /// <summary>The pretrained chat model to tune: a Hugging Face id, a folder, a .gguf file or ollama:name.</summary>
    public string BaseModel { get; init; } = "Qwen/Qwen2.5-0.5B-Instruct";

    /// <summary>Where the model trains (default: the GPU when there is one, else the CPU, which is slow for a language model). <see cref="TunedClassifier.Load"/> runs on the CPU unless told otherwise.</summary>
    public Device? Device { get; init; }

    /// <summary>LoRA rank.</summary>
    public int Rank { get; init; } = 16;

    /// <summary>LoRA alpha (the adapters' output is scaled by alpha / rank).</summary>
    public float Alpha { get; init; } = 32f;

    /// <summary>Peak learning rate.</summary>
    public float LearningRate { get; init; } = 2e-4f;

    /// <summary>Passes over the training messages.</summary>
    public int Epochs { get; init; } = 1;

    /// <summary>Tokens per training batch.</summary>
    public int BatchTokens { get; init; } = 4096;

    /// <summary>Longest message in tokens (with the instruction and the answer); longer ones are cut.</summary>
    public int MaxLength { get; init; } = 256;

    /// <summary>Seed for the adapters and the batch order.</summary>
    public int Seed { get; init; }

    /// <summary>The instruction before each message (null: one listing the intents).</summary>
    public string? SystemPrompt { get; init; }
}

/// <summary>
/// An intent classifier made by fine-tuning a small pretrained chat model (LoRA) to answer each message with its intent:
/// the same labeled data and the same kind of result as <see cref="TextClassifier"/> (a label, its confidence and every
/// label's probability). Predictions score each intent as the model's answer (the probability of its tokens, then a
/// softmax over the intents), so the label is always one of the trained ones; <see cref="StreamAsync"/> also streams the
/// model's own answer token by token. A tuned model is a folder: the PEFT adapter plus qasd-tuned.json (base model,
/// intents, instruction).
/// </summary>
public sealed class TunedClassifier : IDisposable
{
    private const string Format = "qasd-tuned/1";
    private const string SettingsFile = "qasd-tuned.json";
    private readonly PretrainedModel _model;
    private readonly ChatTranscriptEncoder _encoder;
    private readonly string _system;
    private readonly int _maxLength;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private ChatGenerator? _chat;

    private TunedClassifier(PretrainedModel model, string[] labels, string system, int maxLength, string baseModel, string folder)
    {
        _model = model;
        _encoder = new ChatTranscriptEncoder(model.ChatTemplate ?? throw new InvalidDataException($"{baseModel} has no chat template."),
            model.Tokenizer ?? throw new InvalidDataException($"{baseModel} has no tokenizer."));
        Labels = labels;
        _system = system;
        _maxLength = maxLength;
        BaseModel = baseModel;
        Folder = folder;
    }

    /// <summary>The intents, in the order of the probabilities before sorting.</summary>
    public IReadOnlyList<string> Labels { get; }

    /// <summary>The pretrained model that was tuned.</summary>
    public string BaseModel { get; }

    /// <summary>The folder holding the adapter and settings.</summary>
    public string Folder { get; }

    /// <summary>The device the model runs on.</summary>
    public Device Device => _model.Device;

    /// <summary>
    /// Tunes <see cref="TunedOptions.BaseModel"/> on <paramref name="train"/> (each message answered with its label), writes
    /// the adapter and settings to <paramref name="outputFolder"/> and returns the tuned model, ready to predict.
    /// <paramref name="log"/> receives progress lines, and <paramref name="steps"/> each optimizer step (when given, the steps
    /// are not logged).
    /// </summary>
    public static TunedClassifier Train(IReadOnlyList<LabeledText> train, string outputFolder, TunedOptions? options = null, Action<string>? log = null,
        Action<FineTuningProgress>? steps = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(train);
        options ??= new TunedOptions();
        string[] labels = [.. train.Select(e => e.Label.Trim()).Where(l => l.Length > 0).Distinct().Order(StringComparer.Ordinal)];
        if (labels.Length < 2)
        {
            throw new ArgumentException($"Tuning needs messages of at least two intents; got {labels.Length}.", nameof(train));
        }

        string system = options.SystemPrompt ?? DefaultSystem(labels);
        var device = options.Device ?? (Device.IsCudaAvailable ? Device.Cuda() : Device.Cpu);
        string folder = ModelSource.Resolve(options.BaseModel);
        log?.Invoke($"base model {options.BaseModel} ({folder}) on {device.Name}");
        var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device, BFloat16 = device.Type == DeviceType.Cuda });
        Directory.CreateDirectory(outputFolder);
        var classifier = new TunedClassifier(model, labels, system, options.MaxLength, options.BaseModel, Path.GetFullPath(outputFolder));
        try
        {
            var sequences = train.Select(e => classifier.Sequence(e.Text, e.Label.Trim())).OfType<TrainingSequence>().ToList();
            log?.Invoke($"{sequences.Count:N0} training conversations, {sequences.Sum(s => s.Tokens.Length):N0} tokens");
            FineTuner.Train(model, sequences, null, new FineTuningOptions
            {
                Rank = options.Rank,
                Alpha = options.Alpha,
                LearningRate = options.LearningRate,
                Epochs = options.Epochs,
                BatchTokens = options.BatchTokens,
                MaxLength = options.MaxLength,
                Seed = options.Seed,
            }, outputFolder, new SynchronousProgress(steps ?? (p => log?.Invoke(
                $"step {p.Step}/{p.TotalSteps}: loss {p.Loss:F4}, {p.TokensPerSecond:N0} tokens/s"))), cancellationToken);
            File.WriteAllText(Path.Combine(outputFolder, SettingsFile), new JsonObject
            {
                ["format"] = Format,
                ["baseModel"] = options.BaseModel,
                ["labels"] = new JsonArray([.. labels.Select(l => (JsonNode?)JsonValue.Create(l))]),
                ["system"] = system,
                ["maxLength"] = options.MaxLength,
            }.ToJsonString(new System.Text.Json.JsonSerializerOptions { WriteIndented = true, Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping }));
        }
        catch
        {
            classifier.Dispose();
            throw;
        }

        // Training leaves tensors to the garbage collector (which does not see device memory): free them, then load the
        // saved model with the adapter merged, as it is served, so scoring has the device to itself.
        classifier.Dispose();
        FreeDeviceMemory(device);
        log?.Invoke("loading the tuned model with the adapter merged");
        return Load(outputFolder, device);
    }

    private static void FreeDeviceMemory(Device device)
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        ComputeResources.ReleaseCachedMemory(device);
    }

    /// <summary>Loads a tuned model folder written by <see cref="Train"/> (the adapter merged into the base weights, for speed).</summary>
    public static TunedClassifier Load(string folder, Device? device = null)
    {
        var settings = JsonNode.Parse(File.ReadAllText(Path.Combine(folder, SettingsFile)))!.AsObject();
        if ((string?)settings["format"] != Format)
        {
            throw new InvalidDataException($"{folder} is not a tuned Qasd model ({settings["format"]}).");
        }

        string baseModel = (string)settings["baseModel"]!;
        device ??= Device.Cpu;
        var model = PretrainedModel.Load(ModelSource.Resolve(baseModel), new PretrainedOptions
        {
            Device = device, BFloat16 = device.Type == DeviceType.Cuda, MergeAdapter = folder,
        });
        model.Network.Eval();
        return new TunedClassifier(model, [.. settings["labels"]!.AsArray().Select(l => (string)l!)], (string)settings["system"]!,
            (int)settings["maxLength"]!, baseModel, Path.GetFullPath(folder));
    }

    /// <summary>Whether <paramref name="folder"/> holds a tuned model.</summary>
    public static bool IsTunedFolder(string folder) => File.Exists(Path.Combine(folder, SettingsFile));

    /// <summary>The intent of <paramref name="text"/>, with every intent's probability.</summary>
    public TextPrediction Predict(string text) => Predict([text])[0];

    /// <summary>The intents of several messages (each intent scored as the answer to each message).</summary>
    public IReadOnlyList<TextPrediction> Predict(IReadOnlyList<string> texts)
    {
        ArgumentNullException.ThrowIfNull(texts);
        _gate.Wait();
        try
        {
            return Score(texts);
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <summary>Accuracy, per-intent scores and the confusion matrix on labeled messages.</summary>
    public TextClassifierReport Evaluate(IEnumerable<LabeledText> examples, Action<int, int>? progress = null)
    {
        var list = examples.ToList();
        _gate.Wait();
        try
        {
            var predictions = Score([.. list.Select(e => e.Text)], done => progress?.Invoke(done, list.Count));
            return TextClassifierReport.Create(Labels, list, predictions);
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <summary>
    /// The model's own answer to <paramref name="text"/>, streamed as it is generated (greedy): text pieces, then the scored
    /// prediction (the same as <see cref="Predict(string)"/>) as the final item.
    /// </summary>
    public async IAsyncEnumerable<TunedStreamItem> StreamAsync(string text, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            _chat ??= _model.CreateChat(contextLength: Math.Min(_model.MaxPositions, _maxLength + 32));
            var request = new ChatRequest([new ChatMessage("system", _system), new ChatMessage("user", Fitted(text))], null, false,
                new GenerationOptions { Temperature = 0f, TopK = 1, TopP = 1f, RepeatPenalty = 1f, NumPredict = 16, ChunkSize = 1, NumCtx = Math.Min(_model.MaxPositions, _maxLength + 32) });
            await foreach (var chunk in _chat.StreamAsync(request, cancellationToken).ConfigureAwait(false))
            {
                if (chunk.Delta.Content.Length > 0)
                {
                    yield return new TunedStreamItem(chunk.Delta.Content, null);
                }
            }

            yield return new TunedStreamItem(null, Score(text));
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        _model.Dispose();
        _gate.Dispose();
    }

    private static string DefaultSystem(IEnumerable<string> labels) =>
        $"Classify the user's message into one intent: {string.Join(", ", labels)}. Answer with the intent only.";

    // The message as it fits the model (see Sequence), for generation.
    private string Fitted(string text) => Fit(text, Labels[0]).Text;

    // Messages are cut to leave room for the instruction and the answer within MaxLength tokens.
    private string Clip(string text) => text.Length > 4 * _maxLength ? text[..(4 * _maxLength)] : text;

    // The conversation (instruction, message, answer = label) as the training loss and the scoring see it.
    private TrainingSequence? Sequence(string text, string label) => Fit(text, label).Sequence;

    // A message too long for MaxLength tokens is shortened (its start kept) until the whole conversation fits.
    private (string Text, TrainingSequence? Sequence) Fit(string text, string label)
    {
        text = Clip(text);
        for (int attempt = 0; attempt < 40 && text.Length > 0; attempt++)
        {
            var sequence = _encoder.Encode(new ChatTranscript(
                [new ChatMessage("system", _system), new ChatMessage("user", text), new ChatMessage("assistant", label)], [], Think: false), _maxLength);
            if (sequence is not null && sequence.Trained.Skip(1).Any(t => t))
            {
                return (text, sequence);
            }

            text = text[..(text.Length * 3 / 4)];
        }

        return (text, null);
    }

    private TextPrediction Score(string text) => Score([text])[0];

    // log p(label's answer tokens | instruction, message) for each intent, softmaxed over the intents. A batch holds a row
    // per (message, intent), right-padded; the network runs up to its final normalization and the output layer only on the
    // answer rows. Messages go 16 at a time (64 rows with four intents).
    private List<TextPrediction> Score(IReadOnlyList<string> texts, Action<int>? progress = null)
    {
        const int MessagesPerPass = 16;
        string longest = Labels.MaxBy(l => _model.Tokenizer!.Encode(l).Count)!;
        var predictions = new List<TextPrediction>(texts.Count);
        var modules = _model.Network.ToList();
        var head = (Linear)modules[^1];
        for (int first = 0; first < texts.Count; first += MessagesPerPass)
        {
            var chunk = texts.Skip(first).Take(MessagesPerPass).ToList();

            // One shortened message for every intent: fitted to the longest answer, so each intent is scored on the same text.
            var sequences = new List<TrainingSequence?>();
            foreach (string text in chunk)
            {
                string fitted = Fit(text, longest).Text;
                sequences.AddRange(Labels.Select(l => Fit(fitted, l).Sequence));
            }

            int rows = sequences.Count, length = sequences.Max(s => s?.Tokens.Length ?? 2) - 1;
            var input = new float[rows * length];
            var targets = new List<(int Row, int Position, int Token)>();
            for (int r = 0; r < rows; r++)
            {
                if (sequences[r] is not { } s)
                {
                    continue;
                }

                for (int t = 0; t + 1 < s.Tokens.Length; t++)
                {
                    input[r * length + t] = s.Tokens[t];
                    if (s.Trained[t + 1])
                    {
                        targets.Add((r, t, s.Tokens[t + 1]));
                    }
                }
            }

            var scores = new double[rows];
            try
            {
                Pass();
            }
            catch (ResourceLimitExceededException)
            {
                // Memory the garbage collector has not returned yet: collect it and try the pass once more.
                Array.Clear(scores);
                FreeDeviceMemory(Device);
                Pass();
            }

            void Pass()
            {
                using var noGrad = Autograd.NoGrad();
                using var scope = new TensorScope();
                var hidden = Tensor.From(input, [rows, length], Device);
                for (int i = 0; i < modules.Count - 1; i++)
                {
                    hidden = modules[i].Forward(hidden);
                }

                int dim = hidden.Shape[^1];
                var all = hidden.ToArray();
                var picked = new float[targets.Count * dim];
                for (int i = 0; i < targets.Count; i++)
                {
                    Array.Copy(all, (targets[i].Row * length + targets[i].Position) * dim, picked, i * dim, dim);
                }

                var logits = head.Forward(Tensor.From(picked, [targets.Count, dim], Device)).ToArray();
                int vocabulary = logits.Length / targets.Count;
                for (int i = 0; i < targets.Count; i++)
                {
                    var row = logits.AsSpan(i * vocabulary, vocabulary);
                    float max = float.NegativeInfinity;
                    foreach (float v in row)
                    {
                        max = MathF.Max(max, v);
                    }

                    double sum = 0;
                    foreach (float v in row)
                    {
                        sum += Math.Exp(v - max);
                    }

                    scores[targets[i].Row] += row[targets[i].Token] - max - Math.Log(sum);    // log p(token)
                }
            }

            for (int m = 0; m < chunk.Count; m++)
            {
                var own = scores.AsSpan(m * Labels.Count, Labels.Count).ToArray();
                if (sequences.Skip(m * Labels.Count).Take(Labels.Count).Any(s => s is null))
                {
                    // Not even an empty message fits: MaxLength is too small for the instruction; no intent can be preferred.
                    float even = 1f / Labels.Count;
                    predictions.Add(new TextPrediction(Labels[0], even, [.. Labels.Select(l => (l, even))]));
                    continue;
                }

                double best = own.Max();
                var weights = own.Select(x => Math.Exp(x - best)).ToArray();
                double total = weights.Sum();
                var probabilities = Labels.Select((l, i) => (Label: l, Probability: (float)(weights[i] / total))).OrderByDescending(p => p.Probability).ToList();
                predictions.Add(new TextPrediction(probabilities[0].Label, probabilities[0].Probability, probabilities));
            }

            progress?.Invoke(predictions.Count);
        }

        return predictions;
    }

    private sealed class SynchronousProgress(Action<FineTuningProgress> report) : IProgress<FineTuningProgress>
    {
        public void Report(FineTuningProgress value) => report(value);
    }
}

/// <summary>One item of <see cref="TunedClassifier.StreamAsync"/>: a piece of the generated answer, or the final prediction.</summary>
/// <param name="Token">Text the model generated (null on the final item).</param>
/// <param name="Prediction">The scored prediction (only on the final item).</param>
public sealed record TunedStreamItem(string? Token, TextPrediction? Prediction);

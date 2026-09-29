using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;
using NeuralSharp;
using NeuralSharp.Generation;
using NeuralSharp.Layers;
using NeuralSharp.Pretrained;

namespace Qasd;

/// <summary>
/// The intent classifier over a chat model tuned to answer each message with its intent (tuned with nstune, the
/// NeuralSharp fine-tuning tool, on the intent data; see the README): the same kind of result as
/// <see cref="TextClassifier"/> (a label, its confidence and every label's probability). Predictions score each intent as
/// the model's answer (the probability of its tokens, then a softmax over the intents), so the label is always one of the
/// intents; <see cref="StreamAsync"/> also streams the model's own answer token by token.
/// </summary>
public sealed class TunedClassifier : IDisposable
{
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
    /// Loads a tuned adapter folder (written by nstune train: the adapter and neuralsharp-tuning.json, which names the base
    /// model and the system prompt it was tuned with), the adapter merged into the base weights for speed, to classify into
    /// <paramref name="labels"/> (the intents it was tuned to answer with).
    /// </summary>
    public static TunedClassifier Load(string folder, IReadOnlyList<string> labels, Device? device = null)
    {
        UpgradeLegacy(folder);
        var manifest = TuningManifest.Read(folder)
                       ?? throw new InvalidDataException($"{folder} is not a tuned model folder (no {TuningManifest.FileName}; tune one with nstune train).");
        string[] intents = [.. labels.Select(l => l.Trim()).Where(l => l.Length > 0).Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal)];
        if (intents.Length < 2)
        {
            throw new ArgumentException("The tuned model needs at least two intents.", nameof(labels));
        }

        var model = manifest.LoadModel(folder, device ?? Device.Cpu);
        return new TunedClassifier(model, intents, manifest.System ?? "", manifest.MaxLength, manifest.BaseModel, Path.GetFullPath(folder));
    }

    /// <summary>Whether <paramref name="folder"/> holds a tuned model.</summary>
    public static bool IsTunedFolder(string folder) =>
        Directory.Exists(folder) && (TuningManifest.Exists(folder) || File.Exists(Path.Combine(folder, LegacySettings)));

    // Folders from the former qasd-tuned tool keep their settings in qasd-tuned.json: the manifest nstune writes is added
    // from it once (the base model, instruction and length they were tuned with).
    private const string LegacySettings = "qasd-tuned.json";

    private static void UpgradeLegacy(string folder)
    {
        string legacy = Path.Combine(folder, LegacySettings);
        if (TuningManifest.Exists(folder) || !File.Exists(legacy))
        {
            return;
        }

        var settings = JsonNode.Parse(File.ReadAllText(legacy))!.AsObject();
        new TuningManifest
        {
            BaseModel = (string)settings["baseModel"]!, System = (string?)settings["system"], MaxLength = (int?)settings["maxLength"] ?? 256,
        }.Save(folder);
    }

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
            var request = new ChatRequest(Transcript(Fitted(text), "").Messages.SkipLast(1).ToList(), null, false,
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

    // The message as it fits the model (see Sequence), for generation.
    private string Fitted(string text) => Fit(text, Labels[0]).Text;

    // The conversation (instruction, message, answer = label) as the training saw it (nstune with the same system prompt).
    private ChatTranscript Transcript(string text, string label)
    {
        var messages = new List<ChatMessage>();
        if (_system.Length > 0)
        {
            messages.Add(new ChatMessage("system", _system));
        }

        messages.Add(new ChatMessage("user", text));
        messages.Add(new ChatMessage("assistant", label));
        return new ChatTranscript(messages, [], Think: false);
    }

    // A message too long for MaxLength tokens is shortened, its start kept, until the whole conversation fits (see
    // ChatTranscriptEncoder.Fit); every intent is then scored on the same text (fitted to the longest intent).
    private (string Text, TrainingSequence? Sequence) Fit(string text, string label)
    {
        text = text.Length > 8 * _maxLength ? text[..(8 * _maxLength)] : text;       // far longer than any fit
        var fitted = _encoder.Fit(Transcript(text, label), _maxLength);
        if (fitted is null)
        {
            return ("", null);
        }

        return (fitted.Messages[^2].Content, _encoder.Encode(fitted, _maxLength));
    }

    private TextPrediction Score(string text) => Score([text])[0];

    // log p(label's answer tokens | instruction, message) for each intent, softmaxed over the intents. A batch holds a row
    // per (message, intent), right-padded; the network runs up to its final normalization and the output layer only on the
    // answer rows. Messages go 16 at a time (64 rows with four intents), those of similar length together (less padding);
    // the network frees each layer's intermediate results as it goes (see Sequential.ForwardFirst).
    private List<TextPrediction> Score(IReadOnlyList<string> texts, Action<int>? progress = null)
    {
        const int MessagesPerPass = 16;
        string longest = Labels.MaxBy(l => _model.Tokenizer!.Encode(l).Count)!;
        var predictions = new TextPrediction[texts.Count];
        var order = Enumerable.Range(0, texts.Count).OrderBy(i => texts[i].Length).ToArray();
        var modules = _model.Network.ToList();
        var head = (Linear)modules[^1];
        for (int first = 0; first < texts.Count; first += MessagesPerPass)
        {
            var indices = order.AsSpan(first, Math.Min(MessagesPerPass, texts.Count - first)).ToArray();
            var chunk = indices.Select(i => texts[i]).ToList();

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
            using (Autograd.NoGrad())
            using (var scope = new TensorScope())
            {
                var hidden = _model.Network.ForwardFirst(Tensor.From(input, [rows, length], Device), modules.Count - 1);

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
                    predictions[indices[m]] = new TextPrediction(Labels[0], even, [.. Labels.Select(l => (l, even))]);
                    continue;
                }

                double best = own.Max();
                var weights = own.Select(x => Math.Exp(x - best)).ToArray();
                double total = weights.Sum();
                var probabilities = Labels.Select((l, i) => (Label: l, Probability: (float)(weights[i] / total))).OrderByDescending(p => p.Probability).ToList();
                predictions[indices[m]] = new TextPrediction(probabilities[0].Label, probabilities[0].Probability, probabilities);
            }

            progress?.Invoke(first + chunk.Count);
        }

        return [.. predictions];
    }
}

/// <summary>One item of <see cref="TunedClassifier.StreamAsync"/>: a piece of the generated answer, or the final prediction.</summary>
/// <param name="Token">Text the model generated (null on the final item).</param>
/// <param name="Prediction">The scored prediction (only on the final item).</param>
public sealed record TunedStreamItem(string? Token, TextPrediction? Prediction);

using System.Runtime.CompilerServices;
using Idrak;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.LanguageModels;

namespace Qasd;

/// <summary>
/// The intent classifier over a chat model tuned to answer each message with its intent (tuned with idrak-tune, the
/// Idrak fine-tuning tool, on the intent data; see the README): the same kind of result as
/// <see cref="TextClassifier"/> (a label, its confidence and every label's probability). Predictions score each intent as
/// the model's answer (the probability of its tokens, then a softmax over the intents), so the label is always one of the
/// intents; <see cref="StreamAsync"/> also streams the model's own answer token by token.
/// </summary>
public sealed class TunedClassifier : IDisposable
{
    private readonly PretrainedModel _model;
    private readonly AnswerScorer _scorer;
    private readonly string _system;
    private readonly int _maxLength;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private ChatGenerator? _chat;

    private TunedClassifier(PretrainedModel model, string[] labels, string system, int maxLength, string baseModel, string folder)
    {
        _model = model;
        if (model.ChatTemplate is null || model.Tokenizer is null)
        {
            throw new InvalidDataException($"{baseModel} has no chat template or tokenizer.");
        }

        _scorer = new AnswerScorer(model, maxLength);
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
    /// Loads a tuned adapter folder (written by idrak-tune train: the adapter and idrak-tuning.json, which names the base
    /// model and the system prompt it was tuned with), the adapter merged into the base weights for speed, to classify into
    /// <paramref name="labels"/> (the intents it was tuned to answer with).
    /// </summary>
    public static TunedClassifier Load(string folder, IReadOnlyList<string> labels, Device? device = null)
    {
        var manifest = TuningManifest.Read(folder)
                       ?? throw new InvalidDataException($"{folder} is not a tuned model folder (no {TuningManifest.FileName}; tune one with idrak-tune train).");
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
        Directory.Exists(folder) && TuningManifest.Exists(folder);

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
    public IAsyncEnumerable<TunedStreamItem> StreamAsync(string text, CancellationToken cancellationToken = default) =>
        StreamAsync([text], cancellationToken);

    /// <summary>
    /// The model's own answers to <paramref name="texts"/>, generated together and streamed as they come (the library's
    /// batched chat stream: one pass through the model per token for all of them), each piece tagged with its message's
    /// index; then every message's scored prediction (the same as <see cref="Predict(IReadOnlyList{string})"/>).
    /// </summary>
    public async IAsyncEnumerable<TunedStreamItem> StreamAsync(IReadOnlyList<string> texts, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            int context = Math.Min(_model.MaxPositions, _maxLength + 32);
            _chat ??= _model.CreateChat(contextLength: context);
            var options = new GenerationOptions { Temperature = 0f, TopK = 1, TopP = 1f, RepeatPenalty = 1f, NumPredict = 16, ChunkSize = 1, NumCtx = context };
            var requests = texts.Select(t => new ChatRequest(_scorer.Fit(Prompt(t), Labels) ?? Prompt(""), null, false, options)).ToList();
            await foreach (var (index, chunk) in _chat.StreamBatchAsync(requests, cancellationToken).ConfigureAwait(false))
            {
                if (chunk.Delta.Content.Length > 0)
                {
                    yield return new TunedStreamItem(index, chunk.Delta.Content, null);
                }
            }

            var predictions = Score(texts);
            for (int i = 0; i < predictions.Count; i++)
            {
                yield return new TunedStreamItem(i, null, predictions[i]);
            }
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

    // The conversation as the training saw it (idrak-tune with the same system prompt), up to the answer.
    private IReadOnlyList<ChatMessage> Prompt(string text) =>
        _system.Length > 0 ? [new ChatMessage("system", _system), new ChatMessage("user", text)] : [new ChatMessage("user", text)];

    private TextPrediction Score(string text) => Score([text])[0];

    // Each intent scored as the model's answer by the library's AnswerScorer (log p of its tokens, softmaxed over the
    // intents); a message too long for the tuning's length is shortened there, its start kept.
    private List<TextPrediction> Score(IReadOnlyList<string> texts, Action<int>? progress = null)
    {
        var choices = _scorer.Choose([.. texts.Select(Prompt)], Labels, progress);
        return [.. choices.Select(c =>
        {
            var probabilities = Labels.Select((l, i) => (Label: l, Probability: (float)c.Probabilities[i])).OrderByDescending(p => p.Probability).ToList();
            return new TextPrediction(Labels[c.Best], probabilities[0].Probability, probabilities);
        })];
    }
}

/// <summary>One item of <see cref="TunedClassifier.StreamAsync"/>: a piece of the generated answer, or the final prediction.</summary>
/// <param name="Token">Text the model generated (null on the final item).</param>
/// <param name="Prediction">The scored prediction (only on the final item).</param>
public sealed record TunedStreamItem(int Index, string? Token, TextPrediction? Prediction);

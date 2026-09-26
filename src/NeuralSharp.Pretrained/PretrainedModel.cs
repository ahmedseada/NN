using System.Text.Json.Nodes;
using NeuralSharp.Generation;
using NeuralSharp.Layers;

namespace NeuralSharp.Pretrained;

/// <summary>Settings for <see cref="PretrainedModel.Load"/>.</summary>
public sealed record PretrainedOptions
{
    /// <summary>Where the model is created (the default device when null).</summary>
    public Device? Device { get; init; }

    /// <summary>Store the projections as int8 (quantized as they are read; about a quarter of the float32 memory).</summary>
    public bool Int8 { get; init; }

    /// <summary>Longest sequence to support (sizes the rotary tables); the model's maximum when null.</summary>
    public int? MaxPositions { get; init; }

    /// <summary>The architecture to use instead of the one named in config.json.</summary>
    public string? Architecture { get; init; }
}

/// <summary>
/// A pretrained decoder-only language model read from a folder in the Hugging Face layout (config.json, safetensors
/// weights, tokenizer.json, tokenizer_config.json): the <see cref="Network"/> built from its <see cref="Spec"/>, its
/// <see cref="Tokenizer"/> and <see cref="ChatTemplate"/>. The network is an ordinary NeuralSharp model: generate with
/// <see cref="CreateGenerator"/>, chat with <see cref="CreateChat"/>, fine-tune with LoRA, quantize, save as a package.
/// </summary>
public sealed class PretrainedModel : IDisposable
{
    private PretrainedModel(string folder, JsonObject config, DecoderSpec spec, Sequential network, ITokenizer? tokenizer, JinjaChatTemplate? template,
        IReadOnlyList<string> notes, int maxPositions)
    {
        Folder = folder;
        Config = config;
        Spec = spec;
        Network = network;
        Tokenizer = tokenizer;
        ChatTemplate = template;
        Notes = notes;
        MaxPositions = maxPositions;
    }

    /// <summary>The folder the model was read from.</summary>
    public string Folder { get; }

    /// <summary>The model's config.json.</summary>
    public JsonObject Config { get; }

    /// <summary>The architecture as NeuralSharp describes it.</summary>
    public DecoderSpec Spec { get; }

    /// <summary>The model.</summary>
    public Sequential Network { get; }

    /// <summary>The tokenizer (from tokenizer.json), or null when the folder has none.</summary>
    public ITokenizer? Tokenizer { get; }

    /// <summary>The chat template (from tokenizer_config.json), or null when the model has none.</summary>
    public JinjaChatTemplate? ChatTemplate { get; }

    /// <summary>Anything approximated while reading the model (see <see cref="PretrainedArchitectures.CommonSpec"/>).</summary>
    public IReadOnlyList<string> Notes { get; }

    /// <summary>The longest sequence the loaded model supports.</summary>
    public int MaxPositions { get; }

    /// <summary>
    /// Reads the model in <paramref name="folder"/>. Every weight is read from disk one tensor at a time and (with
    /// <see cref="PretrainedOptions.Int8"/>) quantized on the host, so the model is never held twice.
    /// </summary>
    public static PretrainedModel Load(string folder, PretrainedOptions? options = null)
    {
        options ??= new PretrainedOptions();
        var config = JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "config.json")))!.AsObject();
        string name = options.Architecture ?? (string?)config["architectures"]?[0]
            ?? throw new InvalidDataException("config.json names no architecture; pass PretrainedOptions.Architecture.");
        var architecture = PretrainedArchitectures.Get(name);
        var notes = new List<string>();
        var spec = architecture.Spec(config, notes);
        int maxPositions = Math.Min(options.MaxPositions ?? spec.MaxPositions, spec.MaxPositions);
        using var reader = SafeTensorsReader.Open(folder);
        var weights = new CheckpointWeights(reader, architecture);
        var network = spec.Build(weights, new DecoderBuildOptions { Device = options.Device, Int8 = options.Int8, MaxPositions = maxPositions });
        var unused = reader.Tensors.Keys.Where(k => !weights.Used.Contains(k) && !k.EndsWith("rotary_emb.inv_freq", StringComparison.Ordinal)).ToList();
        if (unused.Count > 0)
        {
            notes.Add($"{unused.Count} checkpoint tensors were not used (for example {string.Join(", ", unused.Take(3))}).");
        }

        var tokenizer = File.Exists(Path.Combine(folder, "tokenizer.json")) ? BpeTokenizer.Load(folder) : null;
        tokenizer?.PadVocabulary(spec.Vocabulary);
        var template = JinjaChatTemplate.Load(folder, tokenizer);
        return new PretrainedModel(folder, config, spec, network, tokenizer, template, notes, maxPositions);
    }

    /// <summary>A text generator for the model (int8 KV cache with <paramref name="cacheFormat"/>).</summary>
    public TextGenerator CreateGenerator(KeyValueFormat cacheFormat = KeyValueFormat.Float32, int? contextLength = null)
    {
        Network.Eval();
        return new TextGenerator(Network, Tokenizer ?? throw new InvalidOperationException("The model has no tokenizer."),
            Math.Min(contextLength ?? MaxPositions, MaxPositions)) { CacheFormat = cacheFormat };
    }

    /// <summary>A chat model using the model's own chat template (and its tool-call format).</summary>
    public ChatGenerator CreateChat(KeyValueFormat cacheFormat = KeyValueFormat.Float32, int? contextLength = null) =>
        new(CreateGenerator(cacheFormat, contextLength), ChatTemplate ?? throw new InvalidOperationException("The model has no chat template."));

    /// <inheritdoc />
    public void Dispose() => Network.Dispose();
}

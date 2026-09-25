using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using System.Threading.Channels;
using NeuralSharp.Generation;
using NeuralSharp.Samples.Gpt;

namespace NeuralSharp.Samples.GptApi;

// ------------------------------------------------------------------ wire format (Ollama-compatible JSON)

/// <summary>An /api/chat request in the Ollama format. Unknown fields are ignored; any model name selects the served model.</summary>
public sealed record OllamaChatRequest(
    [property: JsonPropertyName("model")] string? Model,
    [property: JsonPropertyName("messages")] List<OllamaMessage> Messages,
    [property: JsonPropertyName("stream")] bool? Stream = null,
    [property: JsonPropertyName("think")] JsonElement? Think = null,
    [property: JsonPropertyName("keep_alive")] JsonElement? KeepAlive = null,
    [property: JsonPropertyName("options")] Dictionary<string, JsonElement>? Options = null,
    [property: JsonPropertyName("tools")] List<OllamaTool>? Tools = null);

/// <summary>A chat message: role (system, user, assistant, tool), content, and optionally reasoning, tool calls or the tool name.</summary>
public sealed record OllamaMessage(
    [property: JsonPropertyName("role")] string Role,
    [property: JsonPropertyName("content")] string? Content = null,
    [property: JsonPropertyName("thinking"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] string? Thinking = null,
    [property: JsonPropertyName("tool_calls"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] List<OllamaToolCall>? ToolCalls = null,
    [property: JsonPropertyName("tool_name"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] string? ToolName = null);

/// <summary>A tool definition: {"type": "function", "function": {name, description, parameters}}.</summary>
public sealed record OllamaTool(
    [property: JsonPropertyName("type")] string Type,
    [property: JsonPropertyName("function")] OllamaFunction Function);

/// <summary>A function's name, description and JSON-schema parameters.</summary>
public sealed record OllamaFunction(
    [property: JsonPropertyName("name")] string Name,
    [property: JsonPropertyName("description")] string? Description = null,
    [property: JsonPropertyName("parameters")] JsonNode? Parameters = null);

/// <summary>A tool call returned by the model.</summary>
public sealed record OllamaToolCall([property: JsonPropertyName("function")] OllamaCalledFunction Function);

/// <summary>The called function and its arguments.</summary>
public sealed record OllamaCalledFunction(
    [property: JsonPropertyName("name")] string Name,
    [property: JsonPropertyName("arguments")] JsonObject Arguments,
    [property: JsonPropertyName("index"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] int? Index = null);

/// <summary>One streamed line (or the whole non-streamed reply) of /api/chat. Durations are in nanoseconds.</summary>
public sealed record OllamaChatResponse(
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("created_at")] DateTimeOffset CreatedAt,
    [property: JsonPropertyName("message")] OllamaMessage Message,
    [property: JsonPropertyName("done")] bool Done,
    [property: JsonPropertyName("done_reason"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] string? DoneReason = null,
    [property: JsonPropertyName("total_duration"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] long? TotalDuration = null,
    [property: JsonPropertyName("load_duration"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] long? LoadDuration = null,
    [property: JsonPropertyName("prompt_eval_count"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] int? PromptEvalCount = null,
    [property: JsonPropertyName("prompt_eval_duration"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] long? PromptEvalDuration = null,
    [property: JsonPropertyName("eval_count"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] int? EvalCount = null,
    [property: JsonPropertyName("eval_duration"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] long? EvalDuration = null);

/// <summary>A model entry for /api/tags.</summary>
public sealed record OllamaModelTag(
    [property: JsonPropertyName("name")] string Name,
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("modified_at")] DateTimeOffset ModifiedAt,
    [property: JsonPropertyName("size")] long Size,
    [property: JsonPropertyName("details")] OllamaModelDetails Details);

/// <summary>Model details.</summary>
public sealed record OllamaModelDetails(
    [property: JsonPropertyName("format")] string Format,
    [property: JsonPropertyName("family")] string Family,
    [property: JsonPropertyName("parameter_size")] string ParameterSize,
    [property: JsonPropertyName("context_length")] int ContextLength);

/// <summary>A loaded model for /api/ps.</summary>
public sealed record OllamaRunningModel(
    [property: JsonPropertyName("name")] string Name,
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("size")] long Size,
    [property: JsonPropertyName("expires_at")] DateTimeOffset? ExpiresAt,
    [property: JsonPropertyName("size_vram")] long SizeVram,
    [property: JsonPropertyName("context_length")] int ContextLength);

/// <summary>A served model: the GPT, its chat engine, and a gate that serializes generation on it.</summary>
public sealed class HostedChatModel(CharGpt gpt) : IDisposable
{
    public CharGpt Gpt { get; } = gpt;
    public ChatGenerator Chat { get; } = new(new TextGenerator(gpt.Model, new CharTokenizer(gpt.Config.Vocabulary), gpt.Config.Context));
    public SemaphoreSlim Gate { get; } = new(1, 1);

    public void Dispose()
    {
        Gate.Dispose();
        Gpt.Dispose();
    }
}

/// <summary>
/// Ollama-compatible chat on top of NeuralSharp's generation layer: maps the request's options to
/// <see cref="GenerationOptions"/>, keeps the model loaded according to keep_alive, serializes generation per model,
/// and produces Ollama-shaped responses (streamed as NDJSON or as one object).
/// </summary>
public sealed class ChatService : IDisposable
{
    private readonly string _modelPath;
    private readonly string _servedName;
    private readonly ModelHost<HostedChatModel> _host;
    private readonly ILogger<ChatService> _logger;

    public ChatService(IConfiguration configuration, ILogger<ChatService> logger)
    {
        _logger = logger;
        // A chat-trained model (Transformer sample, --chat true) if configured and present; otherwise the text model.
        string textModel = Path.GetFullPath(configuration["Gpt:ModelPath"] ?? Path.Combine(AppContext.BaseDirectory, "models", "transformer.weights"));
        string? chatModel = configuration["Gpt:ChatModelPath"] is { Length: > 0 } c ? Path.GetFullPath(c) : null;
        _modelPath = chatModel is not null && File.Exists(chatModel) ? chatModel : textModel;
        _servedName = configuration["Gpt:ModelName"] ?? "neuralsharp-char-gpt:latest";
        var device = (configuration["Gpt:Device"] ?? "auto") switch
        {
            "cpu" => Device.Cpu,
            "cuda" when Device.IsCudaAvailable => Device.Cuda(),
            _ => Device.Default,
        };
        _host = new ModelHost<HostedChatModel>(_ =>
        {
            if (!File.Exists(_modelPath) || !File.Exists(GptConfig.ConfigPath(_modelPath)))
            {
                throw new FileNotFoundException($"No model at {_modelPath} yet (train one with the Transformer sample, or wait for the background training).");
            }

            _logger.LogInformation("Loading {Path} on {Device}", _modelPath, device);
            return new HostedChatModel(CharGpt.Load(_modelPath, device));
        }, TimeSpan.FromMinutes(5));
    }

    public string ServedName => _servedName;

    public IReadOnlyList<OllamaModelTag> Tags()
    {
        if (!File.Exists(_modelPath) || !File.Exists(GptConfig.ConfigPath(_modelPath)))
        {
            return [];
        }

        var config = GptConfig.Load(_modelPath);
        long parameters = config.Vocabulary.Length * config.Dim + config.Layers * (12L * config.Dim * config.Dim + 13L * config.Dim)
                          + 2 * config.Dim + config.Dim * config.Vocabulary.Length + config.Vocabulary.Length;
        return [new OllamaModelTag(_servedName, _servedName, File.GetLastWriteTimeUtc(_modelPath), new FileInfo(_modelPath).Length,
            new OllamaModelDetails("neuralsharp", "char-gpt", $"{parameters / 1000.0:F0}K", config.Context))];
    }

    /// <summary>Whether a trained model file exists to serve.</summary>
    public bool IsAvailable => File.Exists(_modelPath) && File.Exists(GptConfig.ConfigPath(_modelPath));

    public IReadOnlyList<OllamaRunningModel> Running()
    {
        var loaded = _host.Loaded;
        if (loaded.Count == 0)
        {
            return [];
        }

        var config = GptConfig.Load(_modelPath);
        long size = new FileInfo(_modelPath).Length;
        return [.. loaded.Select(m => new OllamaRunningModel(_servedName, _servedName, size, m.ExpiresAt, 0, config.Context))];
    }

    /// <summary>Validates the request and turns it into the library's chat request; throws <see cref="ArgumentException"/> on bad input.</summary>
    public static (ChatRequest Request, TimeSpan? KeepAlive, bool UseDefaultKeepAlive) Translate(OllamaChatRequest request)
    {
        if (request.Messages is null)
        {
            throw new ArgumentException("messages is required.");
        }

        var options = new GenerationOptions();
        foreach (var (key, value) in request.Options ?? [])
        {
            options = key switch
            {
                "temperature" => options with { Temperature = value.GetSingle() },
                "top_k" => options with { TopK = value.GetInt32() },
                "top_p" => options with { TopP = value.GetSingle() },
                "min_p" => options with { MinP = value.GetSingle() },
                "repeat_penalty" => options with { RepeatPenalty = value.GetSingle() },
                "repeat_last_n" => options with { RepeatLastN = value.GetInt32() },
                "presence_penalty" => options with { PresencePenalty = value.GetSingle() },
                "frequency_penalty" => options with { FrequencyPenalty = value.GetSingle() },
                "seed" => options with { Seed = value.GetInt32() },
                "num_ctx" => options with { NumCtx = value.GetInt32() },
                "num_predict" => options with { NumPredict = value.GetInt32() },
                "stop" => options with { Stop = [.. value.EnumerateArray().Select(s => s.GetString()!)] },
                _ => options,                                           // other options are accepted and ignored
            };
        }

        bool? think = request.Think?.ValueKind switch
        {
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            JsonValueKind.String => request.Think.Value.GetString() is not ("" or "none" or "false"),   // "low" | "medium" | "high"
            _ => null,
        };

        var messages = request.Messages.Select(m => new ChatMessage(m.Role, m.Content ?? "", m.Thinking,
            m.ToolCalls?.Select(c => new ToolCall(c.Function.Name, c.Function.Arguments)).ToList(), m.ToolName)).ToList();
        var tools = (request.Tools ?? []).Where(t => t.Type == "function")
            .Select(t => new ToolDefinition(t.Function.Name, t.Function.Description, t.Function.Parameters)).ToList();
        bool useDefault = request.KeepAlive is null || request.KeepAlive.Value.ValueKind is JsonValueKind.Null or JsonValueKind.Undefined;
        TimeSpan? keepAlive;
        try
        {
            keepAlive = useDefault ? null : KeepAlive.Parse(request.KeepAlive, null);
        }
        catch (FormatException ex)
        {
            throw new ArgumentException($"keep_alive: {ex.Message}");
        }

        return (new ChatRequest(messages, tools, think, options), keepAlive, useDefault);
    }

    /// <summary>Runs the chat and produces Ollama response objects: deltas while streaming, then the final one with statistics.</summary>
    public async IAsyncEnumerable<OllamaChatResponse> ChatAsync(OllamaChatRequest request, bool stream,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var (chatRequest, keepAlive, useDefault) = Translate(request);
        string name = request.Model ?? _servedName;
        var channel = Channel.CreateUnbounded<OllamaChatResponse>(new UnboundedChannelOptions { SingleReader = true });
        var total = Stopwatch.StartNew();
        ModelHost<HostedChatModel>.Lease lease;
        try
        {
            lease = useDefault ? _host.Acquire(_servedName) : _host.Acquire(_servedName, keepAlive);
        }
        catch (FileNotFoundException ex)
        {
            throw new InvalidOperationException(ex.Message);
        }

        _ = Task.Run(async () =>
        {
            using var _ = lease;
            var model = lease.Model;
            try
            {
                await model.Gate.WaitAsync(cancellationToken);
            }
            catch (OperationCanceledException)
            {
                channel.Writer.TryComplete();
                return;
            }

            try
            {
                int toolIndex = 0;
                foreach (var chunk in model.Chat.Stream(chatRequest, cancellationToken))
                {
                    if (!chunk.Done)
                    {
                        if (stream)
                        {
                            channel.Writer.TryWrite(Line(chunk.Delta.Content, chunk.Delta.Thinking, Calls(chunk.Delta.ToolCalls, ref toolIndex)));
                        }

                        continue;
                    }

                    var s = chunk.Stats!;
                    var message = stream
                        ? new OllamaMessage("assistant", chunk.Delta.Content, NullIfEmpty(chunk.Delta.Thinking), Calls(chunk.Delta.ToolCalls, ref toolIndex))
                        : new OllamaMessage("assistant", chunk.Message!.Content, chunk.Message.Thinking, Calls(chunk.Message.ToolCalls ?? [], ref toolIndex));
                    channel.Writer.TryWrite(new OllamaChatResponse(name, DateTimeOffset.UtcNow, message, true, chunk.DoneReason,
                        Nanoseconds(total.Elapsed), Nanoseconds(lease.LoadDuration), s.PromptTokens, Nanoseconds(s.PromptDuration),
                        s.GeneratedTokens, Nanoseconds(s.GenerationDuration)));
                }
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                _logger.LogError(ex, "Chat failed");
                channel.Writer.TryComplete(ex);
                return;
            }
            finally
            {
                model.Gate.Release();
            }

            channel.Writer.TryComplete();
        }, CancellationToken.None);

        await foreach (var item in channel.Reader.ReadAllAsync(cancellationToken))
        {
            yield return item;
        }

        OllamaChatResponse Line(string content, string thinking, List<OllamaToolCall>? calls) =>
            new(name, DateTimeOffset.UtcNow, new OllamaMessage("assistant", content, NullIfEmpty(thinking), calls), false);
    }

    private static List<OllamaToolCall>? Calls(IReadOnlyList<ToolCall> calls, ref int index)
    {
        if (calls.Count == 0)
        {
            return null;
        }

        var list = new List<OllamaToolCall>();
        foreach (var c in calls)
        {
            list.Add(new OllamaToolCall(new OllamaCalledFunction(c.Name, c.Arguments, index++)));
        }

        return list;
    }

    private static string? NullIfEmpty(string text) => text.Length == 0 ? null : text;

    private static long Nanoseconds(TimeSpan t) => t.Ticks * 100;

    public void Dispose() => _host.Dispose();
}

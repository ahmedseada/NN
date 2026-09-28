using System.Diagnostics;
using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace NeuralSharp.Generation;

/// <summary>
/// A chat model behind an OpenAI-compatible <c>/v1/chat/completions</c> endpoint: a local server such as Ollama,
/// llama.cpp's llama-server, LM Studio or vLLM (for example as the teacher that produces agent transcripts). Streams
/// content, reasoning ("reasoning_content" or "reasoning") and tool calls.
/// </summary>
public sealed class OpenAIChatModel : IChatModel, IDisposable
{
    private readonly HttpClient _http;
    private readonly bool _ownsClient;
    private readonly Uri _endpoint;

    /// <summary>
    /// A model at <paramref name="baseUrl"/> (for example http://localhost:11434/v1 for Ollama, http://localhost:8080/v1 for
    /// llama-server; "/chat/completions" is added unless the URL already ends with it).
    /// </summary>
    public OpenAIChatModel(string baseUrl, string model, string? apiKey = null, HttpClient? http = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(baseUrl);
        Model = model;
        string url = baseUrl.TrimEnd('/');
        _endpoint = new Uri(url.EndsWith("/chat/completions", StringComparison.Ordinal) ? url : url + "/chat/completions");
        _ownsClient = http is null;
        _http = http ?? new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
        if (apiKey is not null && _ownsClient)
        {
            _http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", apiKey);
        }
    }

    /// <summary>The model name sent with each request.</summary>
    public string Model { get; }

    /// <summary>
    /// Also send top_k, min_p and repeat_penalty (understood by llama-server, vLLM and LM Studio; strict OpenAI servers
    /// may reject them).
    /// </summary>
    public bool ExtendedSampling { get; init; } = true;

    /// <summary>Send the options' sampling settings (false: the server's defaults, often the model's recommended ones).</summary>
    public bool SendSampling { get; init; } = true;

    /// <summary>The request body for <paramref name="request"/>.</summary>
    public JsonObject CreateBody(ChatRequest request, bool stream = true)
    {
        var body = new JsonObject
        {
            ["model"] = Model,
            ["messages"] = ChatJson.Messages(request.Messages, argumentsAsText: true, includeThinking: false),
            ["stream"] = stream,
        };
        if (stream)
        {
            body["stream_options"] = new JsonObject { ["include_usage"] = true };
        }

        if (request.Tools is { Count: > 0 } tools)
        {
            body["tools"] = ChatJson.Tools(tools);
        }

        if (request.Think is { } think)
        {
            body["chat_template_kwargs"] = new JsonObject { ["enable_thinking"] = think };   // llama-server, vLLM
            body["think"] = think;                                                            // Ollama
        }

        if (request.Options is { } o)
        {
            if (SendSampling)
            {
                body["temperature"] = o.Temperature;
                body["top_p"] = o.TopP;
                if (o.PresencePenalty != 0)
                {
                    body["presence_penalty"] = o.PresencePenalty;
                }

                if (o.FrequencyPenalty != 0)
                {
                    body["frequency_penalty"] = o.FrequencyPenalty;
                }

                if (ExtendedSampling)
                {
                    body["top_k"] = o.TopK;
                    body["min_p"] = o.MinP;
                    body["repeat_penalty"] = o.RepeatPenalty;
                }
            }

            if (o.Seed is { } seed)
            {
                body["seed"] = seed;
            }

            if (o.NumPredict > 0)
            {
                body["max_tokens"] = o.NumPredict;
            }

            if (o.Stop.Count > 0)
            {
                body["stop"] = new JsonArray([.. o.Stop.Select(s => (JsonNode?)JsonValue.Create(s))]);
            }
        }

        return body;
    }

    /// <inheritdoc />
    public async IAsyncEnumerable<ChatChunk> StreamAsync(ChatRequest request, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var watch = Stopwatch.StartNew();
        using var message = new HttpRequestMessage(HttpMethod.Post, _endpoint)
        {
            Content = new StringContent(CreateBody(request).ToJsonString(), Encoding.UTF8, "application/json"),
        };
        using var response = await _http.SendAsync(message, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
        if (!response.IsSuccessStatusCode)
        {
            string error = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
            throw new HttpRequestException($"{_endpoint} answered {(int)response.StatusCode} {response.ReasonPhrase}: {error}", null, response.StatusCode);
        }

        var content = new StringBuilder();
        var thinking = new StringBuilder();
        var calls = new SortedDictionary<int, (StringBuilder Name, StringBuilder Arguments)>();
        string? finish = null;
        int promptTokens = 0, generatedTokens = 0;
        TimeSpan? firstToken = null;
        using var stream = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
        using var reader = new StreamReader(stream, Encoding.UTF8);
        bool sse = response.Content.Headers.ContentType?.MediaType == "text/event-stream";
        if (!sse)
        {
            // A server that ignored "stream": one JSON response.
            var json = JsonNode.Parse(await reader.ReadToEndAsync(cancellationToken).ConfigureAwait(false)) as JsonObject;
            var choice = json?["choices"] is JsonArray { Count: > 0 } choices ? choices[0] : null;
            var whole = choice?["message"] as JsonObject;
            Apply(whole, content, thinking, calls);
            finish = (string?)choice?["finish_reason"];
            (promptTokens, generatedTokens) = Usage(json?["usage"], promptTokens, generatedTokens);
            if (content.Length > 0 || thinking.Length > 0)
            {
                yield return new ChatChunk(new ChatDelta(content.ToString(), thinking.ToString(), []));
            }
        }
        else
        {
            while (await reader.ReadLineAsync(cancellationToken).ConfigureAwait(false) is { } line)
            {
                if (!line.StartsWith("data:", StringComparison.Ordinal))
                {
                    continue;
                }

                string data = line[5..].Trim();
                if (data == "[DONE]")
                {
                    break;
                }

                JsonObject? json;
                try
                {
                    json = JsonNode.Parse(data) as JsonObject;
                }
                catch (JsonException)
                {
                    continue;
                }

                if (json?["error"] is { } error)
                {
                    throw new HttpRequestException($"{_endpoint}: {(string?)error["message"] ?? error.ToJsonString()}");
                }

                (promptTokens, generatedTokens) = Usage(json?["usage"], promptTokens, generatedTokens);
                if (json?["choices"] is not JsonArray { Count: > 0 } choices)
                {
                    continue;
                }

                var choice = choices[0]!;
                finish = (string?)choice["finish_reason"] ?? finish;
                int contentBefore = content.Length, thinkingBefore = thinking.Length;
                Apply(choice["delta"] as JsonObject, content, thinking, calls);
                if (content.Length > contentBefore || thinking.Length > thinkingBefore)
                {
                    firstToken ??= watch.Elapsed;
                    yield return new ChatChunk(new ChatDelta(content.ToString(contentBefore, content.Length - contentBefore),
                        thinking.ToString(thinkingBefore, thinking.Length - thinkingBefore), []));
                }
            }
        }

        var toolCalls = calls.Values.Where(c => c.Name.Length > 0).Select(c => new ToolCall(c.Name.ToString(), Arguments(c.Arguments.ToString()))).ToList();
        if (toolCalls.Count > 0)
        {
            yield return new ChatChunk(new ChatDelta("", "", toolCalls));
        }

        var total = watch.Elapsed;
        var prompt = firstToken ?? total;
        var stats = new GenerationStats(promptTokens, prompt, generatedTokens, total - prompt, total, 0);
        var reply = new ChatMessage("assistant", content.ToString().Trim(), thinking.Length > 0 ? thinking.ToString().Trim() : null, toolCalls.Count > 0 ? toolCalls : null);
        yield return new ChatChunk(new ChatDelta("", "", []), true, finish == "length" ? "length" : "stop", reply, stats);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_ownsClient)
        {
            _http.Dispose();
        }
    }

    private static void Apply(JsonObject? delta, StringBuilder content, StringBuilder thinking, SortedDictionary<int, (StringBuilder Name, StringBuilder Arguments)> calls)
    {
        if (delta is null)
        {
            return;
        }

        content.Append((string?)delta["content"]);
        thinking.Append((string?)delta["reasoning_content"] ?? (string?)delta["reasoning"]);
        if (delta["tool_calls"] is not JsonArray list)
        {
            return;
        }

        for (int i = 0; i < list.Count; i++)
        {
            var call = list[i];
            int index = call?["index"] is JsonValue v && v.TryGetValue<int>(out int n) ? n : calls.Count > 0 && call?["function"]?["name"] is null ? calls.Keys.Last() : calls.Count;
            if (!calls.TryGetValue(index, out var entry))
            {
                calls[index] = entry = (new StringBuilder(), new StringBuilder());
            }

            entry.Name.Append((string?)call?["function"]?["name"]);
            switch (call?["function"]?["arguments"])
            {
                case JsonValue text when text.TryGetValue<string>(out var s):
                    entry.Arguments.Append(s);
                    break;
                case JsonObject o:
                    entry.Arguments.Append(o.ToJsonString());
                    break;
            }
        }
    }

    // Arguments that are not a JSON object are passed on as {"arguments": text}; argument validation then tells the model.
    private static JsonObject Arguments(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return [];
        }

        try
        {
            return JsonNode.Parse(text) as JsonObject ?? new JsonObject { ["arguments"] = text };
        }
        catch (JsonException)
        {
            return new JsonObject { ["arguments"] = text };
        }
    }

    private static (int Prompt, int Generated) Usage(JsonNode? usage, int prompt, int generated) =>
        usage is JsonObject u
            ? (u["prompt_tokens"] is JsonValue p && p.TryGetValue<int>(out int pt) ? pt : prompt, u["completion_tokens"] is JsonValue c && c.TryGetValue<int>(out int ct) ? ct : generated)
            : (prompt, generated);
}

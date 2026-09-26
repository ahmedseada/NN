using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using NeuralSharp.Generation;

namespace NeuralSharp.Pretrained;

/// <summary>
/// A model's own chat template: the Jinja source from its tokenizer_config.json (or chat_template.jinja), rendered the
/// way Hugging Face's <c>apply_chat_template</c> renders it. Messages become dicts with role, content,
/// reasoning_content and tool_calls ([{type: "function", function: {name, arguments}}]); tools become
/// [{type: "function", function: {name, description, parameters}}]; <c>enable_thinking</c> is set when reasoning is
/// requested or suppressed. Because the model's template decides the layout, prompts (and fine-tuning data) match what
/// the model was trained on, whatever the family.
/// </summary>
public sealed class JinjaChatTemplate : ChatTemplate
{
    private readonly JinjaTemplate _template;
    private readonly JinjaTemplate? _toolTemplate;

    /// <summary>A template from its Jinja source.</summary>
    /// <param name="source">The template.</param>
    /// <param name="stopSequences">Text that ends an assistant turn (usually the end-of-turn and end-of-sequence tokens).</param>
    /// <param name="bosToken">The value of <c>bos_token</c>.</param>
    /// <param name="eosToken">The value of <c>eos_token</c>.</param>
    /// <param name="toolSource">A separate template used when tools are given (some models name one "tool_use").</param>
    public JinjaChatTemplate(string source, IReadOnlyList<string> stopSequences, string bosToken = "", string eosToken = "", string? toolSource = null)
    {
        Source = source;
        _template = JinjaTemplate.Parse(source);
        _toolTemplate = toolSource is null ? null : JinjaTemplate.Parse(toolSource);
        StopSequences = stopSequences;
        BosToken = bosToken;
        EosToken = eosToken;
    }

    /// <summary>The Jinja source.</summary>
    public string Source { get; }

    /// <summary>The value of <c>bos_token</c> in the template.</summary>
    public string BosToken { get; }

    /// <summary>The value of <c>eos_token</c> in the template.</summary>
    public string EosToken { get; }

    /// <inheritdoc />
    public override IReadOnlyList<string> StopSequences { get; }

    /// <summary>The reasoning tags the model writes (&lt;think&gt;…&lt;/think&gt; unless set).</summary>
    public (string Open, string Close) ReasoningTags { get; init; } = ("<think>", "</think>");

    /// <summary>The tags around a tool call's JSON in the model's output (&lt;tool_call&gt;…&lt;/tool_call&gt; unless set).</summary>
    public (string Open, string Close) CallTags { get; init; } = ("<tool_call>", "</tool_call>");

    /// <inheritdoc />
    public override (string Open, string Close) ThinkTags => ReasoningTags;

    /// <inheritdoc />
    public override (string Open, string Close) ToolCallTags => CallTags;

    /// <summary>Extra template variables (for example a date or a model-specific switch).</summary>
    public IReadOnlyDictionary<string, object?> Variables { get; init; } = new Dictionary<string, object?>();

    /// <inheritdoc />
    public override string Render(IReadOnlyList<ChatMessage> messages, IReadOnlyList<ToolDefinition> tools, bool? think) =>
        Render(messages, tools, think, addGenerationPrompt: true);

    /// <summary>
    /// Renders the conversation. With <paramref name="addGenerationPrompt"/> false the text ends after the last message,
    /// as training data is rendered.
    /// </summary>
    public string Render(IReadOnlyList<ChatMessage> messages, IReadOnlyList<ToolDefinition> tools, bool? think, bool addGenerationPrompt)
    {
        var variables = new Dictionary<string, object?>(Variables)
        {
            ["messages"] = ToValues(messages),
            ["tools"] = tools.Count > 0 ? tools.Select(ToValue).ToList<object?>() : null,
            ["add_generation_prompt"] = addGenerationPrompt,
            ["bos_token"] = BosToken,
            ["eos_token"] = EosToken,
        };
        if (think is bool enabled)
        {
            variables["enable_thinking"] = enabled;
        }

        return (tools.Count > 0 && _toolTemplate is not null ? _toolTemplate : _template).Render(variables);
    }

    /// <summary>
    /// Loads the chat template of the model in <paramref name="folder"/> (chat_template.jinja, chat_template.json or the
    /// "chat_template" of tokenizer_config.json), or null when it has none. Stop sequences are the end-of-sequence token
    /// and every token listed as eos_token_id in generation_config.json or config.json.
    /// </summary>
    public static JinjaChatTemplate? Load(string folder, BpeTokenizer? tokenizer = null)
    {
        var config = ReadJson(Path.Combine(folder, "tokenizer_config.json"));
        string? source = null, toolSource = null;
        string jinjaFile = Path.Combine(folder, "chat_template.jinja");
        if (File.Exists(jinjaFile))
        {
            source = File.ReadAllText(jinjaFile);
        }
        else if (ReadJson(Path.Combine(folder, "chat_template.json"))?["chat_template"] is { } separate)
        {
            (source, toolSource) = Choose(separate);
        }
        else if (config?["chat_template"] is { } inline)
        {
            (source, toolSource) = Choose(inline);
        }

        if (source is null)
        {
            return null;
        }

        string bos = TokenText(config?["bos_token"]), eos = TokenText(config?["eos_token"]);
        var stops = new List<string>();
        if (eos.Length > 0)
        {
            stops.Add(eos);
        }

        foreach (var file in new[] { "generation_config.json", "config.json" })
        {
            if (ReadJson(Path.Combine(folder, file))?["eos_token_id"] is { } ids && tokenizer is not null)
            {
                foreach (var id in ids is JsonArray array ? array.Select(i => (int)i!) : [(int)ids])
                {
                    if (id >= 0 && id < tokenizer.VocabularySize && tokenizer.TokenOf(id) is { Length: > 0 } token && !stops.Contains(token))
                    {
                        stops.Add(token);
                    }
                }
            }
        }

        return new JinjaChatTemplate(source, stops, bos, eos, toolSource);
    }

    private static (string? Default, string? Tools) Choose(JsonNode node)
    {
        if (node is JsonArray named)
        {
            // [{"name": "default", "template": …}, {"name": "tool_use", "template": …}]
            string? Named(string name) => (string?)named.FirstOrDefault(n => (string?)n?["name"] == name)?["template"];
            return (Named("default") ?? (string?)named[0]?["template"], Named("tool_use"));
        }

        return ((string?)node, null);
    }

    private static string TokenText(JsonNode? node) => node switch
    {
        JsonObject o => (string?)o["content"] ?? "",
        JsonValue v when v.TryGetValue<string>(out var s) => s,
        _ => "",
    };

    private static JsonObject? ReadJson(string path) =>
        File.Exists(path) ? JsonNode.Parse(File.ReadAllText(path)) as JsonObject : null;

    // Tool calls get ids ("call00000", nine characters as some templates require), and each tool result the id of the
    // earliest call not yet answered.
    private static List<object?> ToValues(IReadOnlyList<ChatMessage> messages)
    {
        var values = new List<object?>();
        var pending = new Queue<string>();
        int next = 0;
        foreach (var message in messages)
        {
            var value = ToValue(message, () =>
            {
                string id = $"call{next++:D5}";
                pending.Enqueue(id);
                return id;
            });
            if (message.Role == "tool" && pending.TryDequeue(out var answered))
            {
                value["tool_call_id"] = answered;
            }

            values.Add(value);
        }

        return values;
    }

    private static Dictionary<string, object?> ToValue(ChatMessage message, Func<string> newId)
    {
        var value = new Dictionary<string, object?> { ["role"] = message.Role, ["content"] = message.Content };
        if (message.Thinking is not null)
        {
            value["reasoning_content"] = message.Thinking;
        }

        if (message.ToolCalls is { Count: > 0 } calls)
        {
            value["tool_calls"] = calls.Select(c => (object?)new Dictionary<string, object?>
            {
                ["type"] = "function",
                ["id"] = newId(),
                ["function"] = new Dictionary<string, object?> { ["name"] = c.Name, ["arguments"] = FromJson(c.Arguments) },
            }).ToList();
        }

        if (message.ToolName is not null)
        {
            value["name"] = message.ToolName;
        }

        return value;
    }

    private static Dictionary<string, object?> ToValue(ToolDefinition tool)
    {
        var function = new Dictionary<string, object?> { ["name"] = tool.Name };
        if (tool.Description is not null)
        {
            function["description"] = tool.Description;
        }

        if (tool.Parameters is not null)
        {
            function["parameters"] = FromJson(tool.Parameters);
        }

        return new Dictionary<string, object?> { ["type"] = "function", ["function"] = function };
    }

    /// <summary>A JSON value as a template value (dicts keep their key order).</summary>
    internal static object? FromJson(JsonNode? node) => node switch
    {
        null => null,
        JsonObject o => o.ToDictionary(p => p.Key, p => FromJson(p.Value)),
        JsonArray a => a.Select(FromJson).ToList(),
        JsonValue v => v.GetValueKind() switch
        {
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            JsonValueKind.String => v.GetValue<string>(),
            JsonValueKind.Number => long.TryParse(v.ToJsonString(), NumberStyles.Integer, CultureInfo.InvariantCulture, out long l)
                ? (object)l : double.Parse(v.ToJsonString(), NumberStyles.Float, CultureInfo.InvariantCulture),
            _ => null,
        },
        _ => node.ToJsonString(),
    };
}

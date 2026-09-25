using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace NeuralSharp.Generation;

/// <summary>One message of a conversation.</summary>
/// <param name="Role">"system", "user", "assistant" or "tool".</param>
/// <param name="Content">The text.</param>
/// <param name="Thinking">An assistant's reasoning, kept apart from the answer.</param>
/// <param name="ToolCalls">Functions an assistant asked to call.</param>
/// <param name="ToolName">For role "tool": which function produced this result.</param>
public sealed record ChatMessage(string Role, string Content = "", string? Thinking = null, IReadOnlyList<ToolCall>? ToolCalls = null, string? ToolName = null);

/// <summary>A function the model may call.</summary>
/// <param name="Name">Function name.</param>
/// <param name="Description">What it does.</param>
/// <param name="Parameters">JSON schema of its arguments.</param>
public sealed record ToolDefinition(string Name, string? Description = null, JsonNode? Parameters = null);

/// <summary>A call the model asked for: a function name and JSON arguments.</summary>
public sealed record ToolCall(string Name, JsonObject Arguments);

/// <summary>Formats a conversation (and the available tools) as the prompt text a chat model was trained on.</summary>
public abstract class ChatTemplate
{
    /// <summary>Opening and closing tags of the reasoning block.</summary>
    public virtual (string Open, string Close) ThinkTags => ("<think>", "</think>");

    /// <summary>Opening and closing tags around a tool call's JSON ({"name": …, "arguments": {…}}).</summary>
    public virtual (string Open, string Close) ToolCallTags => ("<tool_call>", "</tool_call>");

    /// <summary>Text that ends an assistant turn; used as a stop sequence.</summary>
    public abstract IReadOnlyList<string> StopSequences { get; }

    /// <summary>
    /// The prompt for the next assistant turn. <paramref name="think"/>: true asks for reasoning, false suppresses it
    /// (the template closes an empty reasoning block), null leaves it to the model.
    /// </summary>
    public abstract string Render(IReadOnlyList<ChatMessage> messages, IReadOnlyList<ToolDefinition> tools, bool? think);
}

/// <summary>
/// The ChatML layout used by Qwen-family models: &lt;|im_start|&gt;role\ncontent&lt;|im_end|&gt;\n per message, tools
/// listed in the system turn inside &lt;tools&gt;&lt;/tools&gt;, tool calls as &lt;tool_call&gt;JSON&lt;/tool_call&gt;, tool
/// results as a user turn with &lt;tool_response&gt;&lt;/tool_response&gt;, and reasoning in &lt;think&gt;&lt;/think&gt;.
/// </summary>
public sealed class ChatMLTemplate : ChatTemplate
{
    private const string Start = "<|im_start|>", End = "<|im_end|>";

    /// <inheritdoc />
    public override IReadOnlyList<string> StopSequences { get; } = [End, Start];

    /// <inheritdoc />
    public override string Render(IReadOnlyList<ChatMessage> messages, IReadOnlyList<ToolDefinition> tools, bool? think)
    {
        var sb = new StringBuilder();
        string system = messages.FirstOrDefault(m => m.Role == "system")?.Content ?? "";
        if (tools.Count > 0)
        {
            system += (system.Length > 0 ? "\n\n" : "") + "# Tools\n\nYou may call one or more functions. Function signatures:\n<tools>\n" +
                      string.Join("\n", tools.Select(Describe)) + "\n</tools>\n\nFor each call, return " + ToolCallTags.Open +
                      "{\"name\": <function-name>, \"arguments\": <args-json-object>}" + ToolCallTags.Close;
        }

        if (system.Length > 0)
        {
            sb.Append(Start).Append("system\n").Append(system).Append(End).Append('\n');
        }

        foreach (var m in messages.Where(m => m.Role != "system"))
        {
            switch (m.Role)
            {
                case "tool":
                    sb.Append(Start).Append("user\n<tool_response>\n").Append(m.Content).Append("\n</tool_response>").Append(End).Append('\n');
                    break;
                case "assistant":
                    sb.Append(Start).Append("assistant\n").Append(m.Content);
                    foreach (var call in m.ToolCalls ?? [])
                    {
                        sb.Append('\n').Append(ToolCallTags.Open)
                          .Append(new JsonObject { ["name"] = call.Name, ["arguments"] = call.Arguments.DeepClone() }.ToJsonString())
                          .Append(ToolCallTags.Close);
                    }

                    sb.Append(End).Append('\n');
                    break;
                default:
                    sb.Append(Start).Append(m.Role).Append('\n').Append(m.Content).Append(End).Append('\n');
                    break;
            }
        }

        sb.Append(Start).Append("assistant\n");
        if (think == false)
        {
            sb.Append(ThinkTags.Open).Append("\n\n").Append(ThinkTags.Close).Append("\n\n");
        }

        return sb.ToString();
    }

    private static string Describe(ToolDefinition tool) =>
        new JsonObject
        {
            ["type"] = "function",
            ["function"] = new JsonObject
            {
                ["name"] = tool.Name,
                ["description"] = tool.Description,
                ["parameters"] = tool.Parameters?.DeepClone(),
            },
        }.ToJsonString();
}

/// <summary>What one piece of streamed assistant output added.</summary>
/// <param name="Content">New answer text.</param>
/// <param name="Thinking">New reasoning text.</param>
/// <param name="ToolCalls">Tool calls completed in this piece.</param>
public sealed record ChatDelta(string Content, string Thinking, IReadOnlyList<ToolCall> ToolCalls)
{
    /// <summary>True when the piece added nothing.</summary>
    public bool IsEmpty => Content.Length == 0 && Thinking.Length == 0 && ToolCalls.Count == 0;
}

/// <summary>
/// Splits streamed assistant text into reasoning (inside the think tags), answer text and tool calls (JSON inside the
/// tool-call tags). Feed text as it arrives; partial tags are held back until they can be decided.
/// </summary>
public sealed class ChatOutputParser(ChatTemplate template, bool separateThinking = true)
{
    private enum Mode { Start, Content, Thinking, ToolCall }

    private readonly StringBuilder _pending = new();
    private readonly StringBuilder _toolText = new();
    private Mode _mode = Mode.Start;
    private bool _afterThinking;

    /// <summary>Processes new text.</summary>
    public ChatDelta Feed(string text)
    {
        _pending.Append(text);
        return Drain(final: false);
    }

    /// <summary>Flushes everything held back at the end of the stream.</summary>
    public ChatDelta Finish() => Drain(final: true);

    private ChatDelta Drain(bool final)
    {
        var content = new StringBuilder();
        var thinking = new StringBuilder();
        var calls = new List<ToolCall>();
        var (thinkOpen, thinkClose) = template.ThinkTags;
        var (callOpen, callClose) = template.ToolCallTags;

        while (_pending.Length > 0 || (final && _mode == Mode.ToolCall && _toolText.Length > 0))
        {
            string p = _pending.ToString();
            switch (_mode)
            {
                case Mode.Start:
                {
                    // Reasoning may only open the turn (after optional whitespace).
                    string trimmed = p.TrimStart();
                    if (trimmed.StartsWith(thinkOpen, StringComparison.Ordinal))
                    {
                        _pending.Remove(0, p.Length - trimmed.Length + thinkOpen.Length);
                        _mode = Mode.Thinking;
                        continue;
                    }

                    if (!final && (trimmed.Length == 0 || thinkOpen.StartsWith(trimmed, StringComparison.Ordinal)))
                    {
                        return Result();                                          // still undecided
                    }

                    _mode = Mode.Content;
                    continue;
                }

                case Mode.Thinking:
                {
                    int close = p.IndexOf(thinkClose, StringComparison.Ordinal);
                    if (close >= 0)
                    {
                        thinking.Append(p[..close]);
                        _pending.Remove(0, close + thinkClose.Length);
                        _mode = Mode.Content;
                        _afterThinking = true;                               // drop the blank lines that follow reasoning
                        continue;
                    }

                    int safe = final ? p.Length : SafeLength(p, thinkClose);
                    thinking.Append(p[..safe]);
                    _pending.Remove(0, safe);
                    return Result();
                }

                case Mode.ToolCall:
                {
                    // Accumulate first, so a closing tag split across chunks is still found.
                    _toolText.Append(p);
                    _pending.Clear();
                    string all = _toolText.ToString();
                    int close = all.IndexOf(callClose, StringComparison.Ordinal);
                    if (close < 0 && !final)
                    {
                        return Result();
                    }

                    string json = close >= 0 ? all[..close] : all;
                    if (close >= 0)
                    {
                        _pending.Append(all, close + callClose.Length, all.Length - close - callClose.Length);
                    }

                    if (TryParseCall(json) is { } call)
                    {
                        calls.Add(call);
                    }
                    else
                    {
                        content.Append(callOpen).Append(json).Append(close >= 0 ? callClose : "");   // not a valid call: keep as text
                    }

                    _toolText.Clear();
                    _mode = Mode.Content;
                    continue;
                }

                default:
                {
                    if (_afterThinking)
                    {
                        int skip = 0;
                        while (skip < p.Length && p[skip] is '\n' or '\r')
                        {
                            skip++;
                        }

                        _pending.Remove(0, skip);
                        if (skip == p.Length)
                        {
                            return Result();                                   // only newlines so far: keep waiting
                        }

                        _afterThinking = false;
                        p = _pending.ToString();
                    }

                    int open = p.IndexOf(callOpen, StringComparison.Ordinal);
                    if (open >= 0)
                    {
                        content.Append(p[..open]);
                        _pending.Remove(0, open + callOpen.Length);
                        _mode = Mode.ToolCall;
                        continue;
                    }

                    int safe = final ? p.Length : SafeLength(p, callOpen);
                    content.Append(p[..safe]);
                    _pending.Remove(0, safe);
                    return Result();
                }
            }
        }

        return Result();

        ChatDelta Result()
        {
            string c = content.ToString(), t = thinking.ToString();
            if (!separateThinking && t.Length > 0)
            {
                c = t + c;
                t = "";
            }

            return new ChatDelta(c, t, calls);
        }
    }

    /// <summary>Length of <paramref name="text"/> that cannot be the start of <paramref name="tag"/>.</summary>
    private static int SafeLength(string text, string tag)
    {
        for (int keep = Math.Min(tag.Length - 1, text.Length); keep > 0; keep--)
        {
            if (tag.StartsWith(text[^keep..], StringComparison.Ordinal))
            {
                return text.Length - keep;
            }
        }

        return text.Length;
    }

    private static ToolCall? TryParseCall(string json)
    {
        try
        {
            if (JsonNode.Parse(json.Trim()) is JsonObject o && o["name"]?.GetValue<string>() is { Length: > 0 } name)
            {
                var args = o["arguments"] switch
                {
                    JsonObject a => (JsonObject)a.DeepClone(),
                    JsonValue v when v.TryGetValue<string>(out var s) && JsonNode.Parse(s) is JsonObject parsed => parsed,
                    _ => new JsonObject(),
                };
                return new ToolCall(name, args);
            }
        }
        catch (JsonException)
        {
        }
        catch (InvalidOperationException)
        {
        }

        return null;
    }
}

namespace NeuralSharp.Generation;

/// <summary>A chat request: the conversation, optional tools, reasoning mode and generation options.</summary>
/// <param name="Messages">The conversation so far (system, user, assistant and tool messages).</param>
/// <param name="Tools">Functions the model may call.</param>
/// <param name="Think">true: return reasoning separately; false: suppress it; null: model default (returned separately if produced).</param>
/// <param name="Options">Sampling and length options; the template's stop sequences are added to <see cref="GenerationOptions.Stop"/>.</param>
public sealed record ChatRequest(IReadOnlyList<ChatMessage> Messages, IReadOnlyList<ToolDefinition>? Tools = null, bool? Think = null,
    GenerationOptions? Options = null);

/// <summary>A streamed piece of the assistant's reply; the final one carries the reason, the full message and statistics.</summary>
/// <param name="Delta">What this piece added (content, reasoning, completed tool calls).</param>
/// <param name="Done">True for the final piece.</param>
/// <param name="DoneReason">"stop" or "length" on the final piece.</param>
/// <param name="Message">The complete assistant message, on the final piece.</param>
/// <param name="Stats">Statistics, on the final piece.</param>
public sealed record ChatChunk(ChatDelta Delta, bool Done = false, string? DoneReason = null, ChatMessage? Message = null, GenerationStats? Stats = null);

/// <summary>
/// Chat on top of a <see cref="TextGenerator"/>: renders the conversation with a <see cref="ChatTemplate"/>, generates
/// until the end-of-turn marker, and splits the output into reasoning, answer and tool calls as it streams.
/// </summary>
public sealed class ChatGenerator(TextGenerator generator, ChatTemplate? template = null)
{
    /// <summary>The underlying text generator.</summary>
    public TextGenerator Generator { get; } = generator;

    /// <summary>The prompt format.</summary>
    public ChatTemplate Template { get; } = template ?? new ChatMLTemplate();

    /// <summary>The prompt text for a request (useful for debugging templates).</summary>
    public string RenderPrompt(ChatRequest request) => Template.Render(request.Messages, request.Tools ?? [], request.Think);

    /// <summary>Generates the complete reply.</summary>
    public ChatChunk Chat(ChatRequest request, CancellationToken cancellationToken = default) =>
        Stream(request, cancellationToken).Last();

    /// <summary>Streams the reply.</summary>
    public IEnumerable<ChatChunk> Stream(ChatRequest request, CancellationToken cancellationToken = default)
    {
        var options = request.Options ?? new GenerationOptions();
        options = options with { Stop = [.. options.Stop, .. Template.StopSequences] };
        var parser = new ChatOutputParser(Template, separateThinking: request.Think != false);
        var content = new System.Text.StringBuilder();
        var thinking = new System.Text.StringBuilder();
        var calls = new List<ToolCall>();

        foreach (var chunk in Generator.Stream(RenderPrompt(request), options, cancellationToken))
        {
            var delta = chunk.Done ? parser.Finish() : parser.Feed(chunk.Text);
            content.Append(delta.Content);
            thinking.Append(delta.Thinking);
            calls.AddRange(delta.ToolCalls);
            if (chunk.Done)
            {
                var message = new ChatMessage("assistant", content.ToString().Trim(), thinking.Length > 0 ? thinking.ToString().Trim() : null,
                    calls.Count > 0 ? calls : null);
                yield return new ChatChunk(delta, true, chunk.DoneReason, message, chunk.Stats);
            }
            else if (!delta.IsEmpty)
            {
                yield return new ChatChunk(delta);
            }
        }
    }
}

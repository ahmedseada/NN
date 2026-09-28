using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Nodes;
using NeuralSharp.Generation;
using NeuralSharp.Layers;
using NeuralSharp.Optimizers;

namespace NeuralSharp.Pretrained;

/// <summary>
/// One conversation to fine-tune on: messages (system, user, assistant with optional reasoning and tool calls, tool
/// results) and the tools that were available. Read agent transcripts with <see cref="ReadJsonLines"/>.
/// </summary>
/// <param name="Messages">The conversation.</param>
/// <param name="Tools">The tools offered to the model (rendered by the chat template as the model saw them).</param>
/// <param name="Think">The reasoning mode passed to the template (enable_thinking), or null for the template's default.</param>
public sealed record ChatTranscript(IReadOnlyList<ChatMessage> Messages, IReadOnlyList<ToolDefinition> Tools, bool? Think = null)
{
    /// <summary>
    /// Reads one transcript: <c>{"messages": [...], "tools": [...]}</c> in the OpenAI / Hugging Face chat format
    /// (content as a string or a list of text parts; reasoning in reasoning_content, thinking or reasoning; tool calls as
    /// [{"function": {"name", "arguments"}}] with arguments as an object or a JSON string; tool results with role "tool"
    /// and a name or a tool_call_id), or ShareGPT's <c>{"conversations": [{"from": "human" | "gpt" | "system", "value"}]}</c>.
    /// Optional "enable_thinking" / "think" sets <see cref="Think"/>.
    /// </summary>
    public static ChatTranscript FromJson(JsonObject json)
    {
        var messages = new List<ChatMessage>();
        var callNames = new Dictionary<string, string>(StringComparer.Ordinal);
        if (json["messages"] is JsonArray list)
        {
            foreach (var node in list)
            {
                messages.Add(Message(node as JsonObject ?? throw new InvalidDataException("A message is not an object."), callNames));
            }
        }
        else if (json["conversations"] is JsonArray shareGpt)
        {
            foreach (var node in shareGpt)
            {
                string from = (string?)node?["from"] ?? "";
                string role = from switch
                {
                    "human" or "user" => "user",
                    "gpt" or "assistant" or "model" => "assistant",
                    "system" => "system",
                    "tool" or "observation" or "function_response" => "tool",
                    _ => throw new InvalidDataException($"Unknown ShareGPT speaker '{from}'."),
                };
                messages.Add(new ChatMessage(role, (string?)node?["value"] ?? ""));
            }
        }
        else
        {
            throw new InvalidDataException("A transcript needs \"messages\" (or ShareGPT \"conversations\").");
        }

        var tools = new List<ToolDefinition>();
        foreach (var node in json["tools"] as JsonArray ?? [])
        {
            var tool = node?["function"] as JsonObject ?? node as JsonObject ?? throw new InvalidDataException("A tool is not an object.");
            tools.Add(new ToolDefinition((string?)tool["name"] ?? throw new InvalidDataException("A tool has no name."),
                (string?)tool["description"], tool["parameters"]?.DeepClone()));
        }

        bool? think = json["enable_thinking"] is JsonValue e ? (bool)e : json["think"] is JsonValue t ? (bool)t : null;
        return new ChatTranscript(messages, tools, think);
    }

    private static ChatMessage Message(JsonObject m, Dictionary<string, string> callNames)
    {
        string role = (string?)m["role"] ?? throw new InvalidDataException("A message has no role.");
        string content = m["content"] switch
        {
            null => "",
            JsonValue v => (string?)v ?? "",
            JsonArray parts => string.Concat(parts.Select(p => p is JsonValue pv ? (string?)pv : (string?)p?["text"] ?? "")),
            _ => throw new InvalidDataException($"Unsupported content in a {role} message."),
        };
        string? thinking = (string?)m["reasoning_content"] ?? (string?)m["thinking"] ?? (string?)m["reasoning"];
        List<ToolCall>? calls = null;
        if (m["tool_calls"] is JsonArray toolCalls && toolCalls.Count > 0)
        {
            calls = [];
            foreach (var node in toolCalls)
            {
                var function = node?["function"] as JsonObject ?? node as JsonObject ?? throw new InvalidDataException("A tool call is not an object.");
                string name = (string?)function["name"] ?? throw new InvalidDataException("A tool call has no name.");
                var arguments = function["arguments"] switch
                {
                    JsonObject o => (JsonObject)o.DeepClone(),
                    JsonValue s when (string?)s is { } text => ParseArguments(text, name),
                    null => new JsonObject(),
                    _ => throw new InvalidDataException($"The arguments of a call to {name} are neither an object nor a JSON string."),
                };
                calls.Add(new ToolCall(name, arguments));
                if ((string?)node?["id"] is { } id)
                {
                    callNames[id] = name;
                }
            }
        }

        string? toolName = (string?)m["name"];
        if (toolName is null && (string?)m["tool_call_id"] is { } callId && callNames.TryGetValue(callId, out var called))
        {
            toolName = called;
        }

        return new ChatMessage(role, content, string.IsNullOrEmpty(thinking) ? null : thinking, calls, role == "tool" ? toolName : null);
    }

    private static JsonObject ParseArguments(string text, string name)
    {
        try
        {
            return JsonNode.Parse(text) as JsonObject ?? throw new InvalidDataException($"The arguments of a call to {name} are not a JSON object.");
        }
        catch (JsonException ex)
        {
            throw new InvalidDataException($"The arguments of a call to {name} are not valid JSON: {ex.Message}", ex);
        }
    }

    /// <summary>Reads one transcript per non-empty line of a JSON Lines file (see <see cref="FromJson"/>).</summary>
    public static IEnumerable<ChatTranscript> ReadJsonLines(string path)
    {
        int number = 0;
        foreach (string line in File.ReadLines(path))
        {
            number++;
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            ChatTranscript transcript;
            try
            {
                transcript = FromJson(JsonNode.Parse(line) as JsonObject ?? throw new InvalidDataException("The line is not a JSON object."));
            }
            catch (Exception ex) when (ex is JsonException or InvalidDataException or InvalidOperationException or FormatException)
            {
                throw new InvalidDataException($"{path}, line {number}: {ex.Message}", ex);
            }

            yield return transcript;
        }
    }
}

/// <summary>
/// One tokenized training sequence: its token ids and, for each token, whether the model learns to produce it (the
/// assistant's turns: reasoning, text, tool calls and the end-of-turn marker; not the system, user or tool messages).
/// </summary>
public sealed record TrainingSequence(int[] Tokens, bool[] Trained)
{
    /// <summary>Tokens the loss is computed on (a token is predicted from the ones before it, so the first never is).</summary>
    public int TrainedTokens => Trained.Skip(1).Count(t => t);
}

/// <summary>
/// Turns transcripts into <see cref="TrainingSequence"/>s with a model's own chat template, so the fine-tuning data has
/// exactly the layout the model sees when it is used (tool definitions, tool-call syntax, reasoning blocks). The
/// assistant's turns are found from the template itself: the text its generation prompt adds (the assistant header)
/// and the text that ends an assistant message, discovered by rendering a probe conversation.
/// </summary>
public sealed class ChatTranscriptEncoder
{
    private const string UserProbe = "⁠ns-user-probe⁠", AssistantProbe = "⁠ns-assistant-probe⁠";

    /// <summary>Creates the encoder for <paramref name="template"/> and <paramref name="tokenizer"/>.</summary>
    public ChatTranscriptEncoder(JinjaChatTemplate template, ITokenizer tokenizer)
    {
        Template = template;
        Tokenizer = tokenizer;
        var user = new ChatMessage("user", UserProbe);
        string prompt = template.Render([user], [], null, addGenerationPrompt: true);
        string bare = template.Render([user], [], null, addGenerationPrompt: false);
        string pair = template.Render([user, new ChatMessage("assistant", AssistantProbe)], [], null, addGenerationPrompt: false);
        int answer = pair.IndexOf(AssistantProbe, StringComparison.Ordinal);
        if (answer < 0)
        {
            throw new InvalidOperationException("The chat template does not render an assistant message's content.");
        }

        string header = prompt.StartsWith(bare, StringComparison.Ordinal) ? prompt[bare.Length..] : "";
        if (header.Trim().Length == 0)
        {
            // No generation prompt (e.g. [INST] … [/INST] templates): the text between the user's content and the answer.
            int userEnd = pair.IndexOf(UserProbe, StringComparison.Ordinal) + UserProbe.Length;
            header = pair[userEnd..answer].Trim();
        }

        string end = pair[(answer + AssistantProbe.Length)..].TrimEnd();
        AssistantHeader = header;
        AssistantEnd = end.Length > 0 ? end : template.EosToken;
        if (AssistantHeader.Length == 0)
        {
            throw new InvalidOperationException("Could not find how the chat template starts an assistant turn.");
        }
    }

    /// <summary>The template.</summary>
    public JinjaChatTemplate Template { get; }

    /// <summary>The tokenizer.</summary>
    public ITokenizer Tokenizer { get; }

    /// <summary>The text that starts an assistant turn (for ChatML: "&lt;|im_start|&gt;assistant\n").</summary>
    public string AssistantHeader { get; }

    /// <summary>The text that ends an assistant turn (for ChatML: "&lt;|im_end|&gt;"); trained, so the model learns to stop.</summary>
    public string AssistantEnd { get; }

    /// <summary>The rendered transcript and the character ranges [start, end) of the assistant's turns in it.</summary>
    public (string Text, IReadOnlyList<(int Start, int End)> Spans) Render(ChatTranscript transcript)
    {
        string text = Template.Render(transcript.Messages, transcript.Tools, transcript.Think, addGenerationPrompt: false);
        var spans = new List<(int, int)>();
        for (int at = text.IndexOf(AssistantHeader, StringComparison.Ordinal); at >= 0; at = text.IndexOf(AssistantHeader, at, StringComparison.Ordinal))
        {
            int start = at + AssistantHeader.Length;
            int end = AssistantEnd.Length == 0 ? -1 : text.IndexOf(AssistantEnd, start, StringComparison.Ordinal);
            end = end < 0 ? text.Length : end + AssistantEnd.Length;
            spans.Add((start, end));
            at = end;
        }

        return (text, spans);
    }

    /// <summary>
    /// The transcript's tokens, with the assistant's turns marked as trained, cut to <paramref name="maxLength"/> + 1
    /// tokens (inputs and targets are the sequence shifted by one); null when nothing trainable remains.
    /// </summary>
    public TrainingSequence? Encode(ChatTranscript transcript, int maxLength)
    {
        var (text, spans) = Render(transcript);
        var tokens = new List<int>();
        var trained = new List<bool>();
        void Add(int from, int to, bool train)
        {
            if (to > from)
            {
                var ids = Tokenizer.Encode(text[from..to]);
                tokens.AddRange(ids);
                trained.AddRange(Enumerable.Repeat(train, ids.Count));
            }
        }

        int position = 0;
        foreach (var (start, end) in spans)
        {
            Add(position, start, false);
            Add(start, end, true);
            position = end;
        }

        Add(position, text.Length, false);
        int keep = Math.Min(tokens.Count, maxLength + 1);
        var sequence = new TrainingSequence([.. tokens.Take(keep)], [.. trained.Take(keep)]);
        return sequence.TrainedTokens > 0 ? sequence : null;
    }

    /// <summary>
    /// Plain text for continued pre-training or domain adaptation: the template's BOS token, the text and its EOS token,
    /// every token trained. Longer texts become several sequences of at most <paramref name="maxLength"/> + 1 tokens that
    /// together predict every token once.
    /// </summary>
    public IEnumerable<TrainingSequence> EncodeText(string text, int maxLength)
    {
        var ids = Tokenizer.Encode(Template.BosToken + text + Template.EosToken);
        for (int start = 0; start + 1 < ids.Count; start += maxLength)
        {
            int count = Math.Min(maxLength + 1, ids.Count - start);
            if (count < 2)
            {
                yield break;
            }

            yield return new TrainingSequence([.. ids.Skip(start).Take(count)], [.. Enumerable.Repeat(true, count)]);
        }
    }

    /// <summary>
    /// A dataset row as training sequences: a conversation (<c>{"messages": [...], "tools": [...]}</c>, see
    /// <see cref="ChatTranscript.FromJson"/>) trains the assistant's turns; a text row (<c>{"text": ...}</c>) trains every
    /// token. Other rows give nothing.
    /// </summary>
    public IEnumerable<TrainingSequence> EncodeRow(JsonObject row, int maxLength)
    {
        if (row.ContainsKey("messages") || row.ContainsKey("conversations"))
        {
            if (Encode(ChatTranscript.FromJson(row), maxLength) is { } sequence)
            {
                yield return sequence;
            }
        }
        else if (row["text"] is JsonValue v && v.TryGetValue<string>(out var text) && text.Length > 0)
        {
            foreach (var sequence in EncodeText(text, maxLength))
            {
                yield return sequence;
            }
        }
    }
}

/// <summary>Settings for <see cref="FineTuner"/>.</summary>
public sealed record FineTuningOptions
{
    /// <summary>LoRA rank (the adapters' inner size).</summary>
    public int Rank { get; init; } = 16;

    /// <summary>LoRA alpha: the adapters' output is scaled by alpha / rank.</summary>
    public float Alpha { get; init; } = 32f;

    /// <summary>Layers that get adapters, by name: q, k, v, o (attention), gate, up, down (feed-forward), head.</summary>
    public IReadOnlyList<string> Targets { get; init; } = ["q", "k", "v", "o", "gate", "up", "down"];

    /// <summary>Peak learning rate (reached after the warm-up, then cosine decay to <see cref="MinLearningRate"/>).</summary>
    public float LearningRate { get; init; } = 2e-4f;

    /// <summary>Learning rate at the end of training.</summary>
    public float MinLearningRate { get; init; }

    /// <summary>Fraction of the optimizer steps spent warming up linearly.</summary>
    public float WarmupFraction { get; init; } = 0.03f;

    /// <summary>AdamW weight decay on the adapters.</summary>
    public float WeightDecay { get; init; }

    /// <summary>Passes over the training data.</summary>
    public int Epochs { get; init; } = 1;

    /// <summary>Longest sequence in tokens (longer transcripts are cut).</summary>
    public int MaxLength { get; init; } = 2048;

    /// <summary>Token budget of one batch (sequences of similar length are batched and padded to the longest).</summary>
    public int BatchTokens { get; init; } = 4096;

    /// <summary>Batches whose gradients are added before each optimizer step.</summary>
    public int GradientAccumulation { get; init; } = 1;

    /// <summary>Gradients are scaled down to at most this norm (0: no clipping).</summary>
    public float MaxGradientNorm { get; init; } = 1f;

    /// <summary>Rows of the output head computed at once by the loss (see <see cref="Losses.TokenCrossEntropy"/>).</summary>
    public int LossChunkRows { get; init; } = 512;

    /// <summary>Evaluate every this many optimizer steps (0: after each epoch).</summary>
    public int EvaluateEvery { get; init; }

    /// <summary>Save the adapters every this many optimizer steps (0: only at the end).</summary>
    public int SaveEvery { get; init; }

    /// <summary>
    /// Recompute each decoder block's activations in the backward pass instead of storing them (about 20× less
    /// activation memory for about a third more compute): needed for long sequences or large batches.
    /// </summary>
    public bool Checkpointing { get; init; } = true;

    /// <summary>Seed for the adapters' initial values and the batch order.</summary>
    public int Seed { get; init; }
}

/// <summary>Where a fine-tuning run is.</summary>
/// <param name="Step">Optimizer steps done.</param>
/// <param name="TotalSteps">Optimizer steps planned.</param>
/// <param name="Epoch">The current epoch (from 1).</param>
/// <param name="Loss">Mean loss per trained token over the last step.</param>
/// <param name="LearningRate">The learning rate of the last step.</param>
/// <param name="TokensPerSecond">Tokens (including prompts) processed per second over the last step.</param>
/// <param name="EvaluationLoss">The latest evaluation loss, when one was computed at this step.</param>
public sealed record FineTuningProgress(int Step, int TotalSteps, int Epoch, float Loss, float LearningRate, double TokensPerSecond, float? EvaluationLoss);

/// <summary>
/// Fine-tunes a pretrained model with LoRA adapters: the base weights stay as loaded (int4, int8 or bfloat16 for QLoRA,
/// or float32) and frozen; only the adapters train, on the assistant tokens of chat transcripts (see
/// <see cref="ChatTranscriptEncoder"/>). AdamW with a linear warm-up and cosine decay, gradient accumulation and
/// clipping, and a token loss that never stores the full logits (<see cref="Losses.TokenCrossEntropy"/>).
/// </summary>
public static class FineTuner
{
    /// <summary>
    /// Adds adapters to <paramref name="model"/> (unless it already has some) and trains them on <paramref name="train"/>;
    /// evaluates on <paramref name="evaluation"/> when given, and writes the adapters to <paramref name="outputFolder"/>
    /// (PEFT format, see <see cref="PretrainedModel.SaveAdapter"/>) when given. Returns the evaluation losses.
    /// </summary>
    /// <param name="model">The model (adapters are added unless it has some).</param>
    /// <param name="train">Training sequences.</param>
    /// <param name="evaluation">Sequences to evaluate on, or null.</param>
    /// <param name="options">Settings.</param>
    /// <param name="outputFolder">Where to write the adapters (and checkpoints), or null.</param>
    /// <param name="progress">Receives one report per optimizer step.</param>
    /// <param name="cancellationToken">Stops between batches.</param>
    /// <param name="trace">Receives a line per batch (its shape and time) and per evaluation, as they finish.</param>
    public static IReadOnlyList<float> Train(PretrainedModel model, IReadOnlyList<TrainingSequence> train, IReadOnlyList<TrainingSequence>? evaluation,
        FineTuningOptions options, string? outputFolder = null, IProgress<FineTuningProgress>? progress = null, CancellationToken cancellationToken = default,
        Action<string>? trace = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(options);
        if (train.Count == 0)
        {
            throw new ArgumentException("There is nothing to train on.", nameof(train));
        }

        var network = model.Network;
        if (!network.Descendants().OfType<Linear>().Any(l => l.Adapter is not null))
        {
            model.AddAdapters(options.Rank, options.Alpha, options.Targets, options.Seed);
        }

        var parameters = network.TrainableParameters().ToList();
        using var optimizer = new AdamW(parameters, options.LearningRate, weightDecay: options.WeightDecay);
        var random = new Random(options.Seed);
        var epochBatches = Enumerable.Range(0, options.Epochs).Select(_ => Batches(train, options.BatchTokens, random)).ToList();
        int accumulation = Math.Max(1, options.GradientAccumulation);
        int totalSteps = epochBatches.Sum(b => (b.Count + accumulation - 1) / accumulation);
        var schedule = new CosineAnnealing(optimizer, Math.Max(1, totalSteps), options.MinLearningRate,
            (int)Math.Round(options.WarmupFraction * totalSteps));
        var evaluations = new List<float>();
        int step = 0;
        for (int epoch = 0; epoch < options.Epochs; epoch++)
        {
            var batches = epochBatches[epoch];
            for (int first = 0; first < batches.Count; first += accumulation)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var group = batches.Skip(first).Take(accumulation).ToList();
                float normalizer = Math.Max(1, group.Sum(b => b.Sum(i => train[i].TrainedTokens)));
                var watch = Stopwatch.StartNew();
                float loss = 0f;
                long tokens = 0;
                network.Train();
                optimizer.ZeroGrad();
                for (int b = 0; b < group.Count; b++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var batch = group[b];
                    var batchWatch = Stopwatch.StartNew();
                    using var scope = new TensorScope();
                    trace?.Invoke($"step {step + 1}/{totalSteps}, batch {first + b + 1}/{batches.Count} of epoch {epoch + 1}: {batch.Length} sequences × "
                        + $"{batch.Max(i => train[i].Tokens.Length) - 1} tokens…");
                    var (lossTensor, count) = BatchLoss(model, train, batch, normalizer, options.LossChunkRows, options.Checkpointing);
                    float batchLoss = lossTensor.Item();                             // waits for the forward pass
                    double forward = batchWatch.Elapsed.TotalSeconds;
                    lossTensor.Backward();
                    model.Device.Synchronize();
                    loss += batchLoss;
                    tokens += count;
                    trace?.Invoke($"  forward {forward:F2} s, backward {batchWatch.Elapsed.TotalSeconds - forward:F2} s, loss {batchLoss * normalizer / Math.Max(1, batch.Sum(i => train[i].TrainedTokens)):F4}");
                }

                if (options.MaxGradientNorm > 0f)
                {
                    optimizer.ClipGradientNorm(options.MaxGradientNorm);
                }

                float rate = optimizer.LearningRate;
                optimizer.Step();
                schedule.Step();
                step++;
                float? evaluationLoss = null;
                bool lastOfEpoch = first + accumulation >= batches.Count;
                if (evaluation is { Count: > 0 } && (options.EvaluateEvery > 0 ? step % options.EvaluateEvery == 0 : lastOfEpoch))
                {
                    evaluationLoss = Evaluate(model, evaluation, options.BatchTokens, options.LossChunkRows, trace);
                    evaluations.Add(evaluationLoss.Value);
                }

                if (outputFolder is not null && options.SaveEvery > 0 && step % options.SaveEvery == 0)
                {
                    model.SaveAdapter(Path.Combine(outputFolder, $"checkpoint-{step}"));
                }

                progress?.Report(new FineTuningProgress(step, totalSteps, epoch + 1, loss, rate, tokens / Math.Max(1e-9, watch.Elapsed.TotalSeconds), evaluationLoss));
            }
        }

        network.Eval();
        if (outputFolder is not null)
        {
            model.SaveAdapter(outputFolder);
        }

        return evaluations;
    }

    /// <summary>Mean loss per trained token of <paramref name="sequences"/> (no gradients).</summary>
    public static float Evaluate(PretrainedModel model, IReadOnlyList<TrainingSequence> sequences, int batchTokens = 4096, int chunkRows = 512,
        Action<string>? trace = null)
    {
        model.Network.Eval();
        double total = 0;
        long trained = 0;
        using (Autograd.NoGrad())
        {
            var batches = Batches(sequences, batchTokens, random: null);
            for (int b = 0; b < batches.Count; b++)
            {
                var batch = batches[b];
                var watch = Stopwatch.StartNew();
                using var scope = new TensorScope();
                int count = batch.Sum(i => sequences[i].TrainedTokens);
                var (loss, _) = BatchLoss(model, sequences, batch, 1f, chunkRows);
                total += loss.Item();
                trained += count;
                trace?.Invoke($"evaluation batch {b + 1}/{batches.Count}: {batch.Length} sequences × {batch.Max(i => sequences[i].Tokens.Length) - 1} tokens, {watch.Elapsed.TotalSeconds:F2} s");
            }
        }

        return (float)(total / Math.Max(1, trained));
    }

    /// <summary>
    /// Groups sequences of similar length into batches of at most <paramref name="batchTokens"/> padded tokens (one
    /// sequence per batch when a sequence alone exceeds it); the batch order is shuffled when <paramref name="random"/> is given.
    /// </summary>
    public static List<int[]> Batches(IReadOnlyList<TrainingSequence> sequences, int batchTokens, Random? random)
    {
        var order = Enumerable.Range(0, sequences.Count).OrderBy(i => sequences[i].Tokens.Length).ToList();
        var batches = new List<int[]>();
        var current = new List<int>();
        int longest = 0;
        foreach (int index in order)
        {
            int length = sequences[index].Tokens.Length - 1;
            if (length < 1)
            {
                continue;
            }

            if (current.Count > 0 && (current.Count + 1) * Math.Max(longest, length) > batchTokens)
            {
                batches.Add([.. current]);
                current.Clear();
                longest = 0;
            }

            current.Add(index);
            longest = Math.Max(longest, length);
        }

        if (current.Count > 0)
        {
            batches.Add([.. current]);
        }

        if (random is not null)
        {
            for (int i = batches.Count - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);
                (batches[i], batches[j]) = (batches[j], batches[i]);
            }
        }

        return batches;
    }

    // The summed weighted loss of one batch divided by normalizer (padding and untrained positions weigh 0), and the
    // number of tokens it covers. The network runs up to its final normalization; the head runs inside the loss.
    private static (Tensor Loss, long Tokens) BatchLoss(PretrainedModel model, IReadOnlyList<TrainingSequence> sequences, int[] batch, float normalizer,
        int chunkRows, bool checkpointing = false)
    {
        int length = batch.Max(i => sequences[i].Tokens.Length) - 1;
        int rows = batch.Length;
        var inputs = new float[rows * length];
        var targets = new float[rows * length];
        var weights = new float[rows * length];
        long tokens = 0;
        for (int b = 0; b < rows; b++)
        {
            var sequence = sequences[batch[b]];
            for (int t = 0; t + 1 < sequence.Tokens.Length; t++)
            {
                inputs[b * length + t] = sequence.Tokens[t];
                targets[b * length + t] = sequence.Tokens[t + 1];
                weights[b * length + t] = sequence.Trained[t + 1] ? 1f : 0f;
                tokens++;
            }
        }

        var device = model.Device;
        var modules = model.Network.ToList();
        if (modules[^1] is not Linear head)
        {
            throw new InvalidOperationException("The network does not end with its output head (a Linear layer).");
        }

        var hidden = Tensor.From(inputs, [rows, length], device);
        for (int i = 0; i < modules.Count - 1; i++)
        {
            if (checkpointing && modules[i] is DecoderBlock && Autograd.IsEnabled)
            {
                hidden = modules[i].ForwardCheckpointed(hidden);                // keeps only the block's output
            }
            else if (!Autograd.IsEnabled)
            {
                // Evaluation: each module's intermediate results, and the previous activation, are freed at once.
                Tensor next;
                using (var scope = new TensorScope())
                {
                    next = scope.Keep(modules[i].Forward(hidden));
                }

                if (!ReferenceEquals(next, hidden))
                {
                    hidden.Dispose();
                }

                hidden = next;
            }
            else
            {
                hidden = modules[i].Forward(hidden);
            }
        }

        var loss = Losses.TokenCrossEntropy(hidden.Reshape(rows * length, hidden.Shape[^1]), h => head.Forward(h),
            Tensor.From(targets, [rows * length], device), Tensor.From(weights, [rows * length], device), normalizer, chunkRows);
        return (loss, tokens);
    }
}

using System.Diagnostics;
using NeuralSharp.Layers;

namespace NeuralSharp.Generation;

/// <summary>Timing and token counts of one generation (durations as in common LLM server APIs).</summary>
/// <param name="PromptTokens">Tokens of the (possibly truncated) prompt that were processed.</param>
/// <param name="PromptDuration">Time to process the prompt (prefill), including the first sampled token.</param>
/// <param name="GeneratedTokens">Tokens generated (stop text excluded).</param>
/// <param name="GenerationDuration">Time spent generating after the prompt.</param>
/// <param name="TotalDuration">Wall time of the whole call.</param>
/// <param name="ContextResets">Times the context window filled up and was re-read from its last half.</param>
public sealed record GenerationStats(int PromptTokens, TimeSpan PromptDuration, int GeneratedTokens, TimeSpan GenerationDuration,
    TimeSpan TotalDuration, int ContextResets)
{
    /// <summary>Generated tokens per second.</summary>
    public double TokensPerSecond => GenerationDuration.TotalSeconds > 0 ? GeneratedTokens / GenerationDuration.TotalSeconds : 0;
}

/// <summary>A piece of streamed output. The last chunk has <see cref="Done"/> set, a reason and the statistics.</summary>
/// <param name="Text">New text since the previous chunk (may be empty).</param>
/// <param name="Done">True for the final chunk.</param>
/// <param name="DoneReason">"stop" (a stop sequence was produced) or "length" (the token limit was reached); null until done.</param>
/// <param name="Stats">Statistics, on the final chunk only.</param>
public sealed record GenerationChunk(string Text, bool Done = false, string? DoneReason = null, GenerationStats? Stats = null);

/// <summary>
/// Autoregressive text generation for a causal language model built as a <see cref="Sequential"/> of
/// <see cref="ICachedModule"/>-capable layers (embedding, positional encoding, causal transformer blocks, norm, head).
/// Handles prompt truncation to the context window, KV-cache decoding with optional graph replay, on-device sampling
/// with every <see cref="GenerationOptions"/> filter and penalty, stop sequences and streaming.
/// </summary>
/// <example>
/// <code>
/// var generator = new TextGenerator(model, new CharTokenizer(vocabulary), contextLength: 64);
/// foreach (var chunk in generator.Stream("once upon a time", new GenerationOptions { NumPredict = 200 }))
///     Console.Write(chunk.Text);
/// </code>
/// </example>
public sealed class TextGenerator(Sequential model, ITokenizer tokenizer, int contextLength)
{
    /// <summary>Upper bound on generated tokens when <see cref="GenerationOptions.NumPredict"/> is negative.</summary>
    public const int MaxTokens = 4096;

    /// <summary>The language model.</summary>
    public Sequential Model { get; } = model;

    /// <summary>The tokenizer matching the model's vocabulary.</summary>
    public ITokenizer Tokenizer { get; } = tokenizer;

    /// <summary>The model's maximum context (its positional-encoding length).</summary>
    public int ContextLength { get; } = contextLength;

    /// <summary>The device the model's parameters live on.</summary>
    public Device Device => Model.Parameters().Concat(Model.Buffers()).First().Device;

    /// <summary>
    /// How the KV cache stores keys and values while generating: <see cref="KeyValueFormat.Int8"/> takes about a quarter
    /// of the memory (longer contexts, more parallel sequences) at a small cost in accuracy.
    /// </summary>
    public KeyValueFormat CacheFormat { get; init; } = KeyValueFormat.Float32;

    /// <summary>Generates the whole continuation of <paramref name="prompt"/>.</summary>
    public (string Text, string DoneReason, GenerationStats Stats) Generate(string prompt, GenerationOptions options, CancellationToken cancellationToken = default)
    {
        var text = new System.Text.StringBuilder();
        foreach (var chunk in Stream(prompt, options, cancellationToken))
        {
            text.Append(chunk.Text);
            if (chunk.Done)
            {
                return (text.ToString(), chunk.DoneReason!, chunk.Stats!);
            }
        }

        throw new InvalidOperationException("Generation ended without a final chunk.");
    }

    /// <summary>
    /// <see cref="Stream"/> on a background thread, as an <c>await foreach</c> stream: the same chunks, and the calling
    /// thread (a UI or request thread) is never blocked by the model.
    /// </summary>
    public IAsyncEnumerable<GenerationChunk> StreamAsync(string prompt, GenerationOptions options, CancellationToken cancellationToken = default) =>
        BackgroundStream.Run(token => Stream(prompt, options, token), cancellationToken);

    /// <summary><see cref="Generate"/> on a background thread.</summary>
    public Task<(string Text, string DoneReason, GenerationStats Stats)> GenerateAsync(string prompt, GenerationOptions options,
        CancellationToken cancellationToken = default) =>
        Task.Run(() => Generate(prompt, options, cancellationToken), CancellationToken.None);

    /// <summary>Streams the continuation of <paramref name="prompt"/> in chunks of about <see cref="GenerationOptions.ChunkSize"/> tokens.</summary>
    public IEnumerable<GenerationChunk> Stream(string prompt, GenerationOptions options, CancellationToken cancellationToken = default)
    {
        var total = Stopwatch.StartNew();
        int context = Math.Clamp(options.NumCtx, 2, ContextLength);
        int limit = options.NumPredict > 0 ? Math.Min(options.NumPredict, MaxTokens) : MaxTokens;
        var history = new List<int>(Tokenizer.Encode(prompt));
        if (history.Count == 0)
        {
            history.Add(0);
        }

        if (history.Count > context - 1)
        {
            history.RemoveRange(0, history.Count - (context - 1));          // keep the most recent part of the prompt
        }

        int promptTokens = history.Count;
        var stops = options.Stop.Where(s => s.Length > 0).ToArray();
        int holdBack = stops.Length == 0 ? 0 : stops.Max(s => s.Length) - 1;

        using var sampler = new TokenSampler(Device, 1, Tokenizer.VocabularySize, limit, Math.Max(1, options.RepeatLastN))
        {
            Temperature = options.Temperature,
            TopK = options.TopK,
            TopP = options.TopP,
            MinP = options.MinP,
            RepeatPenalty = options.RepeatPenalty,
            RepeatLastN = options.RepeatLastN,
            PresencePenalty = options.PresencePenalty,
            FrequencyPenalty = options.FrequencyPenalty,
            Seed = (uint)(options.Seed ?? Random.Shared.Next()),
        };
        sampler.SetHistory(history);

        var generated = new List<int>();
        var text = new System.Text.StringBuilder();
        int emitted = 0, read = 0, resets = 0;
        TimeSpan promptDuration = TimeSpan.Zero;
        string? doneReason = null;

        // Downloads sampled ids up to `produced`, extends the text, and checks stop sequences; returns the new text to emit.
        string Collect(int produced, bool final)
        {
            if (produced > read)
            {
                foreach (var step in sampler.Read(read, produced))
                {
                    generated.Add(step[0].Id);
                    history.Add(step[0].Id);
                }

                text.Append(Tokenizer.Decode(generated.Skip(read)));
                read = produced;
            }

            int end = text.Length;
            foreach (var stop in stops)
            {
                int at = text.ToString().IndexOf(stop, Math.Max(0, emitted - stop.Length + 1), StringComparison.Ordinal);
                if (at >= 0 && at < end)
                {
                    end = at;
                    doneReason = "stop";
                }
            }

            int safe = doneReason is not null || final ? end : Math.Max(emitted, end - holdBack);
            string piece = text.ToString(emitted, safe - emitted);
            emitted = safe;
            return piece;
        }

        Model.Eval();
        {
            // NoGrad is entered per compute call, never held across a yield (it is thread-local state of the caller).
            using var decoding = new DecodingContext(Device, 1, context, CacheFormat);
            ComputeGraph? graph = null;
            try
            {
                Tensor Window(int keep) => Tensor.From([.. history.TakeLast(keep).Select(i => (float)i)], [1, keep], Device);

                void Prefill(int keep)
                {
                    using var noGrad = Autograd.NoGrad();
                    using var scope = new TensorScope();
                    if (options.UseCache)
                    {
                        decoding.Reset();
                        sampler.Sample(Model.ForwardCached(Window(keep), decoding));
                    }
                    else
                    {
                        sampler.Sample(Model.Forward(Window(keep)));
                    }
                }

                void Step() => sampler.Sample(Model.ForwardCached(sampler.Ids.Reshape(1, 1), decoding));

                void RunStep()
                {
                    using var noGrad = Autograd.NoGrad();
                    using var scope = new TensorScope();
                    Step();
                }

                Prefill(history.Count);
                sampler.Read(0, 1);
                promptDuration = total.Elapsed;
                int produced = 1;
                while (doneReason is null && !cancellationToken.IsCancellationRequested)
                {
                    bool flush = produced % Math.Max(1, options.ChunkSize) == 0 || produced >= limit;
                    if (flush)
                    {
                        string piece = Collect(produced, final: false);
                        if (piece.Length > 0)
                        {
                            yield return new GenerationChunk(piece);
                        }
                    }

                    if (doneReason is not null || produced >= limit)
                    {
                        break;
                    }

                    if (!options.UseCache || decoding.Length >= context)
                    {
                        // Full recompute needs every id on the host; a full window is re-read from its last half.
                        string piece = Collect(produced, final: false);
                        if (piece.Length > 0)
                        {
                            yield return new GenerationChunk(piece);
                        }

                        if (doneReason is not null)
                        {
                            break;
                        }

                        if (options.UseCache)
                        {
                            Prefill(context / 2);
                            resets++;
                        }
                        else
                        {
                            Prefill(Math.Min(history.Count, context));
                        }
                    }
                    else
                    {
                        if (graph is null && options.UseGraph)
                        {
                            graph = decoding.CaptureStep(Step);
                        }

                        if (graph is not null)
                        {
                            decoding.ReplayStep(graph);
                        }
                        else
                        {
                            RunStep();
                        }
                    }

                    produced++;
                }

                string rest = Collect(produced, final: true);
                if (rest.Length > 0)
                {
                    yield return new GenerationChunk(rest);
                }
            }
            finally
            {
                graph?.Dispose();
            }
        }

        doneReason ??= "length";
        Device.Synchronize();
        var stats = new GenerationStats(promptTokens, promptDuration, Tokenizer.Encode(text.ToString(0, emitted)).Count, total.Elapsed - promptDuration,
            total.Elapsed, resets);
        yield return new GenerationChunk("", Done: true, DoneReason: doneReason, Stats: stats);
    }
}

/// <summary>Runs a synchronous stream on a thread-pool thread and hands its items to an asynchronous reader.</summary>
internal static class BackgroundStream
{
    public static async IAsyncEnumerable<T> Run<T>(Func<CancellationToken, IEnumerable<T>> source,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken)
    {
        using var stop = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        var channel = System.Threading.Channels.Channel.CreateUnbounded<T>(
            new System.Threading.Channels.UnboundedChannelOptions { SingleReader = true, SingleWriter = true });
        var producer = Task.Run(() =>
        {
            try
            {
                foreach (var item in source(stop.Token))
                {
                    channel.Writer.TryWrite(item);
                }

                channel.Writer.TryComplete();
            }
            catch (Exception ex)
            {
                channel.Writer.TryComplete(ex);
            }
        }, CancellationToken.None);

        try
        {
            await foreach (var item in channel.Reader.ReadAllAsync(cancellationToken).ConfigureAwait(false))
            {
                yield return item;
            }
        }
        finally
        {
            stop.Cancel();
            await producer.ConfigureAwait(false);
        }
    }
}

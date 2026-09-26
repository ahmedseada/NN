namespace NeuralSharp.Layers;

/// <summary>
/// RMS normalization over the last dimension: x / sqrt(mean(x²) + eps) · (gain + offset). The offset is 0 for most models
/// (1 for models that store the gain as a difference from 1).
/// </summary>
public sealed class RMSNorm : Module
{
    /// <summary>Creates the layer with a gain of ones.</summary>
    public RMSNorm(int features, float epsilon = 1e-6f, float offset = 0f, Device? device = null)
        : this(Tensor.Persistent(Enumerable.Repeat(1f - offset, features).ToArray(), [features], device ?? Device.Default, requiresGrad: true), epsilon, offset)
    {
    }

    private RMSNorm(Tensor gain, float epsilon, float offset)
    {
        Gain = gain;
        Epsilon = epsilon;
        Offset = offset;
    }

    /// <summary>A layer around an existing gain [features]; the layer takes ownership.</summary>
    public static RMSNorm FromWeights(Tensor gain, float epsilon, float offset = 0f) => new(gain, epsilon, offset);

    /// <summary>Size of the normalized dimension.</summary>
    public int Features => Gain.Size;

    /// <summary>Added to the variance before the square root.</summary>
    public float Epsilon { get; }

    /// <summary>Added to the gain (the effective scale is gain + offset).</summary>
    public float Offset { get; }

    /// <summary>The learned scale, [features].</summary>
    public Tensor Gain { get; private set; }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input)
    {
        if (input.Shape[^1] != Features)
        {
            throw new ArgumentException($"RMSNorm({Features}) expects [..., {Features}], got {Tensor.FormatShape(input.Shape)}.");
        }

        if (!Autograd.IsEnabled || !input.RequiresGrad && !Gain.RequiresGrad)
        {
            return input.RmsNormAffine(Gain, Epsilon, Offset);                // one kernel when nothing needs gradients
        }

        var scale = Offset == 0f ? Gain : Gain + Offset;
        return input.RmsNormalize(Epsilon).GroupAffine(scale, null, Features, 1);
    }

    /// <inheritdoc />
    public override IEnumerable<Tensor> Parameters() => [Gain];

    /// <inheritdoc />
    protected internal override void MoveTo(Device device) => Gain = MoveTensor(Gain, device);

    /// <inheritdoc />
    public override string ToString() => $"RMSNorm({Features})";
}

/// <summary>How rotary position embeddings stretch their frequencies for contexts longer than the model was trained on.</summary>
/// <param name="Type">"linear" (positions divided by <paramref name="Factor"/>) or "llama3" (low frequencies stretched, high kept).</param>
/// <param name="Factor">The stretch factor.</param>
/// <param name="LowFrequencyFactor">llama3: wavelengths longer than original / this are fully stretched.</param>
/// <param name="HighFrequencyFactor">llama3: wavelengths shorter than original / this are kept.</param>
/// <param name="OriginalMaxPositions">llama3: the training context length.</param>
public sealed record RopeScaling(string Type, float Factor, float LowFrequencyFactor = 1f, float HighFrequencyFactor = 4f, int OriginalMaxPositions = 8192);

/// <summary>Rotary position embedding settings.</summary>
/// <param name="Theta">The base of the frequencies (10000 in the original paper; larger for long-context models).</param>
/// <param name="RotaryDim">Dimensions of each head that rotate (all when null).</param>
/// <param name="Interleaved">Pairs are (2i, 2i+1) instead of (i, i + rotaryDim / 2).</param>
/// <param name="Scaling">Frequency scaling for long contexts, or null.</param>
public sealed record RopeSettings(float Theta, int? RotaryDim = null, bool Interleaved = false, RopeScaling? Scaling = null)
{
    /// <summary>The rotation frequency of each pair (after scaling).</summary>
    public double[] Frequencies(int headDim)
    {
        int rotary = RotaryDim ?? headDim, half = rotary / 2;
        var frequencies = new double[half];
        for (int i = 0; i < half; i++)
        {
            frequencies[i] = 1.0 / Math.Pow(Theta, 2.0 * i / rotary);
        }

        switch (Scaling?.Type)
        {
            case null:
                break;
            case "linear":
                for (int i = 0; i < half; i++)
                {
                    frequencies[i] /= Scaling.Factor;
                }

                break;
            case "llama3":
                double lowWavelength = Scaling.OriginalMaxPositions / Scaling.LowFrequencyFactor;
                double highWavelength = Scaling.OriginalMaxPositions / Scaling.HighFrequencyFactor;
                for (int i = 0; i < half; i++)
                {
                    double wavelength = 2 * Math.PI / frequencies[i];
                    if (wavelength > lowWavelength)
                    {
                        frequencies[i] /= Scaling.Factor;
                    }
                    else if (wavelength >= highWavelength)
                    {
                        double smooth = (Scaling.OriginalMaxPositions / wavelength - Scaling.LowFrequencyFactor)
                            / (Scaling.HighFrequencyFactor - Scaling.LowFrequencyFactor);
                        frequencies[i] = (1 - smooth) * frequencies[i] / Scaling.Factor + smooth * frequencies[i];
                    }
                }

                break;
            default:
                throw new NotSupportedException($"RoPE scaling '{Scaling.Type}' is not supported (linear, llama3).");
        }

        return frequencies;
    }
}

/// <summary>
/// Causal self-attention for decoder-only language models: separate query, key and value projections (named "q", "k",
/// "v"; output "o"), grouped-query attention (fewer key/value heads, each shared by a group of query heads),
/// a head size independent of the model width, optional biases, optional RMS normalization of each head's queries and
/// keys, and rotary position embeddings. Works with the KV cache (float32 or int8) for incremental decoding.
/// </summary>
public sealed class CausalSelfAttention : Module, ICachedModule
{
    private Tensor _cos = null!;
    private Tensor _sin = null!;
    private readonly Dictionary<int, Tensor> _masks = [];
    private Tensor? _iota;

    /// <summary>Creates the layer with random projections.</summary>
    /// <param name="dim">Model width.</param>
    /// <param name="heads">Query heads.</param>
    /// <param name="kvHeads">Key/value heads (a divisor of <paramref name="heads"/>; equal to it for standard attention).</param>
    /// <param name="headDim">Size of each head.</param>
    /// <param name="rope">Rotary embedding settings, or null for none.</param>
    /// <param name="maxPositions">Longest sequence (the size of the rotary tables).</param>
    /// <param name="qkvBias">Biases on the query, key and value projections.</param>
    /// <param name="outputBias">A bias on the output projection.</param>
    /// <param name="qkNormEpsilon">RMS-normalize each head's queries and keys (with this epsilon), or null.</param>
    /// <param name="device">Device.</param>
    /// <param name="random">Initialization.</param>
    public CausalSelfAttention(int dim, int heads, int kvHeads, int headDim, RopeSettings? rope, int maxPositions, bool qkvBias = false,
        bool outputBias = false, float? qkNormEpsilon = null, Device? device = null, Random? random = null)
        : this(new Linear(dim, heads * headDim, qkvBias, device, random), new Linear(dim, kvHeads * headDim, qkvBias, device, random),
            new Linear(dim, kvHeads * headDim, qkvBias, device, random), new Linear(heads * headDim, dim, outputBias, device, random),
            qkNormEpsilon is { } eps ? new RMSNorm(headDim, eps, device: device) : null,
            qkNormEpsilon is { } eps2 ? new RMSNorm(headDim, eps2, device: device) : null,
            heads, kvHeads, headDim, rope, maxPositions)
    {
    }

    /// <summary>Creates the layer from existing projections (for example loaded weights); the layer takes ownership.</summary>
    public CausalSelfAttention(Linear query, Linear key, Linear value, Linear output, RMSNorm? queryNorm, RMSNorm? keyNorm,
        int heads, int kvHeads, int headDim, RopeSettings? rope, int maxPositions)
    {
        if (heads % kvHeads != 0)
        {
            throw new ArgumentException($"Query heads ({heads}) must be a multiple of key/value heads ({kvHeads}).");
        }

        if (query.OutFeatures != heads * headDim || key.OutFeatures != kvHeads * headDim || value.OutFeatures != kvHeads * headDim
            || output.InFeatures != heads * headDim)
        {
            throw new ArgumentException("The projection sizes do not match heads × head size.");
        }

        (Query, Key, Value, Output, QueryNorm, KeyNorm) = (query, key, value, output, queryNorm, keyNorm);
        query.Name = "q";
        key.Name = "k";
        value.Name = "v";
        output.Name = "o";
        if (queryNorm is not null)
        {
            queryNorm.Name = "q_norm";
        }

        if (keyNorm is not null)
        {
            keyNorm.Name = "k_norm";
        }

        Heads = heads;
        KvHeads = kvHeads;
        HeadDim = headDim;
        Rope = rope;
        MaxPositions = maxPositions;
        if (rope is not null)
        {
            var frequencies = rope.Frequencies(headDim);
            int half = frequencies.Length;
            var cos = new float[maxPositions * half];
            var sin = new float[maxPositions * half];
            for (int p = 0; p < maxPositions; p++)
            {
                for (int i = 0; i < half; i++)
                {
                    double angle = p * frequencies[i];
                    cos[p * half + i] = (float)Math.Cos(angle);
                    sin[p * half + i] = (float)Math.Sin(angle);
                }
            }

            var device = query.Device;
            _cos = Tensor.Persistent(cos, [maxPositions, half], device, requiresGrad: false);
            _sin = Tensor.Persistent(sin, [maxPositions, half], device, requiresGrad: false);
        }
    }

    /// <summary>Query heads.</summary>
    public int Heads { get; }

    /// <summary>Key/value heads.</summary>
    public int KvHeads { get; }

    /// <summary>Size of each head.</summary>
    public int HeadDim { get; }

    /// <summary>Rotary embedding settings, or null.</summary>
    public RopeSettings? Rope { get; }

    /// <summary>Longest supported sequence.</summary>
    public int MaxPositions { get; }

    /// <summary>Query projection ("q").</summary>
    public Linear Query { get; }

    /// <summary>Key projection ("k").</summary>
    public Linear Key { get; }

    /// <summary>Value projection ("v").</summary>
    public Linear Value { get; }

    /// <summary>Output projection ("o").</summary>
    public Linear Output { get; }

    /// <summary>RMS normalization of each query head, or null.</summary>
    public RMSNorm? QueryNorm { get; }

    /// <summary>RMS normalization of each key head, or null.</summary>
    public RMSNorm? KeyNorm { get; }

    private int Group => Heads / KvHeads;

    /// <inheritdoc />
    public override IEnumerable<Module> Children()
    {
        yield return Query;
        yield return Key;
        yield return Value;
        yield return Output;
        if (QueryNorm is not null)
        {
            yield return QueryNorm;
        }

        if (KeyNorm is not null)
        {
            yield return KeyNorm;
        }
    }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input)
    {
        int n = input.Shape[0], t = input.Shape[1];
        if (t > MaxPositions)
        {
            throw new ArgumentException($"{t} positions exceed the layer's maximum of {MaxPositions}.");
        }

        var positions = Positions(t);
        var (q, k, v) = Project(input, positions);
        float scale = 1f / MathF.Sqrt(HeadDim);
        if (HeadDim <= Backends.Cuda.PtxKernels.FlashMaxDim)
        {
            // Tiled attention, forward and backward: no [t, t] weights stored (positions[0] = 0 is the causal offset).
            return Merge(Tensor.CausalAttention(q, k, v, positions, t, scale), n, t);
        }

        var raw = q.MatMul(k, transposeB: true);                                  // [n·kv, group·t, t]
        Tensor weights;
        if (!Autograd.IsEnabled)
        {
            weights = raw.ScaleMaskSoftmax(scale, CausalMask(t, 1));             // mask row = query row % t
        }
        else
        {
            weights = (raw * scale + CausalMask(t, Group)).Softmax();            // the mask repeated for each group
        }

        return Merge(weights.MatMul(v), n, t);
    }

    /// <inheritdoc />
    public Tensor ForwardCached(Tensor input, DecodingContext context)
    {
        int n = input.Shape[0], t = input.Shape[1];
        var positions = context.Positions ?? throw new InvalidOperationException("Call DecodingContext.BeginStep first.");
        var cache = context.CacheFor(this, n * KvHeads, HeadDim);
        var (q, k, v) = Project(input, positions);
        float scale = 1f / MathF.Sqrt(HeadDim);
        Tensor context8;
        if (cache.Format == KeyValueFormat.Int8)
        {
            Tensor.WriteKeyValuesInt8(k, cache.Keys, cache.KeyScales!, context.Position);
            Tensor.WriteKeyValuesInt8(v, cache.Values, cache.ValueScales!, context.Position);
            var weights = Tensor.AttentionScoresInt8(q, cache).ScaleMaskSoftmax(scale, context.Mask);
            context8 = Tensor.AttentionContextInt8(weights, cache);
        }
        else
        {
            Tensor.WriteKeyValues(k, cache.Keys, context.Position);
            Tensor.WriteKeyValues(v, cache.Values, context.Position);
            if (t >= 8 && HeadDim <= Backends.Cuda.PtxKernels.FlashMaxDim)
            {
                context8 = Tensor.AttentionTiled(q, cache.Keys, cache.Values, context.Position, t, scale);   // a prompt: tiled
            }
            else if (HeadDim <= Backends.Cuda.PtxKernels.DecodeMaxDim)
            {
                context8 = Tensor.AttentionDecode(q, cache, context.Position, t, scale);           // only the filled positions
            }
            else
            {
                var weights = q.MatMul(cache.Keys, transposeB: true).ScaleMaskSoftmax(scale, context.Mask);   // [n·kv, group·t, capacity]
                context8 = weights.MatMul(cache.Values);
            }
        }

        return Merge(context8, n, t);
    }

    // Projections → q [n·kv, group·t, d] (the query heads sharing a key/value head are stacked), k and v [n·kv, t, d].
    private (Tensor Q, Tensor K, Tensor V) Project(Tensor input, Tensor positions)
    {
        int n = input.Shape[0], t = input.Shape[1], d = HeadDim;
        var projected = Linear.ForwardMany(input, Query, Key, Value);
        var q = projected[0].Reshape(n, t, Heads, d);
        var k = projected[1].Reshape(n, t, KvHeads, d);
        var v = projected[2].Reshape(n, t, KvHeads, d);
        if (QueryNorm is not null)
        {
            q = QueryNorm.Forward(q);
        }

        if (KeyNorm is not null)
        {
            k = KeyNorm.Forward(k);
        }

        if (Rope is not null)
        {
            int half = _cos.Shape[1];
            q = q.Rope(_cos, _sin, positions, half, Rope.Interleaved);
            k = k.Rope(_cos, _sin, positions, half, Rope.Interleaved);
        }

        var queries = q.Reshape(n, t, KvHeads, Group, d).Permute(0, 2, 3, 1, 4).Reshape(n * KvHeads, Group * t, d);
        var keys = k.Permute(0, 2, 1, 3).Reshape(n * KvHeads, t, d);
        var values = v.Permute(0, 2, 1, 3).Reshape(n * KvHeads, t, d);
        return (queries, keys, values);
    }

    // [n·kv, group·t, d] → [n, t, heads·d] → output projection (query head h = kv · group + g, as the heads were split).
    private Tensor Merge(Tensor context, int n, int t) =>
        Output.Forward(context.Reshape(n, KvHeads, Group, t, HeadDim).Permute(0, 3, 1, 2, 4).Reshape(n, t, Heads * HeadDim));

    private Tensor Positions(int t)
    {
        if (_iota is null || _iota.Size < t || _iota.Device != Query.Device)
        {
            _iota?.Dispose();
            _iota = Tensor.Persistent([.. Enumerable.Range(0, MaxPositions).Select(i => (float)i)], [MaxPositions], Query.Device, requiresGrad: false);
        }

        return _iota;
    }

    // A [repeats·t, t] causal mask (0 on and below the diagonal, -1e9 above), cached per size.
    private Tensor CausalMask(int t, int repeats)
    {
        int key = t * 1024 + repeats;
        if (_masks.TryGetValue(key, out var mask) && mask.Device == Query.Device)
        {
            return mask;
        }

        var values = new float[repeats * t * t];
        for (int r = 0; r < repeats; r++)
        {
            for (int i = 0; i < t; i++)
            {
                for (int j = i + 1; j < t; j++)
                {
                    values[(r * t + i) * t + j] = -1e9f;
                }
            }
        }

        mask?.Dispose();
        _masks[key] = mask = Tensor.Persistent(values, [repeats * t, t], Query.Device, requiresGrad: false);
        return mask;
    }

    /// <inheritdoc />
    protected internal override void MoveTo(Device device)
    {
        base.MoveTo(device);
        if (Rope is not null)
        {
            _cos = MoveTensor(_cos, device);
            _sin = MoveTensor(_sin, device);
        }
    }

    /// <inheritdoc />
    public override void Dispose()
    {
        if (Rope is not null)
        {
            _cos.Dispose();
            _sin.Dispose();
        }

        foreach (var mask in _masks.Values)
        {
            mask.Dispose();
        }

        _iota?.Dispose();
        base.Dispose();
    }

    /// <inheritdoc />
    public override string ToString() =>
        $"CausalSelfAttention({Heads} heads{(KvHeads != Heads ? $", {KvHeads} kv heads" : "")}, head size {HeadDim}{(Rope is null ? "" : ", rope")}{(QueryNorm is null ? "" : ", qk-norm")})";
}

/// <summary>The activation of a <see cref="FeedForward"/> block.</summary>
public enum FeedForwardActivation
{
    /// <summary>x · sigmoid(x) (also called swish; with a gate: SwiGLU).</summary>
    Silu,

    /// <summary>GELU, tanh approximation (with a gate: GeGLU).</summary>
    Gelu,

    /// <summary>max(0, x).</summary>
    Relu,
}

/// <summary>
/// A transformer feed-forward block: gated, down(act(gate(x)) · up(x)) (SwiGLU / GeGLU), or plain, down(act(up(x))).
/// Projections are named "gate", "up" and "down".
/// </summary>
public sealed class FeedForward : Module
{
    /// <summary>Creates the block with random projections.</summary>
    public FeedForward(int dim, int hidden, bool gated, FeedForwardActivation activation, bool bias = false, Device? device = null, Random? random = null)
        : this(gated ? new Linear(dim, hidden, bias, device, random) : null, new Linear(dim, hidden, bias, device, random),
            new Linear(hidden, dim, bias, device, random), activation)
    {
    }

    /// <summary>Creates the block from existing projections (gate null for a plain block); the block takes ownership.</summary>
    public FeedForward(Linear? gate, Linear up, Linear down, FeedForwardActivation activation)
    {
        (Gate, Up, Down, Activation) = (gate, up, down, activation);
        if (gate is not null)
        {
            gate.Name = "gate";
        }

        up.Name = "up";
        down.Name = "down";
    }

    /// <summary>The gate projection ("gate"), or null for a plain block.</summary>
    public Linear? Gate { get; }

    /// <summary>The up projection ("up").</summary>
    public Linear Up { get; }

    /// <summary>The down projection ("down").</summary>
    public Linear Down { get; }

    /// <summary>The activation.</summary>
    public FeedForwardActivation Activation { get; }

    /// <inheritdoc />
    public override IEnumerable<Module> Children() => Gate is null ? [Up, Down] : [Gate, Up, Down];

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input)
    {
        Tensor hidden;
        if (Gate is null)
        {
            hidden = Activate(Up.Forward(input));
        }
        else
        {
            var projected = Linear.ForwardMany(input, Gate, Up);
            hidden = Tensor.GatedActivation(projected[0], projected[1], (int)Activation);    // act(gate) · up in one kernel
        }
        return Down.Forward(hidden);
    }

    private Tensor Activate(Tensor x) => Activation switch
    {
        FeedForwardActivation.Silu => x * x.Sigmoid(),
        FeedForwardActivation.Gelu => x.Gelu(),
        _ => x.Relu(),
    };

    /// <inheritdoc />
    public override string ToString() => $"FeedForward({Up.InFeatures} -> {Up.OutFeatures}, {(Gate is null ? "" : "gated ")}{Activation})";
}

/// <summary>
/// One decoder layer. Sequential (the usual): x + attention(norm1(x)), then + feedForward(norm2(·)); optionally with
/// normalizations after each block too. Parallel: x + attention(norm1(x)) + feedForward(norm1(x)).
/// </summary>
public sealed class DecoderBlock : Module, ICachedModule
{
    /// <summary>Creates the block from its parts; the block takes ownership.</summary>
    /// <param name="attentionNorm">Normalization before attention ("attn_norm").</param>
    /// <param name="attention">The attention layer.</param>
    /// <param name="feedForwardNorm">Normalization before the feed-forward block ("mlp_norm"); null for a parallel block.</param>
    /// <param name="feedForward">The feed-forward block.</param>
    /// <param name="postAttentionNorm">Normalization of the attention output before the residual addition, or null.</param>
    /// <param name="postFeedForwardNorm">Normalization of the feed-forward output before the residual addition, or null.</param>
    public DecoderBlock(Module attentionNorm, CausalSelfAttention attention, Module? feedForwardNorm, FeedForward feedForward,
        Module? postAttentionNorm = null, Module? postFeedForwardNorm = null)
    {
        (AttentionNorm, Attention, FeedForwardNorm, FeedForward, PostAttentionNorm, PostFeedForwardNorm) =
            (attentionNorm, attention, feedForwardNorm, feedForward, postAttentionNorm, postFeedForwardNorm);
        attentionNorm.Name = "attn_norm";
        attention.Name = "attn";
        feedForward.Name = "mlp";
        if (feedForwardNorm is not null)
        {
            feedForwardNorm.Name = "mlp_norm";
        }

        if (postAttentionNorm is not null)
        {
            postAttentionNorm.Name = "post_attn_norm";
        }

        if (postFeedForwardNorm is not null)
        {
            postFeedForwardNorm.Name = "post_mlp_norm";
        }
    }

    /// <summary>Normalization before attention.</summary>
    public Module AttentionNorm { get; }

    /// <summary>The attention layer.</summary>
    public CausalSelfAttention Attention { get; }

    /// <summary>Normalization before the feed-forward block, or null for a parallel block.</summary>
    public Module? FeedForwardNorm { get; }

    /// <summary>The feed-forward block.</summary>
    public FeedForward FeedForward { get; }

    /// <summary>Normalization after attention, or null.</summary>
    public Module? PostAttentionNorm { get; }

    /// <summary>Normalization after the feed-forward block, or null.</summary>
    public Module? PostFeedForwardNorm { get; }

    /// <summary>True when attention and the feed-forward block read the same normalized input and are added together.</summary>
    public bool Parallel => FeedForwardNorm is null;

    /// <inheritdoc />
    public override IEnumerable<Module> Children() =>
        new[] { AttentionNorm, Attention, FeedForwardNorm, FeedForward, PostAttentionNorm, PostFeedForwardNorm }.OfType<Module>();

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => Run(input, x => Attention.Forward(x));

    /// <inheritdoc />
    public Tensor ForwardCached(Tensor input, DecodingContext context) => Run(input, x => Attention.ForwardCached(x, context));

    private Tensor Run(Tensor input, Func<Tensor, Tensor> attend)
    {
        var normalized = AttentionNorm.Forward(input);
        var attended = attend(normalized);
        if (PostAttentionNorm is not null)
        {
            attended = PostAttentionNorm.Forward(attended);
        }

        if (Parallel)
        {
            return input + attended + FeedForward.Forward(normalized);
        }

        var x = input + attended;
        var fed = FeedForward.Forward(FeedForwardNorm!.Forward(x));
        if (PostFeedForwardNorm is not null)
        {
            fed = PostFeedForwardNorm.Forward(fed);
        }

        return x + fed;
    }

    /// <inheritdoc />
    public override string ToString() => $"DecoderBlock({(Parallel ? "parallel" : "sequential")})";
}

/// <summary>Multiplies its input by a constant (for example the √dim embedding scale of some models).</summary>
public sealed class Scale(float factor) : Module
{
    /// <summary>The factor.</summary>
    public float Factor { get; } = factor;

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => input * Factor;

    /// <inheritdoc />
    public override string ToString() => $"Scale({Factor})";
}

namespace NeuralSharp.Layers;

/// <summary>
/// Fully connected layer: y = x · W + b, mapping [..., inFeatures] to [..., outFeatures]
/// (any leading dimensions, e.g. [batch, time, features] for sequences).
/// Weights start Xavier/Glorot-uniform and the bias starts at zero.
/// </summary>
public sealed class Linear : Module
{
    /// <summary>Creates the layer on <paramref name="device"/> (default: <see cref="Device.Default"/>).</summary>
    /// <param name="inFeatures">Size of each input row.</param>
    /// <param name="outFeatures">Size of each output row.</param>
    /// <param name="bias">Whether to learn an additive bias.</param>
    /// <param name="device">Where the parameters live.</param>
    /// <param name="random">Source of the initial weights; pass a seeded <see cref="System.Random"/> for reproducible runs.</param>
    public Linear(int inFeatures, int outFeatures, bool bias = true, Device? device = null, Random? random = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inFeatures);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outFeatures);
        InFeatures = inFeatures;
        OutFeatures = outFeatures;
        device ??= Device.Default;
        random ??= Random.Shared;

        float limit = MathF.Sqrt(6f / (inFeatures + outFeatures));
        _weight = CreateParameter(UniformValues(inFeatures * outFeatures, limit, random), [inFeatures, outFeatures], device);
        Bias = bias ? CreateParameter(new float[outFeatures], [outFeatures], device) : null;
    }

    /// <summary>Number of input features.</summary>
    public int InFeatures { get; }

    /// <summary>Number of output features.</summary>
    public int OutFeatures { get; }

    /// <summary>The [inFeatures, outFeatures] weight matrix.</summary>
    public Tensor Weight => _weight ?? throw new InvalidOperationException(
        $"{this} holds int8 weights (see Int8); call DequantizeInt8() on the model to get float weights back.");

    /// <summary>The int8 weights when the layer was quantized with <see cref="ModuleExtensions.QuantizeInt8"/>, else null.</summary>
    public Int8Weight? Int8 { get; private set; }

    private Tensor? _weight;

    /// <summary>The [outFeatures] bias, or null when created with <c>bias: false</c>.</summary>
    public Tensor? Bias { get; private set; }

    /// <summary>
    /// A low-rank adapter (LoRA) added to this layer by <see cref="ModuleExtensions.AddLora"/>, or null. When present the
    /// output is <c>x·W + b + (x·A·B)·scale</c>; its A and B come after W and b in <see cref="Parameters"/>.
    /// </summary>
    public LoraAdapter? Adapter { get; internal set; }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input)
    {
        var product = ProjectWithoutBias(input);
        return Bias is null ? product : product + Bias;
    }

    /// <summary>x·W, plus the adapter's low-rank term when an adapter is attached.</summary>
    internal Tensor ProjectWithoutBias(Tensor input)
    {
        var product = Int8 is { } q ? input.MatMulInt8(q) : input.MatMul(Weight);
        return Adapter is { } a ? product + input.MatMul(a.A).MatMul(a.B) * a.Scale : product;
    }

    /// <inheritdoc />
    public override IEnumerable<Tensor> Parameters()
    {
        IEnumerable<Tensor> own = (_weight, Bias) switch
        {
            (null, null) => [],
            (null, { } b) => [b],
            ({ } w, null) => [w],
            ({ } w, { } b) => [w, b],
        };
        return Adapter is { } a ? own.Concat([a.A, a.B]) : own;
    }

    /// <summary>Folds the adapter into the weight (<c>W += A·B·scale</c>) and removes it; the outputs stay the same.</summary>
    /// <inheritdoc />
    public override IEnumerable<Tensor> Buffers() => Int8 is { } q ? [q.Packed, q.Scales] : [];

    // The float weight values (dequantized when the layer holds int8 weights).
    internal float[] WeightValues()
    {
        if (Int8 is null)
        {
            return Weight.ToArray();
        }

        using var w = Int8.Dequantize();
        return w.ToArray();
    }

    internal Device Device => (_weight ?? Int8!.Packed).Device;

    internal void QuantizeInt8()
    {
        if (Int8 is not null)
        {
            return;
        }

        Int8 = Int8Weight.Quantize(Weight);
        _weight!.Dispose();
        _weight = null;
    }

    internal void DequantizeInt8(bool trainable)
    {
        if (Int8 is not { } q)
        {
            return;
        }

        using (var w = q.Dequantize())
        {
            _weight = Tensor.Persistent(w.ToArray(), [InFeatures, OutFeatures], w.Device, trainable);
        }

        q.Dispose();
        Int8 = null;
    }

    internal void MergeAdapter()
    {
        if (Adapter is not { } a)
        {
            return;
        }

        if (Int8 is not null)
        {
            throw new InvalidOperationException($"{this}: merging a LoRA adapter into int8 weights would lose precision; call DequantizeInt8() first, or keep the adapter.");
        }

        using (Autograd.NoGrad())
        using (var scope = new TensorScope())
        {
            var merged = Weight + a.A.MatMul(a.B) * a.Scale;
            Weight.Load(merged.ToArray());
        }

        Adapter = null;
        a.A.Dispose();
        a.B.Dispose();
    }

    /// <inheritdoc />
    protected internal override void MoveTo(Device device)
    {
        _weight = _weight is null ? null : MoveTensor(_weight, device);
        Int8?.MoveTo(device, MoveTensor);
        Bias = Bias is null ? null : MoveTensor(Bias, device);
        if (Adapter is { } a)
        {
            Adapter = a with { A = MoveTensor(a.A, device), B = MoveTensor(a.B, device) };
        }
    }

    /// <inheritdoc />
    public override string ToString() =>
        $"Linear({InFeatures} -> {OutFeatures}{(Bias is null ? ", no bias" : "")}{(Int8 is null ? "" : ", int8")}{(Adapter is { } a ? $", LoRA rank {a.Rank}" : "")})";
}

/// <summary>
/// A LoRA adapter: two small trainable matrices A [in, rank] and B [rank, out] whose product, times
/// <see cref="Scale"/> = alpha / rank, is added to a <see cref="Linear"/> layer's weight. B starts at zero, so adding
/// an adapter does not change the model's outputs until it is trained.
/// </summary>
/// <param name="A">[inFeatures, rank], small random values.</param>
/// <param name="B">[rank, outFeatures], zeros at creation.</param>
/// <param name="Rank">The rank r.</param>
/// <param name="Scale">alpha / r.</param>
public sealed record LoraAdapter(Tensor A, Tensor B, int Rank, float Scale);

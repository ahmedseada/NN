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
        Weight = CreateParameter(UniformValues(inFeatures * outFeatures, limit, random), [inFeatures, outFeatures], device);
        Bias = bias ? CreateParameter(new float[outFeatures], [outFeatures], device) : null;
    }

    /// <summary>Number of input features.</summary>
    public int InFeatures { get; }

    /// <summary>Number of output features.</summary>
    public int OutFeatures { get; }

    /// <summary>The [inFeatures, outFeatures] weight matrix.</summary>
    public Tensor Weight { get; private set; }

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
        var product = input.MatMul(Weight);
        return Adapter is { } a ? product + input.MatMul(a.A).MatMul(a.B) * a.Scale : product;
    }

    /// <inheritdoc />
    public override IEnumerable<Tensor> Parameters()
    {
        IEnumerable<Tensor> own = Bias is null ? [Weight] : [Weight, Bias];
        return Adapter is { } a ? own.Concat([a.A, a.B]) : own;
    }

    /// <summary>Folds the adapter into the weight (<c>W += A·B·scale</c>) and removes it; the outputs stay the same.</summary>
    internal void MergeAdapter()
    {
        if (Adapter is not { } a)
        {
            return;
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
        Weight = MoveTensor(Weight, device);
        Bias = Bias is null ? null : MoveTensor(Bias, device);
        if (Adapter is { } a)
        {
            Adapter = a with { A = MoveTensor(a.A, device), B = MoveTensor(a.B, device) };
        }
    }

    /// <inheritdoc />
    public override string ToString() =>
        $"Linear({InFeatures} -> {OutFeatures}{(Bias is null ? ", no bias" : "")}{(Adapter is { } a ? $", LoRA rank {a.Rank}" : "")})";
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

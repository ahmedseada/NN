namespace NeuralSharp.Layers;

/// <summary>
/// Fully connected layer: y = x · W + b, mapping [batch, inFeatures] to [batch, outFeatures].
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
        var w = new float[inFeatures * outFeatures];
        for (int i = 0; i < w.Length; i++)
        {
            w[i] = (random.NextSingle() * 2f - 1f) * limit;
        }

        Weight = Tensor.Persistent(w, [inFeatures, outFeatures], device, requiresGrad: true);
        Bias = bias ? Tensor.Persistent(new float[outFeatures], [outFeatures], device, requiresGrad: true) : null;
    }

    /// <summary>Number of input features.</summary>
    public int InFeatures { get; }

    /// <summary>Number of output features.</summary>
    public int OutFeatures { get; }

    /// <summary>The [inFeatures, outFeatures] weight matrix.</summary>
    public Tensor Weight { get; private set; }

    /// <summary>The [outFeatures] bias, or null when created with <c>bias: false</c>.</summary>
    public Tensor? Bias { get; private set; }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input)
    {
        var product = input.MatMul(Weight);
        return Bias is null ? product : product + Bias;
    }

    /// <inheritdoc />
    public override IEnumerable<Tensor> Parameters() => Bias is null ? [Weight] : [Weight, Bias];

    /// <inheritdoc />
    protected internal override void MoveTo(Device device)
    {
        Weight = Move(Weight, device);
        Bias = Bias is null ? null : Move(Bias, device);
    }

    private static Tensor Move(Tensor t, Device device)
    {
        if (t.Device == device)
        {
            return t;
        }

        var moved = Tensor.Persistent(t.ToArray(), t.Shape, device, requiresGrad: true);
        t.Dispose();
        return moved;
    }

    /// <inheritdoc />
    public override string ToString() => $"Linear({InFeatures} -> {OutFeatures}{(Bias is null ? ", no bias" : "")})";
}

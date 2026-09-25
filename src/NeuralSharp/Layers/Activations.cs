namespace NeuralSharp.Layers;

/// <summary>Logistic sigmoid activation; squashes values into (0, 1).</summary>
public sealed class Sigmoid : Module
{
    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => input.Sigmoid();

    /// <inheritdoc />
    public override string ToString() => "Sigmoid";
}

/// <summary>Hyperbolic tangent activation; squashes values into (-1, 1).</summary>
public sealed class Tanh : Module
{
    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => input.Tanh();

    /// <inheritdoc />
    public override string ToString() => "Tanh";
}

/// <summary>Rectified linear unit activation: max(x, 0).</summary>
public sealed class ReLU : Module
{
    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => input.Relu();

    /// <inheritdoc />
    public override string ToString() => "ReLU";
}

/// <summary>
/// Randomly zeroes a fraction <see cref="Probability"/> of the inputs during training (and scales the
/// rest by 1 / (1 - p)) to reduce overfitting. In evaluation mode it passes inputs through unchanged.
/// </summary>
public sealed class Dropout : Module
{
    private readonly Random _random;

    /// <summary>Creates the layer.</summary>
    /// <param name="probability">Fraction of inputs to drop, in [0, 1).</param>
    /// <param name="random">Seed source for the masks; pass a seeded <see cref="System.Random"/> for reproducible runs.</param>
    public Dropout(float probability = 0.5f, Random? random = null)
    {
        if (probability is < 0f or >= 1f)
        {
            throw new ArgumentOutOfRangeException(nameof(probability), probability, "Dropout probability must be in [0, 1).");
        }

        Probability = probability;
        _random = random ?? new Random();
    }

    /// <summary>Fraction of inputs dropped during training.</summary>
    public float Probability { get; }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) =>
        IsTraining && Probability > 0f ? input.Dropout(Probability, (uint)_random.Next()) : input;

    /// <inheritdoc />
    public override string ToString() => $"Dropout(p={Probability})";
}

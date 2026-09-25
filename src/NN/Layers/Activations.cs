namespace NN.Layers;

/// <summary>Logistic sigmoid activation; squashes values into (0, 1).</summary>
public sealed class Sigmoid : Module
{
    /// <inheritdoc />
    public override Tensor Forward(Tensor input) => input.Sigmoid();

    /// <inheritdoc />
    public override string ToString() => "Sigmoid";
}

/// <summary>Hyperbolic tangent activation; squashes values into (-1, 1).</summary>
public sealed class Tanh : Module
{
    /// <inheritdoc />
    public override Tensor Forward(Tensor input) => input.Tanh();

    /// <inheritdoc />
    public override string ToString() => "Tanh";
}

/// <summary>Rectified linear unit activation: max(x, 0).</summary>
public sealed class ReLU : Module
{
    /// <inheritdoc />
    public override Tensor Forward(Tensor input) => input.Relu();

    /// <inheritdoc />
    public override string ToString() => "ReLU";
}

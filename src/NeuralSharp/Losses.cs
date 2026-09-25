namespace NeuralSharp;

/// <summary>Loss functions. Each returns a scalar tensor you can call <see cref="Tensor.Backward()"/> on.</summary>
public static class Losses
{
    /// <summary>Mean squared error: mean((prediction - target)²).</summary>
    /// <remarks>
    /// The intermediate tensors belong to the autograd graph and must stay alive until
    /// <see cref="Tensor.Backward()"/> runs, so they are left to the caller's <see cref="TensorScope"/>.
    /// </remarks>
    public static Tensor MeanSquaredError(Tensor prediction, Tensor target) => (prediction - target).Square().Mean();

    /// <summary>Mean absolute error: mean(|prediction - target|). Less sensitive to outliers than MSE.</summary>
    public static Tensor MeanAbsoluteError(Tensor prediction, Tensor target) => (prediction - target).Abs().Mean();
}

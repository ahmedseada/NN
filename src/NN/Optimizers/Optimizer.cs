namespace NN.Optimizers;

/// <summary>Updates parameters from their gradients. Call <see cref="ZeroGrad"/>, then <c>loss.Backward()</c>, then <see cref="Step"/>.</summary>
public abstract class Optimizer : IDisposable
{
    /// <summary>Creates an optimizer for the given parameters (usually <c>model.Parameters()</c>).</summary>
    protected Optimizer(IEnumerable<Tensor> parameters, float learningRate)
    {
        Parameters = [.. parameters];
        if (Parameters.Count == 0)
        {
            throw new ArgumentException("The optimizer was given no parameters.", nameof(parameters));
        }

        LearningRate = learningRate;
    }

    /// <summary>The tensors this optimizer updates.</summary>
    public IReadOnlyList<Tensor> Parameters { get; }

    /// <summary>Step size; can be changed between steps (e.g. by a schedule).</summary>
    public float LearningRate { get; set; }

    /// <summary>Resets every parameter's gradient to zero.</summary>
    public void ZeroGrad()
    {
        foreach (var p in Parameters)
        {
            p.ZeroGrad();
        }
    }

    /// <summary>Applies one update using the current gradients.</summary>
    public abstract void Step();

    /// <summary>Allocates a zeroed state buffer shaped like <paramref name="parameter"/>, outside any <see cref="TensorScope"/>.</summary>
    protected static Tensor CreateState(Tensor parameter) => Tensor.Persistent(new float[parameter.Size], parameter.Shape, parameter.Device, requiresGrad: false);

    /// <summary>Releases optimizer state (moment buffers).</summary>
    public virtual void Dispose() => GC.SuppressFinalize(this);
}

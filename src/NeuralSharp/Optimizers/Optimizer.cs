namespace NeuralSharp.Optimizers;

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

    /// <summary>Global L2 norm of all gradients (one device synchronization).</summary>
    public double GradientNorm()
    {
        var withGrad = Parameters.Where(p => p.Grad is not null).ToList();
        if (withGrad.Count == 0)
        {
            return 0;
        }

        using var total = Tensor.Zeros([1], withGrad[0].Device);
        using (Autograd.NoGrad())
        {
            foreach (var p in withGrad)
            {
                using var squared = p.Grad!.Square();
                using var sum = squared.Sum();
                total.Backend.AxpyAt(sum.Storage, total.Storage, 0, 1f);
            }
        }

        return Math.Sqrt(total.Item());
    }

    /// <summary>
    /// Scales all gradients down so their global L2 norm is at most <paramref name="maxNorm"/> (no change when
    /// already smaller). Prevents exploding gradients, especially in recurrent networks. Returns the norm before clipping.
    /// </summary>
    public double ClipGradientNorm(float maxNorm)
    {
        double norm = GradientNorm();
        if (norm > maxNorm && norm > 0)
        {
            float factor = (float)(maxNorm / norm);
            foreach (var p in Parameters)
            {
                if (p.Grad is { } g)
                {
                    g.Backend.Affine(g.Storage, g.Storage, g.Size, factor, 0f);
                }
            }
        }

        return norm;
    }

    /// <summary>Adds L2 weight decay to the gradients: g += decay · p.</summary>
    protected void ApplyCoupledWeightDecay(float decay)
    {
        if (decay == 0f)
        {
            return;
        }

        foreach (var p in Parameters)
        {
            if (p.Grad is { } g)
            {
                p.Backend.Axpy(p.Storage, g.Storage, p.Size, decay);
            }
        }
    }

    /// <summary>Allocates a zeroed state buffer shaped like <paramref name="parameter"/>, outside any <see cref="TensorScope"/>.</summary>
    protected static Tensor CreateState(Tensor parameter) => Tensor.Persistent(new float[parameter.Size], parameter.Shape, parameter.Device, requiresGrad: false);

    /// <summary>Releases optimizer state (moment buffers).</summary>
    public virtual void Dispose() => GC.SuppressFinalize(this);
}

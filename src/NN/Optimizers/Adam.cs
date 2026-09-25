namespace NN.Optimizers;

/// <summary>
/// Adam (Kingma &amp; Ba, 2015): per-parameter adaptive steps from running averages of the
/// gradient and its square, with bias correction. A good default for most networks.
/// </summary>
public sealed class Adam(IEnumerable<Tensor> parameters, float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
    : Optimizer(parameters, learningRate)
{
    private Tensor?[]? _m;
    private Tensor?[]? _v;
    private int _step;

    /// <summary>Decay rate of the first-moment (mean) estimate.</summary>
    public float Beta1 { get; } = beta1;

    /// <summary>Decay rate of the second-moment (uncentered variance) estimate.</summary>
    public float Beta2 { get; } = beta2;

    /// <summary>Term added to the denominator for numerical stability.</summary>
    public float Epsilon { get; } = epsilon;

    /// <inheritdoc />
    public override void Step()
    {
        _step++;
        _m ??= new Tensor?[Parameters.Count];
        _v ??= new Tensor?[Parameters.Count];

        // Fold both bias corrections into the step size: lr * sqrt(1 - β2^t) / (1 - β1^t).
        float correctedLr = (float)(LearningRate * Math.Sqrt(1 - Math.Pow(Beta2, _step)) / (1 - Math.Pow(Beta1, _step)));
        for (int i = 0; i < Parameters.Count; i++)
        {
            var p = Parameters[i];
            if (p.Grad is null)
            {
                continue;
            }

            var m = _m[i] ??= CreateState(p);
            var v = _v[i] ??= CreateState(p);
            p.Backend.AdamStep(p.Storage, p.Grad.Storage, m.Storage, v.Storage, p.Size, correctedLr, Beta1, Beta2, Epsilon);
        }
    }

    /// <inheritdoc />
    public override void Dispose()
    {
        foreach (var t in (_m ?? []).Concat(_v ?? []))
        {
            t?.Dispose();
        }

        base.Dispose();
    }
}

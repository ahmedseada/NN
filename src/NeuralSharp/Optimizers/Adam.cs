namespace NeuralSharp.Optimizers;

/// <summary>
/// Adam (Kingma &amp; Ba, 2015): per-parameter adaptive steps from running averages of the
/// gradient and its square, with bias correction. A good default for most networks.
/// </summary>
/// <remarks>
/// <paramref name="weightDecay"/> adds an L2 penalty to the gradients (classic Adam). For decoupled weight
/// decay, which usually generalizes better, use <see cref="AdamW"/>.
/// </remarks>
public class Adam(IEnumerable<Tensor> parameters, float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f, float weightDecay = 0f)
    : Optimizer(parameters, learningRate)
{
    /// <summary>Weight decay factor λ.</summary>
    public float WeightDecay { get; } = weightDecay;

    /// <summary>True for AdamW-style decay (p -= lr·λ·p before the step) instead of L2 on the gradients.</summary>
    protected virtual bool DecoupledWeightDecay => false;

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
        if (WeightDecay != 0f)
        {
            if (DecoupledWeightDecay)
            {
                foreach (var p in Parameters)
                {
                    if (p.Grad is not null)
                    {
                        p.Backend.Affine(p.Storage, p.Storage, p.Size, 1f - LearningRate * WeightDecay, 0f);
                    }
                }
            }
            else
            {
                ApplyCoupledWeightDecay(WeightDecay);
            }
        }

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

/// <summary>
/// AdamW (Loshchilov &amp; Hutter, 2019): Adam with decoupled weight decay, p -= lr·λ·p each step.
/// The usual choice for transformers and other large models.
/// </summary>
public sealed class AdamW(IEnumerable<Tensor> parameters, float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f, float weightDecay = 0.01f)
    : Adam(parameters, learningRate, beta1, beta2, epsilon, weightDecay)
{
    /// <inheritdoc />
    protected override bool DecoupledWeightDecay => true;
}

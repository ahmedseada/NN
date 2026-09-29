namespace NeuralSharp;

/// <summary>
/// How training keeps activations in the decoder layers. Results that no backward step reads (projections before their
/// bias or rotation, head rearrangements, residual sums, the output projections) are released as soon as their last
/// forward use is done (<see cref="ReleaseUnused"/>, on by default: free memory, no extra work). With
/// <see cref="RecomputeFeedForward"/>, the feed-forward block's activation (act(gate) · up, the largest activation of a
/// block) is released after the forward pass too and recomputed from gate and up when the backward pass needs it: one
/// element-wise kernel per block for about a fifth less activation memory.
/// </summary>
/// <example>
/// <code>
/// using (ActivationMemory.Recompute())
/// {
///     loss = model.Forward(tokens) …;   // the feed-forward activations are recomputed in loss.Backward()
///     loss.Backward();
/// }
/// </code>
/// </example>
public static class ActivationMemory
{
    [ThreadStatic]
    private static int t_recompute;

    /// <summary>Release results no backward step reads during the forward pass (default true; false keeps every result until the step ends).</summary>
    public static bool ReleaseUnused { get; set; } = true;

    /// <summary>Whether the feed-forward activations are recomputed in the backward pass on this thread (see <see cref="Recompute"/>).</summary>
    public static bool RecomputeFeedForward => t_recompute > 0;

    /// <summary>Recomputes the feed-forward activations in the backward pass on this thread until the returned scope is disposed.</summary>
    public static Scope Recompute()
    {
        t_recompute++;
        return new Scope();
    }

    /// <summary>Releases the values of results no backward step reads, while gradients are recorded.</summary>
    internal static void Release(params Tensor?[] tensors)
    {
        if (!ReleaseUnused || !Autograd.IsEnabled)
        {
            return;
        }

        foreach (var tensor in tensors)
        {
            tensor?.DropValue();
        }
    }

    /// <summary>Ends <see cref="Recompute"/>.</summary>
    public readonly struct Scope : IDisposable
    {
        /// <inheritdoc />
        public void Dispose() => t_recompute--;
    }
}

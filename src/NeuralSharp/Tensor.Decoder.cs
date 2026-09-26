using NeuralSharp.Diagnostics;

namespace NeuralSharp;

// Operations of decoder-only language model layers.
public sealed partial class Tensor
{
    /// <summary>x / sqrt(mean(x²) + eps) over the last dimension (RMS normalization without a gain).</summary>
    internal Tensor RmsNormalize(float eps)
    {
        ThrowIfDisposed();
        long start = Telemetry.Start(TelemetryLevel.Operations);
        int cols = _shape[^1], rows = Size / cols;
        var y = Empty(_shape, Device);
        var inv = Empty([rows], Device, track: false);
        Backend.RmsNorm(Storage, y.Storage, inv.Storage, rows, cols, eps);
        if (WillRecord(this))
        {
            var x = this;
            y.Record("rms_norm", g =>
            {
                x.Backend.RmsNormBackward(g.Storage, y.Storage, inv.Storage, x.GradStorage(), rows, cols);
                inv.Dispose();
            }, x);
        }
        else
        {
            inv.Dispose();
        }

        return Traced("rms_norm", y, start);
    }

    /// <summary>
    /// input · weights[j] (+ biases[j]) for every j in one pass over the input (inference: not recorded). The input is
    /// [..., k]; each weight is [k, n_j]; results are [..., n_j].
    /// </summary>
    internal static Tensor[] MatMulMany(Tensor input, IReadOnlyList<Tensor> weights, IReadOnlyList<Tensor?> biases)
    {
        input.ThrowIfDisposed();
        long start = Telemetry.Start(TelemetryLevel.Operations);
        int k = input._shape[^1], m = input.Size / k;
        var outputs = new Tensor[weights.Count];
        var products = new (Backends.Storage, Backends.Storage?, Backends.Storage, int)[weights.Count];
        for (int j = 0; j < weights.Count; j++)
        {
            int n = weights[j]._shape[1];
            outputs[j] = Empty([.. input._shape[..^1], n], input.Device);
            products[j] = (weights[j].Storage, biases[j]?.Storage, outputs[j].Storage, n);
        }

        input.Backend.MatMulMany(input.Storage, m, k, products);
        foreach (var output in outputs)
        {
            Traced("matmul_many", output, start);
        }

        return outputs;
    }

    /// <summary>
    /// The layers' packed products of one input in one device pass (few rows, not recorded), or null when the device has
    /// no single-pass version.
    /// </summary>
    internal static Tensor[]? MatMulPackedMany(Tensor input, int kind, IReadOnlyList<Layers.Linear> layers)
    {
        input.ThrowIfDisposed();
        long start = Telemetry.Start(TelemetryLevel.Operations);
        int k = input._shape[^1], m = input.Size / k;
        var outputs = new Tensor[layers.Count];
        var products = new (Backends.Storage, Backends.Storage?, Backends.Storage?, Backends.Storage, int)[layers.Count];
        for (int j = 0; j < layers.Count; j++)
        {
            var layer = layers[j];
            var (packed, scales) = kind switch
            {
                0 => (layer.Int8!.Packed, layer.Int8.Scales),
                1 => (layer.Int4!.Packed, layer.Int4.Scales),
                _ => (layer.BFloat16!.Packed, (Tensor?)null),
            };
            if (packed.Device != input.Device)
            {
                return null;
            }

            outputs[j] = Empty([.. input._shape[..^1], layer.OutFeatures], input.Device);
            products[j] = (packed.Storage, scales?.Storage, layer.Bias?.Storage, outputs[j].Storage, layer.OutFeatures);
        }

        if (!input.Backend.PackedMatMulMany(kind, input.Storage, m, k, products))
        {
            foreach (var output in outputs)
            {
                output.Dispose();
            }

            return null;
        }

        foreach (var output in outputs)
        {
            Traced("matmul_many_packed", output, start);
        }

        return outputs;
    }

    /// <summary>x / sqrt(mean(x²) + eps) · (gain + offset) over the last dimension in one pass (inference: not recorded).</summary>
    internal Tensor RmsNormAffine(Tensor gain, float eps, float offset)
    {
        ThrowIfDisposed();
        long start = Telemetry.Start(TelemetryLevel.Operations);
        int cols = _shape[^1], rows = Size / cols;
        var y = Empty(_shape, Device);
        Backend.RmsNormAffine(Storage, gain.Storage, y.Storage, rows, cols, eps, offset);
        return Traced("rms_norm_affine", y, start);
    }

    /// <summary>a + b (a residual addition) and its RMS normalization with gain, in one pass (inference: not recorded).</summary>
    internal static (Tensor Sum, Tensor Normalized) AddRmsNormAffine(Tensor a, Tensor b, Tensor gain, float eps, float offset)
    {
        a.ThrowIfDisposed();
        b.ThrowIfDisposed();
        CheckSameDevice(a, b);
        long start = Telemetry.Start(TelemetryLevel.Operations);
        int cols = a._shape[^1], rows = a.Size / cols;
        var sum = Empty(a._shape, a.Device);
        var y = Empty(a._shape, a.Device);
        a.Backend.AddRmsNormAffine(a.Storage, b.Storage, sum.Storage, gain.Storage, y.Storage, rows, cols, eps, offset);
        return (sum, Traced("add_rms_norm", y, start));
    }

    /// <summary>
    /// RMS normalization with gain of each head's vector of this [batch, steps, heads, dim] tensor, then the rotary
    /// embedding (see <see cref="Rope"/>), in one pass (inference: not recorded).
    /// </summary>
    internal Tensor RmsNormRope(Tensor gain, float eps, float offset, Tensor cos, Tensor sin, Tensor positions, int half, bool interleaved)
    {
        ThrowIfDisposed();
        long start = Telemetry.Start(TelemetryLevel.Operations);
        int steps = _shape[1], heads = _shape[2], dim = _shape[3], rows = Size / dim;
        var y = Empty(_shape, Device);
        Backend.RmsNormRope(Storage, gain.Storage, cos.Storage, sin.Storage, positions.Storage, y.Storage, rows, dim, eps, offset, heads, steps,
            half, interleaved);
        return Traced("rms_norm_rope", y, start);
    }

    /// <summary><see cref="RmsNormRope"/> of the queries and the keys (same positions and tables) in one pass where the device can.</summary>
    internal static (Tensor Q, Tensor K) RmsNormRopePair(Tensor q, Tensor gainQ, float epsQ, float offsetQ, Tensor k, Tensor gainK, float epsK,
        float offsetK, Tensor cos, Tensor sin, Tensor positions, int half, bool interleaved)
    {
        q.ThrowIfDisposed();
        k.ThrowIfDisposed();
        long start = Telemetry.Start(TelemetryLevel.Operations);
        int steps = q._shape[1], dim = q._shape[3];
        var yq = Empty(q._shape, q.Device);
        var yk = Empty(k._shape, k.Device);
        q.Backend.RmsNormRopePair(q.Storage, gainQ.Storage, yq.Storage, q.Size / dim, epsQ, offsetQ, q._shape[2],
            k.Storage, gainK.Storage, yk.Storage, k.Size / dim, epsK, offsetK, k._shape[2], cos.Storage, sin.Storage, positions.Storage,
            dim, steps, half, interleaved);
        Traced("rms_norm_rope", yq, start);
        return (yq, Traced("rms_norm_rope", yk, start));
    }

    /// <summary>act(gate) · up element-wise (kind 0 = SiLU, 1 = GELU, 2 = ReLU), with its gradient: one pass either way.</summary>
    internal static Tensor GatedActivation(Tensor gate, Tensor up, int kind)
    {
        gate.ThrowIfDisposed();
        up.ThrowIfDisposed();
        CheckSameDevice(gate, up);
        if (!gate._shape.AsSpan().SequenceEqual(up._shape))
        {
            throw new ArgumentException($"Gate {FormatShape(gate._shape)} and up {FormatShape(up._shape)} differ.");
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty(gate._shape, gate.Device);
        gate.Backend.GatedActivation(gate.Storage, up.Storage, y.Storage, gate.Size, kind);
        if (WillRecord(gate, up))
        {
            y.Record("gated_activation", g =>
            {
                int flags = (gate.RequiresGrad ? 1 : 0) | (up.RequiresGrad ? 2 : 0);
                gate.Backend.GatedActivationBackward(gate.Storage, up.Storage, g.Storage,
                    gate.RequiresGrad ? gate.GradStorage() : g.Storage, up.RequiresGrad ? up.GradStorage() : g.Storage, gate.Size, kind, flags);
            }, gate, up);
        }

        return Traced("gated_activation", y, start);
    }

    /// <summary>
    /// Rotary position embedding of this [batch, steps, heads, dim] tensor: pair p of each head's vector at step t
    /// rotates by the angle with cos/sin[positions[t], p] ([positions, half] tables). Dimensions beyond 2·half pass through.
    /// </summary>
    internal Tensor Rope(Tensor cos, Tensor sin, Tensor positions, int half, bool interleaved)
    {
        ThrowIfDisposed();
        if (Rank != 4)
        {
            throw new ArgumentException($"Rotary embedding expects [batch, steps, heads, dim], got {FormatShape(_shape)}.");
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        int steps = _shape[1], heads = _shape[2], dim = _shape[3], rows = Size / dim;
        var y = Empty(_shape, Device);
        if (2 * half < dim)
        {
            Backend.Copy(Storage, y.Storage, Size);                        // dimensions beyond the rotated pairs pass through
        }

        Backend.Rope(Storage, y.Storage, cos.Storage, sin.Storage, positions.Storage, rows, heads, steps, dim, half, interleaved, 1f);
        if (WillRecord(this))
        {
            var x = this;
            y.Record("rope", g =>
            {
                using var back = Empty(x._shape, x.Device, track: false);
                if (2 * half < dim)
                {
                    x.Backend.Copy(g.Storage, back.Storage, x.Size);
                }

                x.Backend.Rope(g.Storage, back.Storage, cos.Storage, sin.Storage, positions.Storage, rows, heads, steps, dim, half, interleaved, -1f);
                x.Backend.Axpy(back.Storage, x.GradStorage(), x.Size, 1f);
            }, x);
        }

        return Traced("rope", y, start);
    }
}

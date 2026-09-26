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
        Backend.Copy(Storage, y.Storage, Size);
        Backend.Rope(Storage, y.Storage, cos.Storage, sin.Storage, positions.Storage, rows, heads, steps, dim, half, interleaved, 1f);
        if (WillRecord(this))
        {
            var x = this;
            y.Record("rope", g =>
            {
                using var back = Empty(x._shape, x.Device, track: false);
                x.Backend.Copy(g.Storage, back.Storage, x.Size);
                x.Backend.Rope(g.Storage, back.Storage, cos.Storage, sin.Storage, positions.Storage, rows, heads, steps, dim, half, interleaved, -1f);
                x.Backend.Axpy(back.Storage, x.GradStorage(), x.Size, 1f);
            }, x);
        }

        return Traced("rope", y, start);
    }
}

using NeuralSharp.Diagnostics;

namespace NeuralSharp;

// Fused inference-only operations and incremental-decoding primitives. None of these record gradients:
// callers use them only when autograd is off (inference), and fall back to the differentiable ops otherwise.
public sealed partial class Tensor
{
    /// <summary>Copies <paramref name="destination"/>.Length elements starting at element <paramref name="offset"/> to the host.</summary>
    public void CopyTo(Span<float> destination, int offset = 0)
    {
        ThrowIfDisposed();
        if (offset < 0 || offset + destination.Length > Size)
        {
            throw new ArgumentOutOfRangeException(nameof(offset), $"[{offset}, {offset + destination.Length}) is outside a tensor of {Size} elements.");
        }

        Backend.DownloadRange(Storage, offset, destination);
    }

    /// <summary>softmax(scale · x + mask) over the last dimension in one kernel; the mask's rows repeat over the rows of x.</summary>
    internal Tensor ScaleMaskSoftmax(float scale, Tensor? mask)
    {
        ThrowIfDisposed();
        int cols = _shape[^1], rows = Size / cols;
        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty(_shape, Device);
        Backend.ScaleMaskSoftmax(Storage, mask?.Storage, y.Storage, rows, cols, mask is null ? 1 : mask.Size / cols, scale);
        return Traced("scale_mask_softmax", y, start);
    }

    /// <summary>LayerNorm over the last dimension with gamma/beta, in one kernel.</summary>
    internal Tensor LayerNormFused(Tensor gamma, Tensor beta, float eps)
    {
        ThrowIfDisposed();
        int cols = _shape[^1];
        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty(_shape, Device);
        Backend.LayerNormFused(Storage, gamma.Storage, beta.Storage, y.Storage, Size / cols, cols, eps);
        return Traced("layernorm_fused", y, start);
    }

    /// <summary>gelu(x + bias) over the last dimension, in one kernel.</summary>
    internal Tensor BiasGelu(Tensor bias)
    {
        ThrowIfDisposed();
        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty(_shape, Device);
        Backend.BiasGelu(Storage, bias.Storage, y.Storage, Size, bias.Size);
        return Traced("bias_gelu", y, start);
    }

    /// <summary>[rows, capacity] causal mask for queries at positions position..position+rows-1 (position read on the device).</summary>
    internal static Tensor DecoderMask(Tensor position, int rows, int capacity)
    {
        var mask = Empty([rows, capacity], position.Device);
        position.Backend.DecoderMask(position.Storage, mask.Storage, rows, capacity);
        return mask;
    }

    /// <summary>Writes [heads, steps, dim] keys or values into a [heads, capacity, dim] cache at the device-side position.</summary>
    internal static void WriteKeyValues(Tensor source, Tensor cache, Tensor position)
    {
        int heads = source._shape[0], steps = source._shape[1], dim = source._shape[2];
        source.Backend.KeyValueWrite(source.Storage, cache.Storage, position.Storage, heads, steps, cache._shape[1], dim);
    }

    /// <summary>Adds <paramref name="value"/> to every element in place (not recorded by autograd).</summary>
    internal void AddInPlace(float value)
    {
        ThrowIfDisposed();
        Backend.Affine(Storage, Storage, Size, 1f, value);
    }

    /// <summary>Sets every element in place (not recorded by autograd).</summary>
    internal void FillInPlace(float value)
    {
        ThrowIfDisposed();
        Backend.Fill(Storage, Size, value);
    }
}

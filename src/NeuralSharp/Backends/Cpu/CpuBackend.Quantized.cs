using System.Numerics;
using System.Runtime.InteropServices;

namespace NeuralSharp.Backends.Cpu;

// Int8 weight-only quantization: signed bytes packed four per float element along each weight row.
internal sealed partial class CpuBackend
{
    private const int Int8Block = 256;   // columns per parallel work item (a multiple of every SIMD width)

    public override void Int8MatMul(Storage x, Storage q, Storage scales, Storage y, int m, int n, int k)
    {
        float[] xv = D(x), sv = D(scales), yv = D(y);
        int stride = (n + 3) / 4 * 4;                                  // bytes per weight row
        int blocks = (n + Int8Block - 1) / Int8Block;
        For(blocks, (long)m * n * k, (first, last) =>
        {
            var weights = Bytes(q, k * stride);                          // spans cannot be captured; re-derive per worker
            var acc = new float[Int8Block];
            for (int block = first; block < last; block++)
            {
                int j0 = block * Int8Block, width = Math.Min(Int8Block, n - j0);
                for (int r = 0; r < m; r++)
                {
                    Array.Clear(acc);
                    int xo = r * k;
                    for (int kk = 0; kk < k; kk++)
                    {
                        float xk = xv[xo + kk];
                        if (xk != 0f)
                        {
                            AddScaled(acc.AsSpan(0, width), weights.Slice(kk * stride + j0, width), xk);
                        }
                    }

                    int yo = r * n + j0;
                    for (int j = 0; j < width; j++)
                    {
                        yv[yo + j] = acc[j] * sv[j0 + j];
                    }
                }
            }
        });
    }

    public override void Int8Dequantize(Storage q, Storage scales, Storage w, int k, int n)
    {
        float[] sv = D(scales), wv = D(w);
        int stride = (n + 3) / 4 * 4;
        For(k, (long)k * n, (first, last) =>
        {
            var weights = Bytes(q, k * stride);
            for (int kk = first; kk < last; kk++)
            {
                var row = weights.Slice(kk * stride, n);
                int o = kk * n;
                for (int j = 0; j < n; j++)
                {
                    wv[o + j] = row[j] * sv[j];
                }
            }
        });
    }

    private static ReadOnlySpan<sbyte> Bytes(Storage q, int count) => MemoryMarshal.Cast<float, sbyte>(D(q).AsSpan())[..count];

    // acc[j] += scale · (float)q[j], vectorized: bytes widen to shorts, then ints, then floats.
    private static void AddScaled(Span<float> acc, ReadOnlySpan<sbyte> q, float scale)
    {
        int j = 0;
        if (Vector.IsHardwareAccelerated && q.Length >= Vector<sbyte>.Count)
        {
            var s = new Vector<float>(scale);
            int floats = Vector<float>.Count;
            for (; j <= q.Length - Vector<sbyte>.Count; j += Vector<sbyte>.Count)
            {
                Vector.Widen(new Vector<sbyte>(q[j..]), out var lowShorts, out var highShorts);
                Vector.Widen(lowShorts, out var i0, out var i1);
                Vector.Widen(highShorts, out var i2, out var i3);
                var a = acc[j..];
                (Vector.ConvertToSingle(i0) * s + new Vector<float>(a)).CopyTo(a);
                (Vector.ConvertToSingle(i1) * s + new Vector<float>(a[floats..])).CopyTo(a[floats..]);
                (Vector.ConvertToSingle(i2) * s + new Vector<float>(a[(2 * floats)..])).CopyTo(a[(2 * floats)..]);
                (Vector.ConvertToSingle(i3) * s + new Vector<float>(a[(3 * floats)..])).CopyTo(a[(3 * floats)..]);
            }
        }

        for (; j < q.Length; j++)
        {
            acc[j] += scale * q[j];
        }
    }
}

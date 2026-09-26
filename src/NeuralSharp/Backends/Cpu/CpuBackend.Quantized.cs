using System.Numerics;
using System.Runtime.InteropServices;

namespace NeuralSharp.Backends.Cpu;

// Int8 weight-only quantization: signed bytes packed four per float element along each weight row.
internal sealed partial class CpuBackend
{
    private const int Int8MinBlock = 64;   // fewest columns per parallel work item (a multiple of every SIMD width)

    public override void Int8MatMul(Storage x, Storage q, Storage scales, Storage y, int m, int n, int k)
    {
        float[] xv = D(x), sv = D(scales), yv = D(y);
        int stride = (n + 3) / 4 * 4;                                  // bytes per weight row
        // About two column blocks per thread, whole SIMD vectors wide.
        int threads = Math.Max(1, ComputeResources.ParallelOptions.MaxDegreeOfParallelism);
        int blockSize = Math.Max(Int8MinBlock, (n / (2 * threads) + Int8MinBlock - 1) / Int8MinBlock * Int8MinBlock);
        int blocks = (n + blockSize - 1) / blockSize;
        For(blocks, (long)m * n * k, (first, last) =>
        {
            var weights = Bytes(q, k * stride);                          // spans cannot be captured; re-derive per worker
            var acc = new float[blockSize];
            for (int block = first; block < last; block++)
            {
                int j0 = block * blockSize, width = Math.Min(blockSize, n - j0);
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

    public override void KeyValueWriteInt8(Storage source, Storage cache, Storage scales, Storage position, int heads, int steps, int capacity, int dim)
    {
        float[] src = D(source), sv = D(scales);
        var bytes = MemoryMarshal.Cast<float, sbyte>(D(cache).AsSpan());
        int start = (int)D(position)[0], stride = (dim + 3) / 4 * 4;
        for (int row = 0; row < heads * steps; row++)
        {
            int slot = row / steps * capacity + start + row % steps;
            var x = src.AsSpan(row * dim, dim);
            float max = 0f;
            foreach (float v in x)
            {
                max = MathF.Max(max, MathF.Abs(v));
            }

            float scale = max / 127f, inverse = max > 0f ? 1f / scale : 0f;
            sv[slot] = scale;
            var target = bytes.Slice(slot * stride, stride);
            target.Clear();
            for (int d = 0; d < dim; d++)
            {
                target[d] = (sbyte)Math.Clamp((int)MathF.Round(x[d] * inverse, MidpointRounding.ToEven), -127, 127);
            }
        }
    }

    public override void AttentionDecode(Storage q, Storage keys, Storage values, Storage position, Storage y, int heads, int rowsPerHead,
        int steps, int capacity, int dim, float scale) =>
        AttentionTiled(q, keys, values, position, y, null, heads, rowsPerHead, steps, capacity, dim, scale);

    public override void AttentionTiled(Storage q, Storage keys, Storage values, Storage position, Storage y, Storage? logSumExp, int heads,
        int rowsPerHead, int steps, int capacity, int dim, float scale)
    {
        float[] qv = D(q), kv = D(keys), vv = D(values), yv = D(y);
        float[]? lv = logSumExp is null ? null : D(logSumExp);
        int position0 = (int)D(position)[0];
        For(heads * rowsPerHead, (long)heads * rowsPerHead * dim * Math.Max(1, position0), (first, last) =>
        {
            var scores = new float[capacity];
            for (int row = first; row < last; row++)
            {
                int h = row / rowsPerHead, count = Math.Min(position0 + row % rowsPerHead % steps, capacity - 1) + 1;
                var query = qv.AsSpan(row * dim, dim);
                float max = float.NegativeInfinity;
                for (int c = 0; c < count; c++)
                {
                    float dot = 0f;
                    var key = kv.AsSpan((int)(((long)h * capacity + c) * dim), dim);
                    for (int d = 0; d < dim; d++)
                    {
                        dot += query[d] * key[d];
                    }

                    scores[c] = dot * scale;
                    max = MathF.Max(max, scores[c]);
                }

                float sum = 0f;
                for (int c = 0; c < count; c++)
                {
                    scores[c] = MathF.Exp(scores[c] - max);
                    sum += scores[c];
                }

                if (lv is not null)
                {
                    lv[row] = max + MathF.Log(sum);
                }

                var output = yv.AsSpan(row * dim, dim);
                output.Clear();
                for (int c = 0; c < count; c++)
                {
                    float weight = scores[c] / sum;
                    var value = vv.AsSpan((int)(((long)h * capacity + c) * dim), dim);
                    for (int d = 0; d < dim; d++)
                    {
                        output[d] += weight * value[d];
                    }
                }
            }
        });
    }

    public override void AttentionTiledBackward(Storage q, Storage keys, Storage values, Storage output, Storage logSumExp, Storage dOutput,
        Storage dq, Storage dkeys, Storage dvalues, int heads, int rowsPerHead, int steps, int capacity, int dim, float scale)
    {
        float[] qv = D(q), kv = D(keys), vv = D(values), ov = D(output), lv = D(logSumExp), gv = D(dOutput);
        float[] dqv = D(dq), dkv = D(dkeys), dvv = D(dvalues);
        For(heads, (long)heads * rowsPerHead * capacity * dim, (first, last) =>
        {
            for (int h = first; h < last; h++)
            {
                for (int i = 0; i < rowsPerHead; i++)
                {
                    int row = h * rowsPerHead + i, count = Math.Min(i % steps, capacity - 1) + 1;
                    var query = qv.AsSpan(row * dim, dim);
                    var gradOut = gv.AsSpan(row * dim, dim);
                    float delta = 0f;
                    for (int d = 0; d < dim; d++)
                    {
                        delta += gradOut[d] * ov[row * dim + d];
                    }

                    for (int c = 0; c < count; c++)
                    {
                        int key = (h * capacity + c) * dim;
                        float dot = 0f, dp = 0f;
                        for (int d = 0; d < dim; d++)
                        {
                            dot += query[d] * kv[key + d];
                            dp += gradOut[d] * vv[key + d];
                        }

                        float p = MathF.Exp(dot * scale - lv[row]), ds = p * (dp - delta);
                        for (int d = 0; d < dim; d++)
                        {
                            dqv[row * dim + d] += scale * ds * kv[key + d];
                            dkv[key + d] += scale * ds * query[d];
                            dvv[key + d] += p * gradOut[d];
                        }
                    }
                }
            }
        });
    }

    public override void AttentionScoresInt8(Storage q, Storage cache, Storage scales, Storage y, int rows, int steps, int capacity, int dim)
    {
        float[] qv = D(q), sv = D(scales), yv = D(y);
        int stride = (dim + 3) / 4 * 4;
        For(rows, (long)rows * steps * capacity * dim, (first, last) =>
        {
            var keys = MemoryMarshal.Cast<float, sbyte>(D(cache).AsSpan());
            for (int r = first; r < last; r++)
            {
                for (int t = 0; t < steps; t++)
                {
                    var qr = qv.AsSpan((r * steps + t) * dim, dim);
                    for (int c = 0; c < capacity; c++)
                    {
                        var k = keys.Slice((r * capacity + c) * stride, dim);
                        float sum = 0f;
                        for (int d = 0; d < dim; d++)
                        {
                            sum += qr[d] * k[d];
                        }

                        yv[(r * steps + t) * capacity + c] = sum * sv[r * capacity + c];
                    }
                }
            }
        });
    }

    public override void AttentionContextInt8(Storage weights, Storage cache, Storage scales, Storage y, int rows, int steps, int capacity, int dim)
    {
        float[] wv = D(weights), sv = D(scales), yv = D(y);
        int stride = (dim + 3) / 4 * 4;
        For(rows, (long)rows * steps * capacity * dim, (first, last) =>
        {
            var values = MemoryMarshal.Cast<float, sbyte>(D(cache).AsSpan());
            for (int r = first; r < last; r++)
            {
                for (int t = 0; t < steps; t++)
                {
                    var output = yv.AsSpan((r * steps + t) * dim, dim);
                    output.Clear();
                    for (int c = 0; c < capacity; c++)
                    {
                        float a = wv[(r * steps + t) * capacity + c] * sv[r * capacity + c];
                        if (a != 0f)
                        {
                            AddScaled(output, values.Slice((r * capacity + c) * stride, dim), a);
                        }
                    }
                }
            }
        });
    }

    public override void RmsNormAffine(Storage x, Storage gain, Storage y, int rows, int cols, float eps, float offset)
    {
        float[] xv = D(x), gv = D(gain), yv = D(y);
        For(rows, (long)rows * cols * 2, (first, last) =>
        {
            for (int r = first; r < last; r++)
            {
                var row = xv.AsSpan(r * cols, cols);
                float sum = 0f;
                foreach (float v in row)
                {
                    sum = MathF.FusedMultiplyAdd(v, v, sum);
                }

                float scale = 1f / MathF.Sqrt(sum / cols + eps);
                var output = yv.AsSpan(r * cols, cols);
                for (int j = 0; j < cols; j++)
                {
                    output[j] = row[j] * scale * (gv[j] + offset);
                }
            }
        });
    }

    // act(g) and act'(g) for the gated feed-forward activations.
    private static (float Value, float Slope) Activation(float g, int kind)
    {
        switch (kind)
        {
            case 0:
            {
                float s = 1f / (1f + MathF.Exp(-g));
                return (g * s, s * (1f + g * (1f - s)));
            }

            case 1:
            {
                const float K = 0.7978845608f, C = 0.044715f;
                float t = MathF.Tanh(K * (g + C * g * g * g));
                return (0.5f * g * (1f + t), 0.5f * (1f + t) + 0.5f * g * (1f - t * t) * K * (1f + 3f * C * g * g));
            }

            default:
                return (g > 0f ? g : 0f, g > 0f ? 1f : 0f);
        }
    }

    public override void GatedActivation(Storage gate, Storage up, Storage y, int n, int kind)
    {
        float[] gv = D(gate), uv = D(up), yv = D(y);
        For(n, (long)n * 8, (first, last) =>
        {
            for (int i = first; i < last; i++)
            {
                yv[i] = Activation(gv[i], kind).Value * uv[i];
            }
        });
    }

    public override void GatedActivationBackward(Storage gate, Storage up, Storage dy, Storage dgate, Storage dup, int n, int kind, int flags)
    {
        float[] gv = D(gate), uv = D(up), dv = D(dy), dg = D(dgate), du = D(dup);
        For(n, (long)n * 8, (first, last) =>
        {
            for (int i = first; i < last; i++)
            {
                var (value, slope) = Activation(gv[i], kind);
                if ((flags & 1) != 0)
                {
                    dg[i] += dv[i] * uv[i] * slope;
                }

                if ((flags & 2) != 0)
                {
                    du[i] += dv[i] * value;
                }
            }
        });
    }

    public override void RmsNorm(Storage x, Storage y, Storage inv, int rows, int cols, float eps)
    {
        float[] xv = D(x), yv = D(y), iv = D(inv);
        For(rows, (long)rows * cols * 2, (first, last) =>
        {
            for (int r = first; r < last; r++)
            {
                var row = xv.AsSpan(r * cols, cols);
                float sum = 0f;
                foreach (float v in row)
                {
                    sum = MathF.FusedMultiplyAdd(v, v, sum);
                }

                float scale = 1f / MathF.Sqrt(sum / cols + eps);
                iv[r] = scale;
                var output = yv.AsSpan(r * cols, cols);
                for (int j = 0; j < cols; j++)
                {
                    output[j] = row[j] * scale;
                }
            }
        });
    }

    public override void RmsNormBackward(Storage dy, Storage y, Storage inv, Storage dx, int rows, int cols)
    {
        float[] gv = D(dy), yv = D(y), iv = D(inv), dv = D(dx);
        For(rows, (long)rows * cols * 3, (first, last) =>
        {
            for (int r = first; r < last; r++)
            {
                int o = r * cols;
                float dot = 0f;
                for (int j = 0; j < cols; j++)
                {
                    dot = MathF.FusedMultiplyAdd(gv[o + j], yv[o + j], dot);
                }

                float mean = dot / cols;
                for (int j = 0; j < cols; j++)
                {
                    dv[o + j] += iv[r] * (gv[o + j] - yv[o + j] * mean);
                }
            }
        });
    }

    public override void Rope(Storage x, Storage y, Storage cos, Storage sin, Storage positions, int rows, int heads, int steps, int dim, int half, bool interleaved, float sign)
    {
        float[] xv = D(x), yv = D(y), cv = D(cos), sv = D(sin), pv = D(positions);
        For(rows, (long)rows * half * 4, (first, last) =>
        {
            for (int r = first; r < last; r++)
            {
                int position = (int)pv[r / heads % steps], o = r * dim;
                for (int p = 0; p < half; p++)
                {
                    float c = cv[position * half + p], s = sv[position * half + p] * sign;
                    int i = o + (interleaved ? 2 * p : p), j = o + (interleaved ? 2 * p + 1 : p + half);
                    float a = xv[i], b = xv[j];
                    yv[i] = MathF.FusedMultiplyAdd(a, c, -(b * s));
                    yv[j] = MathF.FusedMultiplyAdd(b, c, a * s);
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

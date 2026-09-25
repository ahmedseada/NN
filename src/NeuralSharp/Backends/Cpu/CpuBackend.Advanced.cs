using System.Numerics;
using System.Runtime.InteropServices;

namespace NeuralSharp.Backends.Cpu;

// Kernels for classification, normalization, embeddings, convolution, pooling and N-D shape operations.
internal sealed partial class CpuBackend
{
    /// <summary>Runs body(start, end) over [0, count), splitting across cores when work (≈ total flops) is large.</summary>
    private static void For(int count, long work, Action<int, int> body)
    {
        if (count <= 1 || work < ParallelThreshold || !ComputeResources.AllowParallel)
        {
            body(0, count);
            return;
        }

        int chunks = Math.Min(count, ComputeResources.MaxCpuThreads * 4);
        int size = (count + chunks - 1) / chunks;
        Parallel.For(0, chunks, ComputeResources.ParallelOptions, c =>
        {
            int start = c * size;
            if (start < count)
            {
                body(start, Math.Min(start + size, count));
            }
        });
    }

    public override void BatchedMatMul(Storage a, Storage b, Storage c, int batch, int m, int n, int k, bool transA, bool transB, float beta)
    {
        float[] av = D(a), bv = D(b), cv = D(c);
        if (batch == 1)
        {
            CpuMatMul.Multiply(av, bv, cv, m, n, k, transA, transB, beta);
            return;
        }

        // Many small products (attention heads): parallelize across the batch, each product single-threaded.
        int mk = m * k, kn = k * n, mn = m * n;
        For(batch, (long)batch * m * n * k, (start, end) =>
        {
            for (int i = start; i < end; i++)
            {
                CpuMatMul.Multiply(av, i * mk, bv, i * kn, cv, i * mn, m, n, k, transA, transB, beta);
            }
        });
    }

    public override void Softmax(Storage x, Storage y, int rows, int cols, bool log)
    {
        float[] xv = D(x), yv = D(y);
        For(rows, (long)rows * cols * 8, (start, end) =>
        {
            for (int r = start; r < end; r++)
            {
                var xs = xv.AsSpan(r * cols, cols);
                var ys = yv.AsSpan(r * cols, cols);
                float max = float.NegativeInfinity;
                foreach (float v in xs)
                {
                    max = MathF.Max(max, v);
                }

                double sum = 0;
                for (int j = 0; j < cols; j++)
                {
                    float e = MathF.Exp(xs[j] - max);
                    ys[j] = e;
                    sum += e;
                }

                if (log)
                {
                    float logSum = (float)Math.Log(sum) + max;
                    for (int j = 0; j < cols; j++)
                    {
                        ys[j] = xs[j] - logSum;
                    }
                }
                else
                {
                    float inv = (float)(1.0 / sum);
                    for (int j = 0; j < cols; j++)
                    {
                        ys[j] *= inv;
                    }
                }
            }
        });
    }

    public override void SoftmaxBackward(Storage y, Storage dy, Storage dx, int rows, int cols, bool log)
    {
        float[] yv = D(y), gv = D(dy), dv = D(dx);
        For(rows, (long)rows * cols * 4, (start, end) =>
        {
            for (int r = start; r < end; r++)
            {
                int o = r * cols;
                double dot = 0;
                for (int j = 0; j < cols; j++)
                {
                    dot += log ? gv[o + j] : gv[o + j] * yv[o + j];
                }

                float s = (float)dot;
                for (int j = 0; j < cols; j++)
                {
                    dv[o + j] += log ? gv[o + j] - MathF.Exp(yv[o + j]) * s : yv[o + j] * (gv[o + j] - s);
                }
            }
        });
    }

    private static int ArgMaxRow(ReadOnlySpan<float> row)
    {
        int best = 0;
        for (int j = 1; j < row.Length; j++)
        {
            if (row[j] > row[best])
            {
                best = j;
            }
        }

        return best;
    }

    public override void ArgMax(Storage x, Storage y, int rows, int cols)
    {
        float[] xv = D(x), yv = D(y);
        for (int r = 0; r < rows; r++)
        {
            yv[r] = ArgMaxRow(xv.AsSpan(r * cols, cols));
        }
    }

    public override void ClassMatch(Storage predictions, Storage targets, Storage y, int rows, int cols, float threshold)
    {
        float[] pv = D(predictions), tv = D(targets), yv = D(y);
        for (int r = 0; r < rows; r++)
        {
            bool match = cols == 1
                ? (pv[r] >= threshold) == (tv[r] >= 0.5f)
                : ArgMaxRow(pv.AsSpan(r * cols, cols)) == ArgMaxRow(tv.AsSpan(r * cols, cols));
            yv[r] = match ? 1f : 0f;
        }
    }

    public override void NormStats(Storage x, Storage mean, Storage variance, Storage invStd, int outer, int groups, int inner, float eps)
    {
        float[] xv = D(x), mv = D(mean), vv = D(variance), sv = D(invStd);
        int m = outer * inner;
        For(groups, (long)groups * m * 2, (start, end) =>
        {
            for (int g = start; g < end; g++)
            {
                double sum = 0;
                for (int o = 0; o < outer; o++)
                {
                    var block = xv.AsSpan((o * groups + g) * inner, inner);
                    foreach (float v in block)
                    {
                        sum += v;
                    }
                }

                double mu = sum / m;
                double sq = 0;
                for (int o = 0; o < outer; o++)
                {
                    var block = xv.AsSpan((o * groups + g) * inner, inner);
                    foreach (float v in block)
                    {
                        sq += (v - mu) * (v - mu);
                    }
                }

                double var = sq / m;
                mv[g] = (float)mu;
                vv[g] = (float)var;
                sv[g] = (float)(1.0 / Math.Sqrt(var + eps));
            }
        });
    }

    public override void NormApply(Storage x, Storage mean, Storage invStd, Storage y, int outer, int groups, int inner)
    {
        float[] xv = D(x), mv = D(mean), sv = D(invStd), yv = D(y);
        For(outer * groups, (long)outer * groups * inner, (start, end) =>
        {
            for (int b = start; b < end; b++)
            {
                int g = b % groups;
                float mu = mv[g], s = sv[g];
                int o = b * inner;
                for (int i = 0; i < inner; i++)
                {
                    yv[o + i] = (xv[o + i] - mu) * s;
                }
            }
        });
    }

    public override void NormBackward(Storage dxhat, Storage xhat, Storage sum1, Storage sum2, Storage invStd, Storage dx, int outer, int groups, int inner)
    {
        float[] gv = D(dxhat), hv = D(xhat), s1 = D(sum1), s2 = D(sum2), sv = D(invStd), dv = D(dx);
        float m = outer * inner;
        For(outer * groups, (long)outer * groups * inner, (start, end) =>
        {
            for (int b = start; b < end; b++)
            {
                int g = b % groups;
                float scale = sv[g] / m, a = s1[g], c = s2[g];
                int o = b * inner;
                for (int i = 0; i < inner; i++)
                {
                    dv[o + i] += scale * (m * gv[o + i] - a - hv[o + i] * c);
                }
            }
        });
    }

    public override void GroupScaleShift(Storage x, Storage? scale, Storage? shift, Storage y, int n, int groups, int inner, bool accumulate)
    {
        float[] xv = D(x), yv = D(y);
        float[]? sc = scale is null ? null : D(scale);
        float[]? sh = shift is null ? null : D(shift);
        int blocks = n / inner;
        For(blocks, n, (start, end) =>
        {
            for (int b = start; b < end; b++)
            {
                int g = b % groups;
                float s = sc?[g] ?? 1f, t = sh?[g] ?? 0f;
                int o = b * inner;
                for (int i = 0; i < inner; i++)
                {
                    float v = xv[o + i] * s + t;
                    yv[o + i] = accumulate ? yv[o + i] + v : v;
                }
            }
        });
    }

    public override void GroupReduce(Storage a, Storage? b, Storage sumA, Storage? sumAB, int outer, int groups, int inner)
    {
        float[] av = D(a), s1 = D(sumA);
        float[]? bv = b is null ? null : D(b);
        float[]? s2 = sumAB is null ? null : D(sumAB);
        For(groups, (long)groups * outer * inner * 2, (start, end) =>
        {
            for (int g = start; g < end; g++)
            {
                double sa = 0, sab = 0;
                for (int o = 0; o < outer; o++)
                {
                    int offset = (o * groups + g) * inner;
                    for (int i = 0; i < inner; i++)
                    {
                        float v = av[offset + i];
                        sa += v;
                        if (bv is not null)
                        {
                            sab += v * bv[offset + i];
                        }
                    }
                }

                s1[g] += (float)sa;
                if (s2 is not null)
                {
                    s2[g] += (float)sab;
                }
            }
        });
    }

    public override void InvSqrt(Storage x, Storage y, int n, float eps)
    {
        float[] xv = D(x), yv = D(y);
        for (int i = 0; i < n; i++)
        {
            yv[i] = 1f / MathF.Sqrt(xv[i] + eps);
        }
    }

    public override void Gather(Storage table, Storage indices, Storage y, int count, int dim, int vocabulary)
    {
        float[] tv = D(table), iv = D(indices), yv = D(y);
        for (int i = 0; i < count; i++)
        {
            int index = CheckIndex(iv[i], vocabulary);
            tv.AsSpan(index * dim, dim).CopyTo(yv.AsSpan(i * dim, dim));
        }
    }

    public override void ScatterAdd(Storage dy, Storage indices, Storage dtable, int count, int dim, int vocabulary)
    {
        float[] gv = D(dy), iv = D(indices), tv = D(dtable);
        for (int i = 0; i < count; i++)
        {
            int index = CheckIndex(iv[i], vocabulary);
            var target = tv.AsSpan(index * dim, dim);
            AddInPlace(target, gv.AsSpan(i * dim, dim));
        }
    }

    private static int CheckIndex(float value, int vocabulary)
    {
        int index = (int)value;
        if ((uint)index >= (uint)vocabulary || index != value)
        {
            throw new IndexOutOfRangeException($"Embedding index {value} is not an integer in [0, {vocabulary}).");
        }

        return index;
    }

    private static void AddInPlace(Span<float> target, ReadOnlySpan<float> source)
    {
        var tv = MemoryMarshal.Cast<float, Vector<float>>(target);
        var sv = MemoryMarshal.Cast<float, Vector<float>>(source);
        for (int i = 0; i < tv.Length; i++)
        {
            tv[i] += sv[i];
        }

        for (int i = tv.Length * Vector<float>.Count; i < target.Length; i++)
        {
            target[i] += source[i];
        }
    }

    public override void Im2Col(Storage x, Storage cols, in ConvGeometry g)
    {
        float[] xv = D(x), cv = D(cols);
        var geo = g;
        int oh = g.OH, ow = g.OW, patch = g.PatchSize;
        For(g.N * oh, (long)g.Positions * patch, (start, end) =>
        {
            for (int noh = start; noh < end; noh++)
            {
                int n = noh / oh, y = noh % oh;
                for (int x0 = 0; x0 < ow; x0++)
                {
                    int row = (noh * ow + x0) * patch;
                    int col = 0;
                    for (int c = 0; c < geo.C; c++)
                    {
                        int plane = (n * geo.C + c) * geo.H;
                        for (int kh = 0; kh < geo.KH; kh++)
                        {
                            int ih = y * geo.SH - geo.PH + kh;
                            for (int kw = 0; kw < geo.KW; kw++, col++)
                            {
                                int iw = x0 * geo.SW - geo.PW + kw;
                                cv[row + col] = (uint)ih < (uint)geo.H && (uint)iw < (uint)geo.W ? xv[(plane + ih) * geo.W + iw] : 0f;
                            }
                        }
                    }
                }
            }
        });
    }

    public override void Col2Im(Storage dcols, Storage dx, in ConvGeometry g)
    {
        float[] cv = D(dcols), dv = D(dx);
        var geo = g;
        int oh = g.OH, ow = g.OW, patch = g.PatchSize;

        // Images write to disjoint parts of dx, so they can run in parallel without atomics.
        For(g.N, (long)g.Positions * patch, (start, end) =>
        {
            for (int n = start; n < end; n++)
            {
                for (int y = 0; y < oh; y++)
                {
                    for (int x0 = 0; x0 < ow; x0++)
                    {
                        int row = ((n * oh + y) * ow + x0) * patch;
                        int col = 0;
                        for (int c = 0; c < geo.C; c++)
                        {
                            int plane = (n * geo.C + c) * geo.H;
                            for (int kh = 0; kh < geo.KH; kh++)
                            {
                                int ih = y * geo.SH - geo.PH + kh;
                                for (int kw = 0; kw < geo.KW; kw++, col++)
                                {
                                    int iw = x0 * geo.SW - geo.PW + kw;
                                    if ((uint)ih < (uint)geo.H && (uint)iw < (uint)geo.W)
                                    {
                                        dv[(plane + ih) * geo.W + iw] += cv[row + col];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
    }

    public override void MaxPool(Storage x, Storage y, Storage argmax, in ConvGeometry g)
    {
        float[] xv = D(x), yv = D(y);
        var geo = g;
        int oh = g.OH, ow = g.OW;
        var indices = new int[g.N * g.C * oh * ow];
        For(g.N * g.C, (long)indices.Length * g.KH * g.KW, (start, end) =>
        {
            for (int nc = start; nc < end; nc++)
            {
                int plane = nc * geo.H * geo.W;
                for (int y0 = 0; y0 < oh; y0++)
                {
                    for (int x0 = 0; x0 < ow; x0++)
                    {
                        float best = float.NegativeInfinity;
                        int bestIndex = plane;
                        for (int kh = 0; kh < geo.KH; kh++)
                        {
                            int ih = y0 * geo.SH - geo.PH + kh;
                            if ((uint)ih >= (uint)geo.H)
                            {
                                continue;
                            }

                            for (int kw = 0; kw < geo.KW; kw++)
                            {
                                int iw = x0 * geo.SW - geo.PW + kw;
                                if ((uint)iw < (uint)geo.W && xv[plane + ih * geo.W + iw] > best)
                                {
                                    best = xv[plane + ih * geo.W + iw];
                                    bestIndex = plane + ih * geo.W + iw;
                                }
                            }
                        }

                        int o = (nc * oh + y0) * ow + x0;
                        yv[o] = best;
                        indices[o] = bestIndex;
                    }
                }
            }
        });
        indices.AsSpan().CopyTo(MemoryMarshal.Cast<float, int>(D(argmax).AsSpan()));
    }

    public override void MaxPoolBackward(Storage dy, Storage argmax, Storage dx, int count)
    {
        float[] gv = D(dy), dv = D(dx);
        var indices = MemoryMarshal.Cast<float, int>(D(argmax).AsSpan(0, count));
        for (int i = 0; i < count; i++)
        {
            dv[indices[i]] += gv[i];
        }
    }

    public override void Permute(Storage x, Storage y, ReadOnlySpan<int> outShape, ReadOnlySpan<int> inStrides, bool accumulate)
    {
        float[] xv = D(x), yv = D(y);
        int rank = outShape.Length;
        int[] shape = outShape.ToArray(), strides = inStrides.ToArray();
        int total = 1;
        foreach (int d in shape)
        {
            total *= d;
        }

        // The innermost output dimension is walked with a fixed input stride; outer coordinates are decoded once per row.
        int last = shape[rank - 1], lastStride = strides[rank - 1];
        int rows = total / Math.Max(last, 1);
        For(rows, total, (start, end) =>
        {
            for (int r = start; r < end; r++)
            {
                int rem = r, offset = 0;
                for (int d = rank - 2; d >= 0; d--)
                {
                    offset += rem % shape[d] * strides[d];
                    rem /= shape[d];
                }

                int o = r * last;
                if (accumulate)
                {
                    for (int j = 0; j < last; j++)
                    {
                        yv[o + j] += xv[offset + j * lastStride];
                    }
                }
                else
                {
                    for (int j = 0; j < last; j++)
                    {
                        yv[o + j] = xv[offset + j * lastStride];
                    }
                }
            }
        });
    }

    public override void Copy2D(Storage src, int srcOffset, int srcStride, Storage dst, int dstOffset, int dstStride, int rows, int cols, bool accumulate)
    {
        float[] sv = D(src), dv = D(dst);
        for (int r = 0; r < rows; r++)
        {
            var from = sv.AsSpan(srcOffset + r * srcStride, cols);
            var to = dv.AsSpan(dstOffset + r * dstStride, cols);
            if (accumulate)
            {
                AddInPlace(to, from);
            }
            else
            {
                from.CopyTo(to);
            }
        }
    }

    public override void SumAxis(Storage x, Storage y, int outer, int dim, int inner, float scale, bool accumulate)
    {
        float[] xv = D(x), yv = D(y);
        For(outer, (long)outer * dim * inner, (start, end) =>
        {
            Span<float> acc = inner <= 4096 ? stackalloc float[inner] : new float[inner];
            for (int o = start; o < end; o++)
            {
                acc.Clear();
                for (int d = 0; d < dim; d++)
                {
                    AddInPlace(acc, xv.AsSpan((o * dim + d) * inner, inner));
                }

                var target = yv.AsSpan(o * inner, inner);
                for (int i = 0; i < inner; i++)
                {
                    target[i] = accumulate ? target[i] + scale * acc[i] : scale * acc[i];
                }
            }
        });
    }

    public override void BroadcastAxis(Storage dy, Storage dx, int outer, int dim, int inner, float scale)
    {
        float[] gv = D(dy), dv = D(dx);
        For(outer, (long)outer * dim * inner, (start, end) =>
        {
            for (int o = start; o < end; o++)
            {
                var source = gv.AsSpan(o * inner, inner);
                for (int d = 0; d < dim; d++)
                {
                    var target = dv.AsSpan((o * dim + d) * inner, inner);
                    for (int i = 0; i < inner; i++)
                    {
                        target[i] += scale * source[i];
                    }
                }
            }
        });
    }

    // ------------------------------------------------------------ element-wise kernels

    private readonly struct ExpKernel(float[] x, float[] y) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            var xs = x.AsSpan(start, end - start);
            var ys = y.AsSpan(start, end - start);
            var xv = MemoryMarshal.Cast<float, Vector<float>>(xs);
            var yv = MemoryMarshal.Cast<float, Vector<float>>(ys);
            for (int i = 0; i < xv.Length; i++)
            {
                yv[i] = Vector.Exp(xv[i]);
            }

            for (int i = xv.Length * Vector<float>.Count; i < xs.Length; i++)
            {
                ys[i] = MathF.Exp(xs[i]);
            }
        }
    }

    private readonly struct LogKernel(float[] x, float[] y) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            var xs = x.AsSpan(start, end - start);
            var ys = y.AsSpan(start, end - start);
            var xv = MemoryMarshal.Cast<float, Vector<float>>(xs);
            var yv = MemoryMarshal.Cast<float, Vector<float>>(ys);
            for (int i = 0; i < xv.Length; i++)
            {
                yv[i] = Vector.Log(xv[i]);
            }

            for (int i = xv.Length * Vector<float>.Count; i < xs.Length; i++)
            {
                ys[i] = MathF.Log(xs[i]);
            }
        }
    }

    private readonly struct LogBackwardKernel(float[] x, float[] dy, float[] dx) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            for (int i = start; i < end; i++)
            {
                dx[i] += dy[i] / x[i];
            }
        }
    }

    // GELU, tanh approximation: 0.5 x (1 + tanh(k (x + 0.044715 x³))), k = sqrt(2/π).
    private const float GeluK = 0.7978845608f;
    private const float GeluC = 0.044715f;

    private readonly struct GeluKernel(float[] x, float[] y) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            for (int i = start; i < end; i++)
            {
                float v = x[i];
                y[i] = 0.5f * v * (1f + MathF.Tanh(GeluK * (v + GeluC * v * v * v)));
            }
        }
    }

    private readonly struct GeluBackwardKernel(float[] x, float[] dy, float[] dx) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            for (int i = start; i < end; i++)
            {
                float v = x[i];
                float t = MathF.Tanh(GeluK * (v + GeluC * v * v * v));
                float derivative = 0.5f * (1f + t) + 0.5f * v * (1f - t * t) * GeluK * (1f + 3f * GeluC * v * v);
                dx[i] += dy[i] * derivative;
            }
        }
    }
}

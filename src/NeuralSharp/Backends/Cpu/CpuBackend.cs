using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace NeuralSharp.Backends.Cpu;

internal sealed class CpuStorage(CpuBackend backend, float[] data, int length) : Storage(backend, length)
{
    public readonly float[] Data = data;
}

/// <summary>
/// CPU implementation: pooled managed arrays, <see cref="Vector{T}"/> SIMD kernels, and
/// <see cref="Parallel"/> for large tensors. Small tensors stay on the calling thread so
/// tiny networks are not dominated by scheduling overhead.
/// </summary>
internal sealed class CpuBackend : Backend
{
    public static readonly CpuBackend Instance = new();

    /// <summary>Element count above which element-wise kernels split work across cores.</summary>
    internal const int ParallelThreshold = 1 << 16;

    private readonly Dictionary<int, Stack<float[]>> _pool = [];
    private readonly MemoryAccountant _memory = new(() => ComputeResources.CpuMemoryLimit, "the CPU");

    private CpuBackend()
    {
    }

    public override Storage Allocate(int length, bool zeroed)
    {
        long bytes = (long)length * sizeof(float);
        bool releaseCache = _memory.MustReleaseCacheFor(bytes); // throws when over the in-use limit
        float[]? data = null;
        lock (_pool)
        {
            if (_pool.TryGetValue(length, out var bucket) && bucket.Count > 0)
            {
                data = bucket.Pop();
                _memory.Reused(bytes);
            }
        }

        if (data is null)
        {
            if (releaseCache)
            {
                ReleaseCachedMemory();
            }

            data = zeroed ? new float[length] : GC.AllocateUninitializedArray<float>(length);
            _memory.Allocated(bytes);
        }
        else if (zeroed)
        {
            Array.Clear(data);
        }

        return new CpuStorage(this, data, length);
    }

    public override void Return(Storage storage)
    {
        var data = ((CpuStorage)storage).Data;
        lock (_pool)
        {
            if (!_pool.TryGetValue(data.Length, out var bucket))
            {
                _pool[data.Length] = bucket = new Stack<float[]>();
            }

            bucket.Push(data);
            _memory.Returned((long)data.Length * sizeof(float));
        }
    }

    public override MemoryUsage GetMemoryUsage() => _memory.Usage;

    public override void ReleaseCachedMemory()
    {
        lock (_pool)
        {
            foreach (var (length, bucket) in _pool)
            {
                _memory.Freed((long)length * sizeof(float) * bucket.Count);
                bucket.Clear();
            }
        }
    }

    public override void Upload(ReadOnlySpan<float> source, Storage destination) => source.CopyTo(D(destination));

    public override void Download(Storage source, Span<float> destination) => D(source).AsSpan(0, destination.Length).CopyTo(destination);

    public override void Fill(Storage y, int n, float value) => D(y).AsSpan(0, n).Fill(value);

    public override void Copy(Storage x, Storage y, int n) => D(x).AsSpan(0, n).CopyTo(D(y));

    public override void Unary(UnaryOp op, Storage x, Storage y, int n)
    {
        switch (op)
        {
            case UnaryOp.Sigmoid: Run(new SigmoidKernel(D(x), D(y)), n); break;
            case UnaryOp.Tanh: Run(new TanhKernel(D(x), D(y)), n); break;
            case UnaryOp.Relu: Run(new ReluKernel(D(x), D(y)), n); break;
            case UnaryOp.Square: Run(new SquareKernel(D(x), D(y)), n); break;
            case UnaryOp.Abs: Run(new AbsKernel(D(x), D(y)), n); break;
            default: throw new ArgumentOutOfRangeException(nameof(op));
        }
    }

    public override void UnaryBackward(UnaryOp op, Storage x, Storage y, Storage dy, Storage dx, int n)
    {
        switch (op)
        {
            case UnaryOp.Sigmoid: Run(new SigmoidBackwardKernel(D(y), D(dy), D(dx)), n); break;
            case UnaryOp.Tanh: Run(new TanhBackwardKernel(D(y), D(dy), D(dx)), n); break;
            case UnaryOp.Relu: Run(new ReluBackwardKernel(D(x), D(dy), D(dx)), n); break;
            case UnaryOp.Square: Run(new SquareBackwardKernel(D(x), D(dy), D(dx)), n); break;
            case UnaryOp.Abs: Run(new AbsBackwardKernel(D(x), D(dy), D(dx)), n); break;
            default: throw new ArgumentOutOfRangeException(nameof(op));
        }
    }

    public override void Binary(BinaryOp op, Storage a, Storage b, Storage c, int n)
    {
        switch (op)
        {
            case BinaryOp.Add: Run(new AddKernel(D(a), D(b), D(c)), n); break;
            case BinaryOp.Sub: Run(new SubKernel(D(a), D(b), D(c)), n); break;
            case BinaryOp.Mul: Run(new MulKernel(D(a), D(b), D(c)), n); break;
            default: throw new ArgumentOutOfRangeException(nameof(op));
        }
    }

    public override void Affine(Storage x, Storage y, int n, float alpha, float beta) => Run(new AffineKernel(D(x), D(y), alpha, beta), n);

    public override void Axpy(Storage x, Storage y, int n, float alpha) => Run(new AxpyKernel(D(x), D(y), alpha), n);

    public override void MulAdd(Storage a, Storage b, Storage c, int n) => Run(new MulAddKernel(D(a), D(b), D(c)), n);

    public override void AddRowVector(Storage a, Storage v, Storage c, int rows, int cols)
    {
        float[] av = D(a), vv = D(v), cv = D(c);
        var bias = vv.AsSpan(0, cols);
        for (int r = 0; r < rows; r++)
        {
            int o = r * cols;
            AddKernel.Apply(av.AsSpan(o, cols), bias, cv.AsSpan(o, cols));
        }
    }

    public override void SumRows(Storage x, Storage y, int rows, int cols)
    {
        float[] xv = D(x);
        var acc = D(y).AsSpan(0, cols);
        for (int r = 0; r < rows; r++)
        {
            AddKernel.Apply(acc, xv.AsSpan(r * cols, cols), acc);
        }
    }

    public override void Sum(Storage x, Storage result, int n, float scale)
    {
        float[] xv = D(x);
        double total;
        if (n < ParallelThreshold || !ComputeResources.AllowParallel)
        {
            total = SumSpan(xv.AsSpan(0, n));
        }
        else
        {
            int chunks = ChunkCount(n);
            int size = ChunkSize(n, chunks);
            var partials = new double[chunks];
            Parallel.For(0, chunks, ComputeResources.ParallelOptions, c =>
            {
                int start = c * size;
                partials[c] = SumSpan(xv.AsSpan(start, Math.Min(size, n - start)));
            });
            total = partials.Sum();
        }

        D(result)[0] = (float)(total * scale);
    }

    public override void AxpyAt(Storage x, Storage y, int offset, float alpha) => D(y)[offset] += alpha * D(x)[0];

    public override void AddBroadcastScalar(Storage s, Storage y, int n, float scale)
    {
        float value = D(s)[0] * scale;
        Run(new AffineKernel(D(y), D(y), 1f, value), n);
    }

    public override void MatMul(Storage a, Storage b, Storage c, int m, int n, int k, bool transA, bool transB, float beta) =>
        CpuMatMul.Multiply(D(a), D(b), D(c), m, n, k, transA, transB, beta);

    public override void SgdStep(Storage p, Storage g, Storage? v, int n, float lr, float momentum)
    {
        if (v is null)
        {
            Run(new AxpyKernel(D(g), D(p), -lr), n);
        }
        else
        {
            Run(new SgdMomentumKernel(D(p), D(g), D(v), lr, momentum), n);
        }
    }

    public override void AdamStep(Storage p, Storage g, Storage m, Storage v, int n, float lr, float beta1, float beta2, float eps) =>
        Run(new AdamKernel(D(p), D(g), D(m), D(v), lr, beta1, beta2, eps), n);

    public override void Dropout(Storage x, Storage y, int n, float p, uint seed) => Run(new DropoutKernel(D(x), D(y), p, seed, accumulate: false), n);

    public override void DropoutBackward(Storage dy, Storage dx, int n, float p, uint seed) => Run(new DropoutKernel(D(dy), D(dx), p, seed, accumulate: true), n);

    public override void Synchronize()
    {
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float[] D(Storage s) => ((CpuStorage)s).Data;

    private static double SumSpan(ReadOnlySpan<float> x)
    {
        var acc = Vector<float>.Zero;
        int i = 0;
        var xv = MemoryMarshal.Cast<float, Vector<float>>(x);
        foreach (var v in xv)
        {
            acc += v;
        }

        i = xv.Length * Vector<float>.Count;
        double total = Vector.Sum(acc);
        for (; i < x.Length; i++)
        {
            total += x[i];
        }

        return total;
    }

    private static int ChunkCount(int n) => Math.Max(1, Math.Min(ComputeResources.MaxCpuThreads * 2, n / (ParallelThreshold / 4)));

    private static int ChunkSize(int n, int chunks)
    {
        int size = (n + chunks - 1) / chunks;
        int w = Vector<float>.Count;
        return (size + w - 1) / w * w;
    }

    /// <summary>Runs a range kernel inline for small inputs and in parallel chunks for large ones.</summary>
    private static void Run<TKernel>(TKernel kernel, int n)
        where TKernel : struct, IRangeKernel
    {
        if (n < ParallelThreshold || !ComputeResources.AllowParallel)
        {
            kernel.Execute(0, n);
            return;
        }

        int chunks = ChunkCount(n);
        int size = ChunkSize(n, chunks);
        Parallel.For(0, chunks, ComputeResources.ParallelOptions, c =>
        {
            int start = c * size;
            kernel.Execute(start, Math.Min(start + size, n));
        });
    }

    private interface IRangeKernel
    {
        void Execute(int start, int end);
    }

    // Each kernel processes [start, end): a SIMD main loop over Vector<float> lanes, then a scalar tail.

    private readonly struct SigmoidKernel(float[] x, float[] y) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            var xs = x.AsSpan(start, end - start);
            var ys = y.AsSpan(start, end - start);
            var xv = MemoryMarshal.Cast<float, Vector<float>>(xs);
            var yv = MemoryMarshal.Cast<float, Vector<float>>(ys);
            var one = Vector<float>.One;
            for (int i = 0; i < xv.Length; i++)
            {
                yv[i] = one / (one + Vector.Exp(-xv[i]));
            }

            for (int i = xv.Length * Vector<float>.Count; i < xs.Length; i++)
            {
                ys[i] = 1f / (1f + MathF.Exp(-xs[i]));
            }
        }
    }

    private readonly struct TanhKernel(float[] x, float[] y) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            var xs = x.AsSpan(start, end - start);
            var ys = y.AsSpan(start, end - start);
            var xv = MemoryMarshal.Cast<float, Vector<float>>(xs);
            var yv = MemoryMarshal.Cast<float, Vector<float>>(ys);
            var one = Vector<float>.One;
            var two = new Vector<float>(2f);
            for (int i = 0; i < xv.Length; i++)
            {
                // tanh(x) = 1 - 2 / (e^(2x) + 1); saturates cleanly to +-1 when e^(2x) overflows or underflows.
                yv[i] = one - two / (Vector.Exp(xv[i] * two) + one);
            }

            for (int i = xv.Length * Vector<float>.Count; i < xs.Length; i++)
            {
                ys[i] = MathF.Tanh(xs[i]);
            }
        }
    }

    private readonly struct ReluKernel(float[] x, float[] y) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            var xs = x.AsSpan(start, end - start);
            var ys = y.AsSpan(start, end - start);
            var xv = MemoryMarshal.Cast<float, Vector<float>>(xs);
            var yv = MemoryMarshal.Cast<float, Vector<float>>(ys);
            for (int i = 0; i < xv.Length; i++)
            {
                yv[i] = Vector.Max(xv[i], Vector<float>.Zero);
            }

            for (int i = xv.Length * Vector<float>.Count; i < xs.Length; i++)
            {
                ys[i] = MathF.Max(xs[i], 0f);
            }
        }
    }

    private readonly struct SquareKernel(float[] x, float[] y) : IRangeKernel
    {
        public void Execute(int start, int end) => MulKernel.Apply(x.AsSpan(start, end - start), x.AsSpan(start, end - start), y.AsSpan(start, end - start));
    }

    private readonly struct AbsKernel(float[] x, float[] y) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            var xs = x.AsSpan(start, end - start);
            var ys = y.AsSpan(start, end - start);
            var xv = MemoryMarshal.Cast<float, Vector<float>>(xs);
            var yv = MemoryMarshal.Cast<float, Vector<float>>(ys);
            for (int i = 0; i < xv.Length; i++)
            {
                yv[i] = Vector.Abs(xv[i]);
            }

            for (int i = xv.Length * Vector<float>.Count; i < xs.Length; i++)
            {
                ys[i] = MathF.Abs(xs[i]);
            }
        }
    }

    private readonly struct DropoutKernel(float[] x, float[] y, float p, uint seed, bool accumulate) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            float scale = 1f / (1f - p);
            for (int i = start; i < end; i++)
            {
                float v = DropoutMask.Keep(seed, (uint)i, p) ? x[i] * scale : 0f;
                y[i] = accumulate ? y[i] + v : v;
            }
        }
    }

    private readonly struct AbsBackwardKernel(float[] x, float[] dy, float[] dx) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            var xs = x.AsSpan(start, end - start);
            var gs = dy.AsSpan(start, end - start);
            var ds = dx.AsSpan(start, end - start);
            var xv = MemoryMarshal.Cast<float, Vector<float>>(xs);
            var gv = MemoryMarshal.Cast<float, Vector<float>>(gs);
            var dv = MemoryMarshal.Cast<float, Vector<float>>(ds);
            for (int i = 0; i < xv.Length; i++)
            {
                var positive = Vector.ConditionalSelect(Vector.GreaterThan(xv[i], Vector<float>.Zero), gv[i], Vector<float>.Zero);
                var negative = Vector.ConditionalSelect(Vector.LessThan(xv[i], Vector<float>.Zero), gv[i], Vector<float>.Zero);
                dv[i] += positive - negative;
            }

            for (int i = xv.Length * Vector<float>.Count; i < xs.Length; i++)
            {
                ds[i] += xs[i] > 0f ? gs[i] : xs[i] < 0f ? -gs[i] : 0f;
            }
        }
    }

    private readonly struct SigmoidBackwardKernel(float[] y, float[] dy, float[] dx) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            var ys = y.AsSpan(start, end - start);
            var gs = dy.AsSpan(start, end - start);
            var ds = dx.AsSpan(start, end - start);
            var yv = MemoryMarshal.Cast<float, Vector<float>>(ys);
            var gv = MemoryMarshal.Cast<float, Vector<float>>(gs);
            var dv = MemoryMarshal.Cast<float, Vector<float>>(ds);
            var one = Vector<float>.One;
            for (int i = 0; i < yv.Length; i++)
            {
                dv[i] = Vector.FusedMultiplyAdd(gv[i], yv[i] * (one - yv[i]), dv[i]);
            }

            for (int i = yv.Length * Vector<float>.Count; i < ys.Length; i++)
            {
                ds[i] += gs[i] * ys[i] * (1f - ys[i]);
            }
        }
    }

    private readonly struct TanhBackwardKernel(float[] y, float[] dy, float[] dx) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            var ys = y.AsSpan(start, end - start);
            var gs = dy.AsSpan(start, end - start);
            var ds = dx.AsSpan(start, end - start);
            var yv = MemoryMarshal.Cast<float, Vector<float>>(ys);
            var gv = MemoryMarshal.Cast<float, Vector<float>>(gs);
            var dv = MemoryMarshal.Cast<float, Vector<float>>(ds);
            var one = Vector<float>.One;
            for (int i = 0; i < yv.Length; i++)
            {
                dv[i] = Vector.FusedMultiplyAdd(gv[i], one - yv[i] * yv[i], dv[i]);
            }

            for (int i = yv.Length * Vector<float>.Count; i < ys.Length; i++)
            {
                ds[i] += gs[i] * (1f - ys[i] * ys[i]);
            }
        }
    }

    private readonly struct ReluBackwardKernel(float[] x, float[] dy, float[] dx) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            var xs = x.AsSpan(start, end - start);
            var gs = dy.AsSpan(start, end - start);
            var ds = dx.AsSpan(start, end - start);
            var xv = MemoryMarshal.Cast<float, Vector<float>>(xs);
            var gv = MemoryMarshal.Cast<float, Vector<float>>(gs);
            var dv = MemoryMarshal.Cast<float, Vector<float>>(ds);
            for (int i = 0; i < xv.Length; i++)
            {
                dv[i] += Vector.ConditionalSelect(Vector.GreaterThan(xv[i], Vector<float>.Zero), gv[i], Vector<float>.Zero);
            }

            for (int i = xv.Length * Vector<float>.Count; i < xs.Length; i++)
            {
                if (xs[i] > 0f)
                {
                    ds[i] += gs[i];
                }
            }
        }
    }

    private readonly struct SquareBackwardKernel(float[] x, float[] dy, float[] dx) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            var xs = x.AsSpan(start, end - start);
            var gs = dy.AsSpan(start, end - start);
            var ds = dx.AsSpan(start, end - start);
            var xv = MemoryMarshal.Cast<float, Vector<float>>(xs);
            var gv = MemoryMarshal.Cast<float, Vector<float>>(gs);
            var dv = MemoryMarshal.Cast<float, Vector<float>>(ds);
            var two = new Vector<float>(2f);
            for (int i = 0; i < xv.Length; i++)
            {
                dv[i] = Vector.FusedMultiplyAdd(two * xv[i], gv[i], dv[i]);
            }

            for (int i = xv.Length * Vector<float>.Count; i < xs.Length; i++)
            {
                ds[i] += 2f * xs[i] * gs[i];
            }
        }
    }

    private readonly struct AddKernel(float[] a, float[] b, float[] c) : IRangeKernel
    {
        public void Execute(int start, int end) => Apply(a.AsSpan(start, end - start), b.AsSpan(start, end - start), c.AsSpan(start, end - start));

        public static void Apply(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c)
        {
            var av = MemoryMarshal.Cast<float, Vector<float>>(a);
            var bv = MemoryMarshal.Cast<float, Vector<float>>(b);
            var cv = MemoryMarshal.Cast<float, Vector<float>>(c);
            for (int i = 0; i < av.Length; i++)
            {
                cv[i] = av[i] + bv[i];
            }

            for (int i = av.Length * Vector<float>.Count; i < a.Length; i++)
            {
                c[i] = a[i] + b[i];
            }
        }
    }

    private readonly struct SubKernel(float[] a, float[] b, float[] c) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            var @as = a.AsSpan(start, end - start);
            var bs = b.AsSpan(start, end - start);
            var cs = c.AsSpan(start, end - start);
            var av = MemoryMarshal.Cast<float, Vector<float>>(@as);
            var bv = MemoryMarshal.Cast<float, Vector<float>>(bs);
            var cv = MemoryMarshal.Cast<float, Vector<float>>(cs);
            for (int i = 0; i < av.Length; i++)
            {
                cv[i] = av[i] - bv[i];
            }

            for (int i = av.Length * Vector<float>.Count; i < @as.Length; i++)
            {
                cs[i] = @as[i] - bs[i];
            }
        }
    }

    private readonly struct MulKernel(float[] a, float[] b, float[] c) : IRangeKernel
    {
        public void Execute(int start, int end) => Apply(a.AsSpan(start, end - start), b.AsSpan(start, end - start), c.AsSpan(start, end - start));

        public static void Apply(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c)
        {
            var av = MemoryMarshal.Cast<float, Vector<float>>(a);
            var bv = MemoryMarshal.Cast<float, Vector<float>>(b);
            var cv = MemoryMarshal.Cast<float, Vector<float>>(c);
            for (int i = 0; i < av.Length; i++)
            {
                cv[i] = av[i] * bv[i];
            }

            for (int i = av.Length * Vector<float>.Count; i < a.Length; i++)
            {
                c[i] = a[i] * b[i];
            }
        }
    }

    private readonly struct AffineKernel(float[] x, float[] y, float alpha, float beta) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            var xs = x.AsSpan(start, end - start);
            var ys = y.AsSpan(start, end - start);
            var xv = MemoryMarshal.Cast<float, Vector<float>>(xs);
            var yv = MemoryMarshal.Cast<float, Vector<float>>(ys);
            var va = new Vector<float>(alpha);
            var vb = new Vector<float>(beta);
            for (int i = 0; i < xv.Length; i++)
            {
                yv[i] = Vector.FusedMultiplyAdd(xv[i], va, vb);
            }

            for (int i = xv.Length * Vector<float>.Count; i < xs.Length; i++)
            {
                ys[i] = MathF.FusedMultiplyAdd(xs[i], alpha, beta);
            }
        }
    }

    private readonly struct AxpyKernel(float[] x, float[] y, float alpha) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            var xs = x.AsSpan(start, end - start);
            var ys = y.AsSpan(start, end - start);
            var xv = MemoryMarshal.Cast<float, Vector<float>>(xs);
            var yv = MemoryMarshal.Cast<float, Vector<float>>(ys);
            var va = new Vector<float>(alpha);
            for (int i = 0; i < xv.Length; i++)
            {
                yv[i] = Vector.FusedMultiplyAdd(xv[i], va, yv[i]);
            }

            for (int i = xv.Length * Vector<float>.Count; i < xs.Length; i++)
            {
                ys[i] = MathF.FusedMultiplyAdd(xs[i], alpha, ys[i]);
            }
        }
    }

    private readonly struct MulAddKernel(float[] a, float[] b, float[] c) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            var @as = a.AsSpan(start, end - start);
            var bs = b.AsSpan(start, end - start);
            var cs = c.AsSpan(start, end - start);
            var av = MemoryMarshal.Cast<float, Vector<float>>(@as);
            var bv = MemoryMarshal.Cast<float, Vector<float>>(bs);
            var cv = MemoryMarshal.Cast<float, Vector<float>>(cs);
            for (int i = 0; i < av.Length; i++)
            {
                cv[i] = Vector.FusedMultiplyAdd(av[i], bv[i], cv[i]);
            }

            for (int i = av.Length * Vector<float>.Count; i < @as.Length; i++)
            {
                cs[i] = MathF.FusedMultiplyAdd(@as[i], bs[i], cs[i]);
            }
        }
    }

    private readonly struct SgdMomentumKernel(float[] p, float[] g, float[] v, float lr, float momentum) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            var ps = p.AsSpan(start, end - start);
            var gs = g.AsSpan(start, end - start);
            var vs = v.AsSpan(start, end - start);
            var pv = MemoryMarshal.Cast<float, Vector<float>>(ps);
            var gv = MemoryMarshal.Cast<float, Vector<float>>(gs);
            var vv = MemoryMarshal.Cast<float, Vector<float>>(vs);
            var mu = new Vector<float>(momentum);
            var nlr = new Vector<float>(-lr);
            for (int i = 0; i < pv.Length; i++)
            {
                var vel = Vector.FusedMultiplyAdd(mu, vv[i], gv[i]);
                vv[i] = vel;
                pv[i] = Vector.FusedMultiplyAdd(nlr, vel, pv[i]);
            }

            for (int i = pv.Length * Vector<float>.Count; i < ps.Length; i++)
            {
                float vel = momentum * vs[i] + gs[i];
                vs[i] = vel;
                ps[i] -= lr * vel;
            }
        }
    }

    private readonly struct AdamKernel(float[] p, float[] g, float[] m, float[] v, float lr, float beta1, float beta2, float eps) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            var ps = p.AsSpan(start, end - start);
            var gs = g.AsSpan(start, end - start);
            var ms = m.AsSpan(start, end - start);
            var vs = v.AsSpan(start, end - start);
            var pv = MemoryMarshal.Cast<float, Vector<float>>(ps);
            var gv = MemoryMarshal.Cast<float, Vector<float>>(gs);
            var mv = MemoryMarshal.Cast<float, Vector<float>>(ms);
            var vv = MemoryMarshal.Cast<float, Vector<float>>(vs);
            var b1 = new Vector<float>(beta1);
            var b2 = new Vector<float>(beta2);
            var c1 = new Vector<float>(1f - beta1);
            var c2 = new Vector<float>(1f - beta2);
            var ve = new Vector<float>(eps);
            var vlr = new Vector<float>(lr);
            for (int i = 0; i < pv.Length; i++)
            {
                var grad = gv[i];
                var mom = Vector.FusedMultiplyAdd(b1, mv[i], c1 * grad);
                var vel = Vector.FusedMultiplyAdd(b2, vv[i], c2 * grad * grad);
                mv[i] = mom;
                vv[i] = vel;
                pv[i] -= vlr * mom / (Vector.SquareRoot(vel) + ve);
            }

            for (int i = pv.Length * Vector<float>.Count; i < ps.Length; i++)
            {
                float grad = gs[i];
                float mom = beta1 * ms[i] + (1f - beta1) * grad;
                float vel = beta2 * vs[i] + (1f - beta2) * grad * grad;
                ms[i] = mom;
                vs[i] = vel;
                ps[i] -= lr * mom / (MathF.Sqrt(vel) + eps);
            }
        }
    }
}

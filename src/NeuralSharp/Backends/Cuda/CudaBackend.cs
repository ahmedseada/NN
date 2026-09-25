using System.Runtime.InteropServices;
using System.Text;
using static NeuralSharp.Backends.Cuda.CudaDriver;

namespace NeuralSharp.Backends.Cuda;

internal sealed class CudaStorage(CudaBackend backend, ulong pointer, int length) : Storage(backend, length)
{
    public readonly ulong Pointer = pointer;
}

/// <summary>
/// CUDA implementation. Memory comes from a caching allocator (freed blocks are kept per size
/// and reused, so a training loop stops calling cuMemAlloc after its first iteration), and all
/// kernels are the PTX from <see cref="PtxKernels"/>, launched on the default stream.
/// </summary>
internal sealed unsafe partial class CudaBackend : Backend
{
    private static readonly Lazy<(int Count, string Reason)> Probe = new(ProbeDriver);
    private static readonly Lazy<CudaBackend>[] Instances = CreateInstances();

    [ThreadStatic]
    private static IntPtr t_currentContext;

    private readonly IntPtr _context;

    /// <summary>
    /// All work runs on this (blocking) stream rather than the legacy default stream, because only work on an
    /// explicit stream can be recorded into a CUDA Graph. Synchronous copies on the legacy stream still order
    /// correctly with it.
    /// </summary>
    private readonly IntPtr _stream;

    /// <summary>Non-null while a graph is being recorded: blocks freed during capture, owned by the graph.</summary>
    private Dictionary<int, Stack<ulong>>? _captureFree;
    private readonly Dictionary<int, Stack<ulong>> _pool = [];
    private readonly MemoryAccountant _memory;
    private readonly int _multiprocessors;

    private readonly IntPtr _fill, _affine, _axpy, _mulAdd, _add, _sub, _mul;
    private readonly IntPtr _sigmoid, _tanh, _relu, _square, _abs;
    private readonly IntPtr _sigmoidBwd, _tanhBwd, _reluBwd, _squareBwd, _absBwd, _dropout;
    private readonly IntPtr _addRowVec, _addScalar, _sumRows, _sum, _sgdMomentum, _adam, _matmul;

    private CudaBackend(int ordinal)
    {
        Check(cuDeviceGet(out int device, ordinal), nameof(cuDeviceGet));
        Check(cuDevicePrimaryCtxRetain(out _context, device), nameof(cuDevicePrimaryCtxRetain));
        MakeCurrent();
        Check(cuStreamCreate(out _stream, 0), nameof(cuStreamCreate));

        byte* name = stackalloc byte[256];
        Check(cuDeviceGetName(name, 256, device), nameof(cuDeviceGetName));
        Check(cuDeviceTotalMem(out nuint memory, device), nameof(cuDeviceTotalMem));
        Check(cuDeviceGetAttribute(out _multiprocessors, AttributeMultiprocessorCount, device), nameof(cuDeviceGetAttribute));
        Name = $"{Marshal.PtrToStringAnsi((IntPtr)name)} ({memory / (1024 * 1024)} MiB, {_multiprocessors} SMs)";
        _memory = new MemoryAccountant(() => ComputeResources.GpuMemoryLimit, $"cuda:{ordinal}");

        IntPtr module = LoadModule(PtxKernels.Source);
        IntPtr Fn(string kernel)
        {
            byte[] bytes = Encoding.ASCII.GetBytes(kernel + "\0");
            fixed (byte* p = bytes)
            {
                Check(cuModuleGetFunction(out IntPtr function, module, p), $"cuModuleGetFunction({kernel})");
                return function;
            }
        }

        _fill = Fn("fill_f32");
        _affine = Fn("affine_f32");
        _axpy = Fn("axpy_f32");
        _mulAdd = Fn("muladd_f32");
        _add = Fn("add_f32");
        _sub = Fn("sub_f32");
        _mul = Fn("mul_f32");
        _sigmoid = Fn("sigmoid_f32");
        _tanh = Fn("tanh_f32");
        _relu = Fn("relu_f32");
        _square = Fn("square_f32");
        _abs = Fn("abs_f32");
        _sigmoidBwd = Fn("sigmoid_bwd_f32");
        _tanhBwd = Fn("tanh_bwd_f32");
        _reluBwd = Fn("relu_bwd_f32");
        _squareBwd = Fn("square_bwd_f32");
        _absBwd = Fn("abs_bwd_f32");
        _dropout = Fn("dropout_f32");
        _addRowVec = Fn("add_rowvec_f32");
        _addScalar = Fn("add_scalar_f32");
        _sumRows = Fn("sum_rows_f32");
        _sum = Fn("sum_f32");
        _sgdMomentum = Fn("sgd_momentum_f32");
        _adam = Fn("adam_f32");
        _matmul = Fn("matmul_f32");
        _kernels = PtxKernels.AdvancedNames.Concat(PtxKernels.DecodingNames).ToDictionary(k => k, Fn);
    }

    public static int DeviceCount => Probe.Value.Count;

    public static string UnavailableReason => Probe.Value.Reason;

    public string Name { get; }

    public static CudaBackend Get(int ordinal) => Instances[ordinal].Value;

    public static bool IsInitialized(int ordinal) => Instances[ordinal].IsValueCreated;

    private static (int, string) ProbeDriver()
    {
        if (Environment.GetEnvironmentVariable("NEURALSHARP_DISABLE_CUDA") is "1" or "true")
        {
            return (0, "disabled by the NEURALSHARP_DISABLE_CUDA environment variable");
        }

        if (!TryLoad(out string reason))
        {
            return (0, reason);
        }

        try
        {
            int result = cuInit(0);
            if (result != 0)
            {
                return (0, $"cuInit failed with error {result} (no usable NVIDIA GPU?)");
            }

            Check(cuDeviceGetCount(out int count), nameof(cuDeviceGetCount));
            return count > 0 ? (count, "") : (0, "the NVIDIA driver reports no CUDA devices");
        }
        catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException or CudaException)
        {
            return (0, ex.Message);
        }
    }

    private static Lazy<CudaBackend>[] CreateInstances()
    {
        int count = DeviceCount;
        var instances = new Lazy<CudaBackend>[count];
        for (int i = 0; i < count; i++)
        {
            int ordinal = i;
            instances[i] = new Lazy<CudaBackend>(() => new CudaBackend(ordinal));
        }

        return instances;
    }

    private static IntPtr LoadModule(string ptx)
    {
        const int LogSize = 16 * 1024;
        byte[] image = Encoding.ASCII.GetBytes(ptx + "\0");
        byte* log = stackalloc byte[LogSize];
        log[0] = 0;
        int* options = stackalloc int[] { JitErrorLogBuffer, JitErrorLogBufferSizeBytes };
        void** values = stackalloc void*[] { log, (void*)LogSize };
        fixed (byte* p = image)
        {
            int result = cuModuleLoadDataEx(out IntPtr module, p, 2, options, values);
            if (result != 0)
            {
                string details = Marshal.PtrToStringAnsi((IntPtr)log) ?? "";
                throw new CudaException($"The CUDA driver could not JIT-compile the NeuralSharp kernels (error {result}). {details}");
            }

            return module;
        }
    }

    /// <summary>Binds this device's context to the calling thread (cheap no-op when already bound).</summary>
    private void MakeCurrent()
    {
        if (t_currentContext != _context)
        {
            Check(cuCtxSetCurrent(_context), nameof(cuCtxSetCurrent));
            t_currentContext = _context;
        }
    }

    public override Storage Allocate(int length, bool zeroed)
    {
        MakeCurrent();
        long bytes = BlockBytes(length);
        bool releaseCache = _memory.MustReleaseCacheFor(bytes); // throws when over the in-use limit
        ulong pointer = 0;
        lock (_pool)
        {
            if (_captureFree is not null && _captureFree.TryGetValue(length, out var captured) && captured.Count > 0)
            {
                // Reuse a block freed earlier in this capture: stream order keeps the recorded uses apart.
                pointer = captured.Pop();
                _memory.Reused(bytes);
            }
            else if (_pool.TryGetValue(length, out var bucket) && bucket.Count > 0)
            {
                pointer = bucket.Pop();
                _memory.Reused(bytes);
            }
        }

        if (pointer == 0)
        {
            if (releaseCache)
            {
                ReleaseCachedMemory();
            }

            pointer = AllocateDevice(length);
            _memory.Allocated(bytes);
        }

        if (zeroed && length > 0)
        {
            Check(cuMemsetD32Async(pointer, 0, (nuint)length, _stream), nameof(cuMemsetD32Async));
        }

        return new CudaStorage(this, pointer, length);
    }

    private ulong AllocateDevice(int length)
    {
        nuint bytes = (nuint)BlockBytes(length);
        int result = cuMemAlloc(out ulong pointer, bytes);
        if (result == ErrorOutOfMemory)
        {
            // Give back cached blocks and anything held only by unreachable tensors, then retry once.
            GC.Collect();
            GC.WaitForPendingFinalizers();
            ReleaseCachedMemory();
            result = cuMemAlloc(out pointer, bytes);
        }

        Check(result, nameof(cuMemAlloc));
        return pointer;
    }

    private static long BlockBytes(int length) => (long)Math.Max(length, 1) * sizeof(float);

    public override MemoryUsage GetMemoryUsage() => _memory.Usage;

    /// <summary>Frees every cached (currently unused) device block back to the driver.</summary>
    public override void ReleaseCachedMemory()
    {
        MakeCurrent();
        lock (_pool)
        {
            foreach (var (length, bucket) in _pool)
            {
                while (bucket.Count > 0)
                {
                    Check(cuMemFree(bucket.Pop()), nameof(cuMemFree));
                    _memory.Freed(BlockBytes(length));
                }
            }
        }
    }

    // Called when the last reference is released, possibly from the finalizer thread, so it only
    // touches the pool and never the driver.
    public override void Return(Storage storage)
    {
        var s = (CudaStorage)storage;
        lock (_pool)
        {
            // While recording a graph, freed blocks belong to the graph: returning them to the shared pool would let
            // unrelated tensors reuse memory the graph writes on every replay.
            var target = _captureFree ?? _pool;
            if (!target.TryGetValue(s.Length, out var bucket))
            {
                target[s.Length] = bucket = new Stack<ulong>();
            }

            bucket.Push(s.Pointer);
            _memory.Returned(BlockBytes(s.Length));
        }
    }

    public override void Upload(ReadOnlySpan<float> source, Storage destination)
    {
        MakeCurrent();
        fixed (float* p = source)
        {
            Check(cuMemcpyHtoD(P(destination), p, (nuint)source.Length * sizeof(float)), nameof(cuMemcpyHtoD));
        }
    }

    public override void Download(Storage source, Span<float> destination) => DownloadRange(source, 0, destination);

    public override void DownloadRange(Storage source, int offset, Span<float> destination)
    {
        MakeCurrent();
        fixed (float* p = destination)
        {
            Check(cuMemcpyDtoH(p, P(source) + (ulong)offset * sizeof(float), (nuint)destination.Length * sizeof(float)), nameof(cuMemcpyDtoH));
        }
    }

    public override void Fill(Storage y, int n, float value) => Launch1D(_fill, n, P(y), F(value), U(n));

    public override void Copy(Storage x, Storage y, int n)
    {
        MakeCurrent();
        Check(cuMemcpyDtoDAsync(P(y), P(x), (nuint)n * sizeof(float), _stream), nameof(cuMemcpyDtoDAsync));
    }

    public override void Unary(UnaryOp op, Storage x, Storage y, int n)
    {
        IntPtr fn = op switch
        {
            UnaryOp.Sigmoid => _sigmoid,
            UnaryOp.Tanh => _tanh,
            UnaryOp.Relu => _relu,
            UnaryOp.Square => _square,
            UnaryOp.Abs => _abs,
            UnaryOp.Exp => _kernels["exp_f32"],
            UnaryOp.Log => _kernels["log_f32"],
            UnaryOp.Gelu => _kernels["gelu_f32"],
            _ => throw new ArgumentOutOfRangeException(nameof(op)),
        };
        Launch1D(fn, n, P(x), P(y), U(n));
    }

    public override void UnaryBackward(UnaryOp op, Storage x, Storage y, Storage dy, Storage dx, int n)
    {
        IntPtr fn = op switch
        {
            UnaryOp.Sigmoid => _sigmoidBwd,
            UnaryOp.Tanh => _tanhBwd,
            UnaryOp.Relu => _reluBwd,
            UnaryOp.Square => _squareBwd,
            UnaryOp.Abs => _absBwd,
            UnaryOp.Exp => _kernels["exp_bwd_f32"],
            UnaryOp.Log => _kernels["log_bwd_f32"],
            UnaryOp.Gelu => _kernels["gelu_bwd_f32"],
            _ => throw new ArgumentOutOfRangeException(nameof(op)),
        };
        Launch1D(fn, n, P(x), P(y), P(dy), P(dx), U(n));
    }

    public override void Binary(BinaryOp op, Storage a, Storage b, Storage c, int n)
    {
        IntPtr fn = op switch
        {
            BinaryOp.Add => _add,
            BinaryOp.Sub => _sub,
            BinaryOp.Mul => _mul,
            _ => throw new ArgumentOutOfRangeException(nameof(op)),
        };
        Launch1D(fn, n, P(a), P(b), P(c), U(n));
    }

    public override void Affine(Storage x, Storage y, int n, float alpha, float beta) => Launch1D(_affine, n, P(x), P(y), F(alpha), F(beta), U(n));

    public override void Axpy(Storage x, Storage y, int n, float alpha) => Launch1D(_axpy, n, P(x), P(y), F(alpha), U(n));

    public override void MulAdd(Storage a, Storage b, Storage c, int n) => Launch1D(_mulAdd, n, P(a), P(b), P(c), U(n));

    public override void AddRowVector(Storage a, Storage v, Storage c, int rows, int cols)
    {
        int n = rows * cols;
        Launch1D(_addRowVec, n, P(a), P(v), P(c), U(cols), U(n));
    }

    public override void SumRows(Storage x, Storage y, int rows, int cols) => Launch1D(_sumRows, cols, P(x), P(y), U(rows), U(cols));

    public override void Sum(Storage x, Storage result, int n, float scale)
    {
        MakeCurrent();
        Check(cuMemsetD32Async(P(result), 0, 1, _stream), nameof(cuMemsetD32Async));
        if (n == 0)
        {
            return;
        }

        // Enough blocks to fill the GPU; the grid-stride loop covers the rest.
        uint blocks = (uint)Math.Min((n + PtxKernels.BlockSize - 1) / PtxKernels.BlockSize, Math.Max(1, _multiprocessors) * 8);
        Launch(_sum, blocks, 1, PtxKernels.BlockSize, 1, P(x), P(result), U(n), F(scale));
    }

    public override void AxpyAt(Storage x, Storage y, int offset, float alpha) =>
        Launch1D(_addScalar, 1, P(x), P(y) + (ulong)offset * sizeof(float), F(alpha), U(1));

    public override void AddBroadcastScalar(Storage s, Storage y, int n, float scale) => Launch1D(_addScalar, n, P(s), P(y), F(scale), U(n));

    public override void BatchedMatMul(Storage a, Storage b, Storage c, int batch, int m, int n, int k, bool transA, bool transB, float beta)
    {
        if (m == 0 || n == 0 || batch == 0)
        {
            return;
        }

        const int T = PtxKernels.Tile;
        const int MaxGridZ = 65535;
        ulong mk = (ulong)m * (ulong)k, kn = (ulong)k * (ulong)n, mn = (ulong)m * (ulong)n;
        for (int first = 0; first < batch; first += MaxGridZ)
        {
            int count = Math.Min(MaxGridZ, batch - first);
            ulong offset = (ulong)first * sizeof(float);
            Launch(_matmul, (uint)((n + T - 1) / T), (uint)((m + T - 1) / T), (uint)count, T, T,
                P(a) + offset * mk, P(b) + offset * kn, P(c) + offset * mn, U(m), U(n), U(k), U(transA ? 1 : 0), U(transB ? 1 : 0), F(beta),
                mk, kn, mn);
        }
    }

    public override void SgdStep(Storage p, Storage g, Storage? v, int n, float lr, float momentum)
    {
        if (v is null)
        {
            Axpy(g, p, n, -lr);
        }
        else
        {
            Launch1D(_sgdMomentum, n, P(p), P(g), P(v), F(-lr), F(momentum), U(n));
        }
    }

    public override void AdamStep(Storage p, Storage g, Storage m, Storage v, int n, float lr, float beta1, float beta2, float eps) =>
        Launch1D(_adam, n, P(p), P(g), P(m), P(v), F(lr), F(beta1), F(beta2), F(1f - beta1), F(1f - beta2), F(eps), U(n));

    public override void Dropout(Storage x, Storage y, int n, float p, uint seed) =>
        Launch1D(_dropout, n, P(x), P(y), F(p), F(1f / (1f - p)), seed, 0UL, U(n));

    public override void DropoutBackward(Storage dy, Storage dx, int n, float p, uint seed) =>
        Launch1D(_dropout, n, P(dy), P(dx), F(p), F(1f / (1f - p)), seed, 1UL, U(n));

    public override void Synchronize()
    {
        MakeCurrent();
        Check(cuCtxSynchronize(), nameof(cuCtxSynchronize));
    }

    private void Launch1D(IntPtr function, int n, params ReadOnlySpan<ulong> args)
    {
        if (n > 0)
        {
            Launch(function, (uint)((n + PtxKernels.BlockSize - 1) / PtxKernels.BlockSize), 1, PtxKernels.BlockSize, 1, args);
        }
    }

    /// <summary>
    /// Launches a kernel. Every argument is widened to a 64-bit slot; the driver reads each
    /// parameter's actual size from the slot's start, which on little-endian hosts is the value itself.
    /// </summary>
    private void Launch(IntPtr function, uint gridX, uint gridY, uint blockX, uint blockY, params ReadOnlySpan<ulong> args) =>
        Launch(function, gridX, gridY, 1, blockX, blockY, args);

    private void Launch(IntPtr function, uint gridX, uint gridY, uint gridZ, uint blockX, uint blockY, params ReadOnlySpan<ulong> args)
    {
        MakeCurrent();
        ulong* values = stackalloc ulong[args.Length];
        void** pointers = stackalloc void*[args.Length];
        for (int i = 0; i < args.Length; i++)
        {
            values[i] = args[i];
            pointers[i] = &values[i];
        }

        Check(cuLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY, 1, 0, _stream, pointers, null), nameof(cuLaunchKernel));
    }

    private static ulong P(Storage s) => ((CudaStorage)s).Pointer;

    private static ulong F(float value) => BitConverter.SingleToUInt32Bits(value);

    private static ulong U(int value) => (uint)value;
}

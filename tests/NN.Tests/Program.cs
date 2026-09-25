// Self-contained test runner (no test framework dependency).
//   dotnet run --project tests/NN.Tests                  run every test on every available device
//   dotnet run --project tests/NN.Tests -- --dump-ptx f  write the generated CUDA kernels to f

using System.Diagnostics;
using NN;
using NN.Backends.Cuda;
using NN.Layers;
using NN.Optimizers;

if (args is ["--dump-ptx", var ptxPath])
{
    File.WriteAllText(ptxPath, PtxKernels.Source);
    Console.WriteLine($"Wrote {PtxKernels.Source.Length} characters of PTX to {ptxPath}");
    return 0;
}

var devices = new List<Device> { Device.Cpu };
if (Device.IsCudaAvailable)
{
    devices.Add(Device.Cuda());
}
else
{
    Console.WriteLine($"CUDA not available ({CudaBackend.UnavailableReason}); testing the CPU only.");
}

int failed = 0, passed = 0;
foreach (var device in devices)
{
    Console.WriteLine($"== {device}: {device.Name}");
    foreach (var (name, test) in Tests.All)
    {
        var sw = Stopwatch.StartNew();
        try
        {
            using (new TensorScope())
            {
                test(device);
            }

            passed++;
            Console.WriteLine($"  PASS {name} ({sw.ElapsedMilliseconds} ms)");
        }
        catch (Exception ex)
        {
            failed++;
            Console.WriteLine($"  FAIL {name}: {ex.Message}");
        }
    }
}

Console.WriteLine($"{passed} passed, {failed} failed");
return failed == 0 ? 0 : 1;

internal static class Tests
{
    public static readonly (string Name, Action<Device> Run)[] All =
    [
        ("matmul matches reference (all transposes, beta 0 and 1)", MatMulReference),
        ("element-wise ops match reference", ElementWiseReference),
        ("large tensors (parallel / multi-block paths)", LargeTensors),
        ("gradient: sigmoid, sum", d => GradCheck(d, [3, 4], x => x.Sigmoid().Sum())),
        ("gradient: tanh, mean", d => GradCheck(d, [5, 3], x => x.Tanh().Mean())),
        ("gradient: relu, square", d => GradCheck(d, [4, 4], x => (x.Relu() + x.Square()).Sum(), avoidZero: true)),
        ("gradient: mul, sub, scalar ops", d => GradCheck(d, [2, 6], x => ((x * x) - (2f * x) + 1f - (x / 4f)).Mean() * 3f)),
        ("gradient: matmul left and right", GradMatMul),
        ("gradient: bias broadcast, reshape", GradBias),
        ("gradient: mean squared error", GradMse),
        ("backward accumulates and ZeroGrad resets", Accumulation),
        ("optimizers minimise a quadratic", Optimizers),
        ("save and load round-trip", SaveLoad),
        ("tensor scope releases intermediates", Scope),
    ];

    private static void MatMulReference(Device device)
    {
        var random = new Random(1);
        foreach (var (m, n, k) in new[] { (1, 1, 1), (3, 5, 2), (17, 33, 9), (64, 48, 70), (130, 67, 91) })
        {
            foreach (bool ta in new[] { false, true })
            {
                foreach (bool tb in new[] { false, true })
                {
                    foreach (float beta in new[] { 0f, 1f })
                    {
                        var a = RandomArray(random, m * k);
                        var b = RandomArray(random, k * n);
                        var c = RandomArray(random, m * n);
                        var expected = new float[m * n];
                        for (int i = 0; i < m; i++)
                        {
                            for (int j = 0; j < n; j++)
                            {
                                double acc = 0;
                                for (int p = 0; p < k; p++)
                                {
                                    acc += (double)a[ta ? p * m + i : i * k + p] * b[tb ? j * k + p : p * n + j];
                                }

                                expected[i * n + j] = (float)(acc + beta * c[i * n + j]);
                            }
                        }

                        var backend = device.Backend;
                        using var ta_ = Tensor.From(a, device);
                        using var tb_ = Tensor.From(b, device);
                        using var tc = Tensor.From(c, device);
                        backend.MatMul(ta_.Storage, tb_.Storage, tc.Storage, m, n, k, ta, tb, beta);
                        AssertClose(expected, tc.ToArray(), 1e-4f, $"matmul m={m} n={n} k={k} transA={ta} transB={tb} beta={beta}");
                    }
                }
            }
        }
    }

    private static void ElementWiseReference(Device device)
    {
        var random = new Random(2);
        var x = RandomArray(random, 37, 4f);
        var y = RandomArray(random, 37, 4f);
        using var tx = Tensor.From(x, device);
        using var ty = Tensor.From(y, device);
        AssertClose(x.Select(v => 1f / (1f + MathF.Exp(-v))).ToArray(), tx.Sigmoid().ToArray(), 1e-5f, "sigmoid");
        AssertClose(x.Select(MathF.Tanh).ToArray(), tx.Tanh().ToArray(), 1e-5f, "tanh");
        AssertClose(x.Select(v => MathF.Max(v, 0)).ToArray(), tx.Relu().ToArray(), 0, "relu");
        AssertClose(x.Select(v => v * v).ToArray(), tx.Square().ToArray(), 1e-6f, "square");
        AssertClose(x.Zip(y, (a, b) => a + b).ToArray(), (tx + ty).ToArray(), 1e-6f, "add");
        AssertClose(x.Zip(y, (a, b) => a - b).ToArray(), (tx - ty).ToArray(), 1e-6f, "sub");
        AssertClose(x.Zip(y, (a, b) => a * b).ToArray(), (tx * ty).ToArray(), 1e-6f, "mul");
        AssertClose(x.Select(v => 3f * v - 2f).ToArray(), (3f * tx - 2f).ToArray(), 1e-5f, "affine");
        AssertClose([x.Sum()], [tx.Sum().Item()], 1e-4f, "sum");
        AssertClose([x.Average()], [tx.Mean().Item()], 1e-5f, "mean");
        using var saturated = Tensor.From([-100f, 100f, -1000f, 1000f], device);
        AssertClose([0f, 1f, 0f, 1f], saturated.Sigmoid().ToArray(), 1e-6f, "sigmoid saturation");
        AssertClose([-1f, 1f, -1f, 1f], saturated.Tanh().ToArray(), 1e-6f, "tanh saturation");
    }

    private static void LargeTensors(Device device)
    {
        const int N = 1_000_003;
        var random = new Random(3);
        var x = RandomArray(random, N);
        using var tx = Tensor.From(x, device);
        double expectedSum = x.Sum(v => (double)v);
        AssertClose([(float)expectedSum], [tx.Sum().Item()], 1e-2f, "large sum");
        var sig = tx.Sigmoid().ToArray();
        for (int i = 0; i < N; i += 9973)
        {
            AssertClose([1f / (1f + MathF.Exp(-x[i]))], [sig[i]], 1e-5f, $"large sigmoid[{i}]");
        }

        using var matrix = Tensor.From(x.AsSpan(0, 1000 * 1000), [1000, 1000], device);
        using var ones = Tensor.Ones([1000], device);
        var rowSums = matrix.MatMul(ones.Reshape(1000, 1)).ToArray();
        for (int r = 0; r < 1000; r += 97)
        {
            double expected = 0;
            for (int c = 0; c < 1000; c++)
            {
                expected += x[r * 1000 + c];
            }

            AssertClose([(float)expected], [rowSums[r]], 1e-3f, $"large matmul row {r}");
        }
    }

    private static void GradMatMul(Device device)
    {
        var random = new Random(4);
        using var w = Tensor.From(RandomArray(random, 4 * 3), [4, 3], device);
        using var v = Tensor.From(RandomArray(random, 5 * 4), [5, 4], device);
        GradCheck(device, [5, 4], x => x.MatMul(w).Tanh().Sum());
        GradCheck(device, [4, 3], x => v.MatMul(x).Sigmoid().Mean());
    }

    private static void GradBias(Device device)
    {
        var random = new Random(5);
        using var input = Tensor.From(RandomArray(random, 6 * 3), [6, 3], device);
        GradCheck(device, [3], b => (input + b).Tanh().Sum());
        GradCheck(device, [6, 3], x => x.Reshape(2, 9).Square().Sum());
    }

    private static void GradMse(Device device)
    {
        var random = new Random(6);
        using var target = Tensor.From(RandomArray(random, 8), [4, 2], device);
        GradCheck(device, [4, 2], x => Losses.MeanSquaredError(x.Sigmoid(), target));
    }

    private static void Accumulation(Device device)
    {
        using var x = Tensor.From([1f, 2f, 3f], device, requiresGrad: true);
        (x * x).Sum().Backward();
        (x * x).Sum().Backward();
        AssertClose([4f, 8f, 12f], x.Grad!.ToArray(), 1e-6f, "accumulated gradient");
        x.ZeroGrad();
        AssertClose([0f, 0f, 0f], x.Grad!.ToArray(), 0, "zeroed gradient");
        using (Autograd.NoGrad())
        {
            Check(!(x * x).RequiresGrad, "NoGrad should stop recording");
        }
    }

    private static void Optimizers(Device device)
    {
        foreach (var make in new Func<IEnumerable<Tensor>, Optimizer>[]
        {
            p => new Sgd(p, 0.1f),
            p => new Sgd(p, 0.05f, momentum: 0.9f),
            p => new Adam(p, 0.1f),
        })
        {
            using var x = Tensor.From([5f, -3f, 2f, 7f, -1f, 4f, 0.5f, -6f, 9f], device, requiresGrad: true);
            using var optimizer = make([x]);
            for (int i = 0; i < 300; i++)
            {
                using var scope = new TensorScope();
                optimizer.ZeroGrad();
                (x - 1f).Square().Sum().Backward();
                optimizer.Step();
            }

            AssertClose(Enumerable.Repeat(1f, 9).ToArray(), x.ToArray(), 1e-2f, optimizer.GetType().Name);
        }
    }

    private static void SaveLoad(Device device)
    {
        string path = Path.GetTempFileName();
        try
        {
            using var a = new Sequential { new Linear(3, 5, device: device), new Tanh(), new Linear(5, 2, device: device) };
            using var b = new Sequential { new Linear(3, 5, device: device), new Tanh(), new Linear(5, 2, device: device) };
            a.Save(path);
            b.Load(path);
            foreach (var (pa, pb) in a.Parameters().Zip(b.Parameters()))
            {
                AssertClose(pa.ToArray(), pb.ToArray(), 0, "loaded parameter");
            }
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static void Scope(Device device)
    {
        Tensor inner;
        using (var scope = new TensorScope())
        {
            inner = Tensor.Ones([4], device) * 2f;
            var kept = scope.Keep(Tensor.Ones([4], device));
            kept.Dispose();
        }

        try
        {
            inner.ToArray();
            throw new Exception("tensor created in a disposed scope is still usable");
        }
        catch (ObjectDisposedException)
        {
        }
    }

    /// <summary>Compares autograd's gradient of f at a random point with central finite differences.</summary>
    private static void GradCheck(Device device, int[] shape, Func<Tensor, Tensor> f, bool avoidZero = false)
    {
        var random = new Random(shape.Sum() * 31 + shape.Length);
        var x0 = RandomArray(random, shape.Aggregate(1, (a, b) => a * b));
        if (avoidZero)
        {
            for (int i = 0; i < x0.Length; i++)
            {
                if (MathF.Abs(x0[i]) < 0.1f)
                {
                    x0[i] += 0.3f;
                }
            }
        }

        using var x = Tensor.From(x0, shape, device, requiresGrad: true);
        f(x).Backward();
        var analytic = x.Grad!.ToArray();

        const float h = 1e-2f;
        var numeric = new float[x0.Length];
        using (Autograd.NoGrad())
        {
            for (int i = 0; i < x0.Length; i++)
            {
                var plus = (float[])x0.Clone();
                var minus = (float[])x0.Clone();
                plus[i] += h;
                minus[i] -= h;
                using var tp = Tensor.From(plus, shape, device);
                using var tm = Tensor.From(minus, shape, device);
                numeric[i] = (f(tp).Item() - f(tm).Item()) / (2 * h);
            }
        }

        AssertClose(numeric, analytic, 2e-2f, "gradient");
    }

    private static float[] RandomArray(Random random, int n, float scale = 1f)
    {
        var values = new float[n];
        for (int i = 0; i < n; i++)
        {
            values[i] = (random.NextSingle() * 2f - 1f) * scale;
        }

        return values;
    }

    private static void AssertClose(float[] expected, float[] actual, float tolerance, string what)
    {
        Check(expected.Length == actual.Length, $"{what}: length {actual.Length}, expected {expected.Length}");
        for (int i = 0; i < expected.Length; i++)
        {
            float allowed = tolerance * Math.Max(1f, MathF.Abs(expected[i]));
            if (!(MathF.Abs(expected[i] - actual[i]) <= allowed))
            {
                throw new Exception($"{what}: element {i} is {actual[i]}, expected {expected[i]}");
            }
        }
    }

    private static void Check(bool condition, string message)
    {
        if (!condition)
        {
            throw new Exception(message);
        }
    }
}

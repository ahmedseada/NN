using NeuralSharp;
using NeuralSharp.Backends.Cuda;
using NeuralSharp.Layers;
using NeuralSharp.Optimizers;

internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] MixedPrecisionGroup =
    [
        ("mixed precision: bfloat16 tensor-core products (every transpose, edges, batches, beta) match bfloat16-rounded references", TensorCoreProducts),
        ("mixed precision: tensor-core flash attention (forward, log-sum-exp, dq, dk, dv; head sizes 64 and 128) matches float32", TensorCoreAttention),
        ("optimizer: 8-bit AdamW (dynamic code map, nearest codes, tracks 32-bit AdamW, CPU parity)", EightBitAdam),
        ("decoder: GPT-2 style (learned positions, LayerNorm, GELU, dropout): checkpointed blocks replay the dropout masks", GptStyleDecoder),
    ];

    private static void TensorCoreAttention(Device device)
    {
        bool tensorCores = MixedPrecision.TensorCoresUnavailable(device) is null;
        var random = new Random(8);
        float[] Values(int n) => [.. Enumerable.Range(0, n).Select(_ => random.NextSingle() * 2 - 1)];
        static double RelativeError(float[] expected, float[] actual) =>
            Math.Sqrt(expected.Zip(actual).Sum(p => (double)(p.First - p.Second) * (p.First - p.Second)) / Math.Max(1e-12, expected.Sum(v => (double)v * v)));

        foreach (int dim in new[] { 64, 128 })
        {
            foreach (var (rows, steps, capacity, offset, backward) in new[] { (100, 100, 100, 0, true), (130, 65, 65, 0, true), (70, 70, 150, 40, false), (192, 192, 192, 0, true) })
            {
                const int Heads = 3;
                float scale = 1f / MathF.Sqrt(dim);
                float[] q = Values(Heads * rows * dim), k = Values(Heads * capacity * dim), v = Values(Heads * capacity * dim);
                float[] dOut = Values(Heads * rows * dim), dq0 = Values(Heads * rows * dim), dk0 = Values(Heads * capacity * dim), dv0 = Values(Heads * capacity * dim);
                (float[] Y, float[] Lse, float[] Dq, float[] Dk, float[] Dv) Run(Device on)
                {
                    var backend = on.Backend;
                    using var tq = Tensor.From(q, [q.Length], on);
                    using var tk = Tensor.From(k, [k.Length], on);
                    using var tv = Tensor.From(v, [v.Length], on);
                    using var position = Tensor.From([offset], [1], on);
                    using var y = Tensor.Zeros([q.Length], on);
                    using var lse = Tensor.Zeros([Heads * rows], on);
                    using var tdo = Tensor.From(dOut, [dOut.Length], on);
                    using var dq = Tensor.From(dq0, [dq0.Length], on);
                    using var dk = Tensor.From(dk0, [dk0.Length], on);
                    using var dv = Tensor.From(dv0, [dv0.Length], on);
                    using (MixedPrecision.BFloat16())
                    {
                        backend.AttentionTiled(tq.Storage, tk.Storage, tv.Storage, position.Storage, y.Storage, lse.Storage, Heads, rows, steps, capacity, dim, scale);
                        if (backward)
                        {
                            backend.AttentionTiledBackward(tq.Storage, tk.Storage, tv.Storage, y.Storage, lse.Storage, tdo.Storage, dq.Storage, dk.Storage, dv.Storage,
                                Heads, rows, steps, capacity, dim, scale);
                        }
                    }

                    return (y.ToArray(), lse.ToArray(), dq.ToArray(), dk.ToArray(), dv.ToArray());
                }

                var expected = Run(Device.Cpu);
                var got = Run(device);
                string what = $"d {dim}, rows {rows}, steps {steps}, capacity {capacity}, offset {offset}";
                double tolerance = tensorCores ? 1.5e-2 : 1e-4;
                Check(RelativeError(expected.Y, got.Y) < tolerance, $"{what}: output error {RelativeError(expected.Y, got.Y):G3}");
                Check(RelativeError(expected.Lse, got.Lse) < tolerance, $"{what}: log-sum-exp error {RelativeError(expected.Lse, got.Lse):G3}");
                if (backward)
                {
                    // Relative to the gradients alone (the tensors started from random values they are added to).
                    float[] Minus(float[] a, float[] b) => [.. a.Zip(b).Select(p => p.First - p.Second)];
                    foreach (var (name, e, g, start) in new[] { ("dq", expected.Dq, got.Dq, dq0), ("dk", expected.Dk, got.Dk, dk0), ("dv", expected.Dv, got.Dv, dv0) })
                    {
                        double error = RelativeError(Minus(e, start), Minus(g, start));
                        Check(error < 2 * tolerance, $"{what}: {name} error {error:G3}");
                    }
                }
            }
        }
    }

    private static void EightBitAdam(Device device)
    {
        var signedMap = AdamW8Bit.DynamicMap(signed: true);
        var unsignedMap = AdamW8Bit.DynamicMap(signed: false);
        Check(signedMap.Length == 256 && unsignedMap.Length == 256, "256 codes");
        Check(signedMap.Zip(signedMap.Skip(1)).All(p => p.First <= p.Second) && signedMap.Contains(0f) && signedMap[^1] == 1f && signedMap[0] > -1f,
            "signed map sorted in [-1, 1] with 0 and 1");
        Check(unsignedMap[0] == 0f && unsignedMap[^1] == 1f && unsignedMap.Distinct().Count() == 256, "unsigned map: 256 distinct codes in [0, 1]");
        var random = new Random(5);
        for (int i = 0; i < 2000; i++)
        {
            float x = random.NextSingle() * 2 - 1;
            var map = i % 2 == 0 ? signedMap : unsignedMap;
            x = i % 2 == 0 ? x : MathF.Abs(x);
            int got = AdamW8Bit.Nearest(map, x);
            float best = map.Min(c => MathF.Abs(c - x));
            Check(MathF.Abs(map[got] - x) <= best, $"nearest code to {x}: {map[got]}, best distance {best}");
        }

        const int N = 20000;
        float[] start = [.. Enumerable.Range(0, N).Select(_ => random.NextSingle() * 2 - 1)];
        float[] target = [.. Enumerable.Range(0, N).Select(_ => random.NextSingle() * 2 - 1)];
        (float Loss, float[] Values) Run(Device on, bool eightBit)
        {
            using var p = Tensor.Persistent(start, [N], on, requiresGrad: true);
            using var t = Tensor.From(target, [N], on);
            using Optimizer optimizer = eightBit ? new AdamW8Bit([p], 0.01f, 0.9f, 0.95f, weightDecay: 0.1f) : new AdamW([p], 0.01f, 0.9f, 0.95f, weightDecay: 0.1f);
            float loss = 0;
            for (int step = 0; step < 100; step++)
            {
                using var scope = new TensorScope();
                optimizer.ZeroGrad();
                var l = (p - t).Square().Sum() * 0.5f;
                l.Backward();
                optimizer.ClipGradientNorm(50f);                     // active early on (the norm starts near 115)
                optimizer.Step();
                loss = l.Item();
            }

            return (loss, p.ToArray());
        }

        var full = Run(device, eightBit: false);
        var quantized = Run(device, eightBit: true);
        Check(quantized.Loss < 0.2f * N / 6 && Math.Abs(quantized.Loss - full.Loss) < 0.05f * full.Loss + 1f,
            $"8-bit AdamW loss {quantized.Loss:F2}, 32-bit {full.Loss:F2}");
        if (device.Type != DeviceType.Cpu)
        {
            var cpu = Run(Device.Cpu, eightBit: true);
            AssertClose(cpu.Values, quantized.Values, 2e-3f, "8-bit AdamW on the CPU and the device");
        }
    }

    private static void GptStyleDecoder(Device device)
    {
        var spec = new DecoderSpec
        {
            Vocabulary = 23, Dim = 16, Layers = 2, Heads = 2, KvHeads = 2, HeadDim = 8, FfDim = 64, MaxPositions = 12,
            Norm = DecoderNorm.Layer, NormEpsilon = 1e-5f, Gated = false, Activation = FeedForwardActivation.Gelu, Rope = null,
            FeedForwardBias = true, TieEmbeddings = true, LearnedPositions = true, Dropout = 0.2f,
        };
        Check(DecoderSpec.FromJson(spec.ToJson()) == spec, "JSON round trip");
        float[] ids = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7];
        (float Loss, float[][] Grads) Run(bool checkpointed)
        {
            using var model = spec.Build(options: new DecoderBuildOptions { Device = device, Seed = 7, InitStd = 0.02f });
            model.Train();
            using var scope = new TensorScope();
            var hidden = Tensor.From(ids, [2, 7], device);
            foreach (var module in model)
            {
                hidden = checkpointed && module is DecoderBlock block ? block.ForwardCheckpointed(hidden) : module.Forward(hidden);
            }

            var weights = Tensor.From([.. Enumerable.Range(0, hidden.Size).Select(i => MathF.Cos(i))], hidden.Shape, device);
            var loss = (hidden * weights).Sum();
            loss.Backward();
            var position = model.OfType<PositionEmbedding>().Single().Weight.Grad!.ToArray();
            Check(position.Take(7 * spec.Dim).Any(v => v != 0) && position.Skip(7 * spec.Dim).All(v => v == 0), "positions used get gradients, the rest none");
            return (loss.Item(), [.. model.Parameters().Select(p => p.Grad!.ToArray())]);
        }

        var plain = Run(false);
        var checkpointed = Run(true);
        AssertClose([plain.Loss], [checkpointed.Loss], 1e-4f, "loss");
        for (int i = 0; i < plain.Grads.Length; i++)
        {
            AssertClose(plain.Grads[i], checkpointed.Grads[i], 1e-3f, $"gradient {i}");
        }

        using var eval = spec.Build(options: new DecoderBuildOptions { Device = device, Seed = 7, InitStd = 0.02f });
        eval.Eval();
        using var x = Tensor.From(ids, [2, 7], device);
        using var first = eval.Forward(x);
        using var second = eval.Forward(x);
        AssertClose(first.ToArray(), second.ToArray(), 0f, "no dropout in evaluation");
    }

    private static float RoundBFloat16(float x)
    {
        uint bits = BitConverter.SingleToUInt32Bits(x);
        bits += 0x7FFFu + ((bits >> 16) & 1u);
        return BitConverter.UInt32BitsToSingle(bits & 0xFFFF0000u);
    }

    private static void TensorCoreProducts(Device device)
    {
        Check(PtxKernels.TensorCoreNames.Where(k => k.StartsWith("gemm")).All(k => PtxKernels.TensorCoreParameterCounts.TryGetValue(k, out int n) && n == 10)
              && PtxKernels.TensorCoreNames.All(PtxKernels.TensorCoreParameterCounts.ContainsKey), "tensor-core kernel signatures");
        var random = new Random(3);
        bool tensorCores = device.Type == DeviceType.Cuda && ((CudaBackend)device.Backend) is var backend && backend.TensorCoreUnavailableReason is null;
        long launchesBefore = device.Type == DeviceType.Cuda ? ((CudaBackend)device.Backend).TensorCoreLaunches : 0;
        bool anyDifferent = false;
        foreach (var (m, n, k, batch) in new[] { (64, 64, 32, 1), (200, 130, 71, 2), (257, 300, 129, 1), (128, 256, 512, 1) })
        {
            foreach (bool ta in new[] { false, true })
            {
                foreach (bool tb in new[] { false, true })
                {
                    foreach (float beta in new[] { 0f, 0.5f })
                    {
                        float[] a = [.. Enumerable.Range(0, batch * m * k).Select(_ => random.NextSingle() * 2 - 1)];
                        float[] b = [.. Enumerable.Range(0, batch * k * n).Select(_ => random.NextSingle() * 2 - 1)];
                        float[] c = [.. Enumerable.Range(0, batch * m * n).Select(_ => random.NextSingle() * 2 - 1)];
                        using var ta_ = Tensor.From(a, [a.Length], device);
                        using var tb_ = Tensor.From(b, [b.Length], device);
                        using var tc_ = Tensor.From(c, [c.Length], device);
                        using (MixedPrecision.BFloat16())
                        {
                            device.Backend.BatchedMatMul(ta_.Storage, tb_.Storage, tc_.Storage, batch, m, n, k, ta, tb, beta);
                        }

                        var got = tc_.ToArray();
                        var rounded = new double[m * n * batch];
                        var exact = new double[m * n * batch];
                        for (int s = 0; s < batch; s++)
                        {
                            for (int i = 0; i < m; i++)
                            {
                                for (int j = 0; j < n; j++)
                                {
                                    double sum = 0, sumExact = 0;
                                    for (int q = 0; q < k; q++)
                                    {
                                        float x = a[s * m * k + (ta ? q * m + i : i * k + q)], y = b[s * k * n + (tb ? j * k + q : q * n + j)];
                                        sum += (double)RoundBFloat16(x) * RoundBFloat16(y);
                                        sumExact += (double)x * y;
                                    }

                                    int o = s * m * n + i * n + j;
                                    rounded[o] = sum + beta * c[o];
                                    exact[o] = sumExact + beta * c[o];
                                }
                            }
                        }

                        string what = $"{m}×{n}×{k} batch {batch}, {(ta ? "t" : "n")}{(tb ? "t" : "n")}, beta {beta}";
                        var reference = tensorCores ? rounded : exact;
                        AssertClose([.. reference.Select(v => (float)v)], got, 1e-3f, what);
                        anyDifferent |= got.Zip(exact).Any(p => Math.Abs(p.First - p.Second) > 1e-3 * Math.Max(1, Math.Abs(p.Second)));
                    }
                }
            }
        }

        if (tensorCores)
        {
            Check(((CudaBackend)device.Backend).TensorCoreLaunches > launchesBefore && anyDifferent, "the products ran on the tensor cores in bfloat16");
        }
        else if (device.Type == DeviceType.Cuda)
        {
            Console.WriteLine($"    (no tensor cores: {((CudaBackend)device.Backend).TensorCoreUnavailableReason}; float32 checked)");
        }
    }
}

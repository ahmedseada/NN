using NeuralSharp;
using NeuralSharp.Data;
using NeuralSharp.Generation;
using NeuralSharp.Layers;
using NeuralSharp.Onnx;
using NeuralSharp.Optimizers;
using NeuralSharp.Training;

// Int8 weight-only quantization and half-precision weight files.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] Quantization =
    [
        ("int8: quantize/dequantize per column; direct and dequantized products match; input gradient", Int8Kernels),
        ("int8: quantized GPT: cached decoding matches the full pass; small logit change; ONNX export dequantizes", Int8Gpt),
        ("int8: save/load (a float model loads a quantized file), move between devices, dequantize, memory", Int8SaveLoad),
        ("int8: QLoRA — adapters train on top of frozen int8 weights", Int8Lora),
        ("weights: Float16 and BFloat16 files are half the size and round as expected", HalfPrecisionFiles),
    ];

    private static void Int8Kernels(Device device)
    {
        const int K = 37, N = 23;                                              // N not a multiple of 4: padded rows
        var r = new Random(40);
        var values = Enumerable.Range(0, K * N).Select(_ => (float)(r.NextDouble() * 2 - 1) * (1 + r.Next(3))).ToArray();
        using var weight = Tensor.From(values, [K, N], device);
        using var q = Int8Weight.Quantize(weight);
        using var dequantized = q.Dequantize();
        var deq = dequantized.ToArray();
        for (int j = 0; j < N; j++)
        {
            float scale = Enumerable.Range(0, K).Max(i => MathF.Abs(values[i * N + j])) / 127f;
            for (int i = 0; i < K; i++)
            {
                float expected = MathF.Round(values[i * N + j] / scale) * scale;
                Check(MathF.Abs(deq[i * N + j] - expected) < 1e-5f, $"dequantized [{i}, {j}]");
                Check(MathF.Abs(deq[i * N + j] - values[i * N + j]) <= scale / 2 + 1e-6f, $"rounding error [{i}, {j}]");
            }
        }

        foreach (int m in new[] { 1, 3, 8, 20 })                           // ≤ 8 rows: direct int8 kernel; 20: dequantize + matmul
        {
            var input = Enumerable.Range(0, m * K).Select(_ => (float)(r.NextDouble() * 2 - 1)).ToArray();
            using var x = Tensor.From(input, [m, K], device, requiresGrad: true);
            using var reference = Tensor.From(input, [m, K], device, requiresGrad: true);
            var y = x.MatMulInt8(q);
            var want = reference.MatMul(dequantized);
            AssertClose(want.ToArray(), y.ToArray(), 1e-4f, $"int8 product, {m} rows");
            y.Sum().Backward();
            want.Sum().Backward();
            AssertClose(reference.Grad!.ToArray(), x.Grad!.ToArray(), 1e-4f, $"input gradient, {m} rows");
        }

        using var batched = Tensor.From([.. Enumerable.Range(0, 2 * 3 * K).Select(i => MathF.Sin(i))], [2, 3, K], device);
        using var flatInput = batched.Reshape(6, K);
        AssertClose(flatInput.MatMul(dequantized).ToArray(), batched.MatMulInt8(q).ToArray(), 1e-4f, "[batch, time, k] input");
    }

    private static void Int8Gpt(Device device)
    {
        const int T = 10, V = 11;
        using var model = TinyGpt(device);
        var random = new Random(41);
        var ids = Enumerable.Range(0, T).Select(_ => (float)random.Next(V)).ToArray();
        using var sequence = Tensor.From(ids, [1, T], device);
        float[] original;
        using (Autograd.NoGrad())
        {
            original = model.Forward(sequence).ToArray();
        }

        int quantized = model.QuantizeInt8();
        Check(quantized == model.Descendants().OfType<Linear>().Count() && quantized > 0, $"{quantized} layers quantized");
        using (Autograd.NoGrad())
        {
            var full = model.Forward(sequence).ToArray();                  // 10 rows per product: dequantize + matmul
            float change = full.Zip(original, (a, b) => MathF.Abs(a - b)).Max(), range = original.Max(MathF.Abs);
            Check(change < 0.05f * range && change > 0, $"logit change {change} (logit range {range})");

            using var context = new DecodingContext(device, 1, 12);        // one row per product: the direct int8 kernel
            using var prompt = Tensor.From(ids.AsSpan(0, 4), [1, 4], device);
            AssertClose(full[..(4 * V)], model.ForwardCached(prompt, context).ToArray(), 1e-4f, "prefill");
            for (int t = 4; t < T; t++)
            {
                using var next = Tensor.From(ids.AsSpan(t, 1), [1, 1], device);
                AssertClose(full[(t * V)..((t + 1) * V)], model.ForwardCached(next, context).ToArray(), 1e-4f, $"cached step {t}");
            }

            using var exported = OnnxImport.Load(OnnxExport.For(model).Input(T).ToBytes(), device);
            AssertClose(full, exported.Model.Forward(sequence).ToArray(), 1e-4f, "ONNX export of an int8 model (dequantized weights)");
        }
    }

    private static void Int8SaveLoad(Device device)
    {
        var builder = Network.Input(64).Seed(42).Linear(256).ReLU().Linear(256).ReLU().Linear(10);
        using var model = builder.OnDevice(device).Build();
        using var x = Tensor.From([.. Enumerable.Range(0, 5 * 64).Select(i => MathF.Cos(i * 0.1f))], [5, 64], device);
        long floatBytes = model.Parameters().Sum(p => 4L * p.Size);
        model.QuantizeInt8();
        using var expected = model.Predict(x);
        long int8Bytes = model.Parameters().Concat(model.Buffers()).Sum(p => 4L * p.Size);
        Check(int8Bytes < floatBytes * 0.3, $"memory {int8Bytes:N0} bytes vs {floatBytes:N0} as float32");
        Check(model.Parameters().All(p => p.Rank == 1), "only biases remain trainable parameters");
        Check(model.ToString()!.Length > 0 && model.OfType<Linear>().All(l => l.ToString().Contains("int8")), "layers show int8");

        string path = Path.Combine(Path.GetTempPath(), $"ns-{Guid.NewGuid():N}.nsw");
        string floatPath = Path.ChangeExtension(path, ".f32.nsw");
        try
        {
            model.Save(path);
            using (var reference = builder.OnDevice(device).Build())
            {
                reference.Save(floatPath);
            }

            Check(new FileInfo(path).Length < new FileInfo(floatPath).Length * 0.3, $"file {new FileInfo(path).Length:N0} vs {new FileInfo(floatPath).Length:N0} bytes");
            using var loaded = builder.OnDevice(device).Build();                  // a float model: the file makes it int8
            loaded.Load(path);
            Check(loaded.OfType<Linear>().All(l => l.Int8 is not null), "loading quantizes the layers listed in the file");
            AssertClose(expected.ToArray(), loaded.Predict(x).ToArray(), 1e-6f, "reloaded int8 model");
        }
        finally
        {
            File.Delete(path);
            File.Delete(floatPath);
        }

        using var moved = builder.OnDevice(Device.Cpu).Build();
        moved.QuantizeInt8();
        using var cpuResult = moved.Predict(x.To(Device.Cpu));
        moved.To(device);
        AssertClose(cpuResult.ToArray(), moved.Predict(x).ToArray(), 1e-4f, "int8 model moved to the device");

        var deq = model.OfType<Linear>().First().Int8!.Dequantize();
        model.DequantizeInt8();
        Check(model.OfType<Linear>().All(l => l.Int8 is null), "dequantized");
        AssertClose(deq.ToArray(), model.OfType<Linear>().First().Weight.ToArray(), 0f, "dequantized weights are the int8 values");
        deq.Dispose();
        AssertClose(expected.ToArray(), model.Predict(x).ToArray(), 1e-4f, "same outputs after dequantizing");
        Check(model.Parameters().Any(p => p.Rank == 2 && p.RequiresGrad), "weights are trainable again");
    }

    private static void Int8Lora(Device device)
    {
        var random = new Random(43);
        var features = new float[64, 8];
        var targets = new float[64, 1];
        for (int i = 0; i < 64; i++)
        {
            for (int j = 0; j < 8; j++)
            {
                features[i, j] = (float)(random.NextDouble() * 2 - 1);
            }

            targets[i, 0] = features[i, 0] * 2 - features[i, 3] + 0.5f;
        }

        using var model = Network.Input(8).OnDevice(device).Seed(44).Linear(32).Tanh().Linear(1).Build();
        model.QuantizeInt8();
        var before = model.OfType<Linear>().Select(l => l.Int8!.Dequantize()).ToList();
        int adapters = model.AddLora(rank: 4, alpha: 8, targets: _ => true, freezeBase: true, random: new Random(45));
        Check(adapters == 2, $"{adapters} adapters");
        var data = Dataset.FromArrays(features, targets);
        using var trainer = new Trainer(model, Losses.MeanSquaredError, p => new Adam(p, 0.02f));
        var history = trainer.Fit(new DataLoader(data, 16, shuffle: true, device: device, seed: 46), 40);
        Check(history.Epochs[^1].Loss < history.Epochs[0].Loss * 0.5, $"loss {history.Epochs[0].Loss:F4} → {history.Epochs[^1].Loss:F4}");
        foreach (var (linear, weight) in model.OfType<Linear>().Zip(before))
        {
            using var after = linear.Int8!.Dequantize();
            AssertClose(weight.ToArray(), after.ToArray(), 0f, "int8 weights unchanged by training");
            weight.Dispose();
        }
    }

    private static void HalfPrecisionFiles(Device device)
    {
        var builder = Network.Input(32).Seed(47).Linear(128).ReLU().Linear(32);
        using var model = builder.OnDevice(device).Build();
        var original = model.Parameters().SelectMany(p => p.ToArray()).ToArray();
        string dir = Path.Combine(Path.GetTempPath(), $"ns-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            var sizes = new Dictionary<WeightFormat, long>();
            foreach (var format in new[] { WeightFormat.Float32, WeightFormat.Float16, WeightFormat.BFloat16 })
            {
                string path = Path.Combine(dir, $"{format}.nsw");
                model.Save(path, format);
                sizes[format] = new FileInfo(path).Length;
                using var loaded = builder.OnDevice(device).Build();
                loaded.Load(path);
                var values = loaded.Parameters().SelectMany(p => p.ToArray()).ToArray();
                float relative = format switch { WeightFormat.Float32 => 0f, WeightFormat.Float16 => 1f / 1024, _ => 1f / 128 };
                for (int i = 0; i < values.Length; i++)
                {
                    Check(MathF.Abs(values[i] - original[i]) <= relative * MathF.Abs(original[i]) + 1e-7f, $"{format} value {i}: {values[i]} vs {original[i]}");
                }
            }

            Check(sizes[WeightFormat.Float16] < sizes[WeightFormat.Float32] * 0.52 && sizes[WeightFormat.BFloat16] == sizes[WeightFormat.Float16],
                $"sizes {string.Join(", ", sizes)}");

            // A quantized model saved as Float16 keeps its int8 bytes exact.
            model.QuantizeInt8();
            using var expected = model.Predict(Tensor.Ones([2, 32], device));
            string mixed = Path.Combine(dir, "int8-f16.nsw");
            model.Save(mixed, WeightFormat.Float16);
            using var reloaded = builder.OnDevice(device).Build();
            reloaded.Load(mixed);
            AssertClose(expected.ToArray(), reloaded.Predict(Tensor.Ones([2, 32], device)).ToArray(), 1e-2f, "int8 + float16 biases");
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }
}

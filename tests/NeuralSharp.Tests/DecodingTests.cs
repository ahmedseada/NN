using NeuralSharp;
using NeuralSharp.Layers;

// Tests for fused inference kernels, KV-cache decoding, on-device sampling and compute graphs.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] Decoding =
    [
        ("fused kernels match their unfused equivalents", FusedKernels),
        ("KV-cache decoding matches the full forward pass at every position", CachedDecodingMatchesFull),
        ("batched cached decoding matches sequences decoded one by one", BatchedDecoding),
        ("graph replay gives the same tokens as direct execution", GraphReplay),
        ("sampler: distribution, top-k, temperature, determinism, CPU parity", SamplerBehaviour),
    ];

    private static Sequential TinyGpt(Device device, int vocabulary = 11, int dim = 16, int context = 12)
    {
        var random = new Random(50);
        var model = new Sequential
        {
            new Embedding(vocabulary, dim, device, random),
            new PositionalEncoding(context, dim, device),
            new TransformerEncoderLayer(dim, 4, ffDim: 32, dropout: 0f, causal: true, device: device, random: random),
            new TransformerEncoderLayer(dim, 4, ffDim: 32, dropout: 0f, causal: true, device: device, random: random),
            new LayerNorm(dim, device: device),
            new Linear(dim, vocabulary, device: device, random: random),
        };
        model.Eval();
        return model;
    }

    private static void FusedKernels(Device device)
    {
        var random = new Random(51);
        using var x = Tensor.From(RandomArray(random, 3 * 4 * 6, 3f), [3, 4, 6], device);
        using var mask = Tensor.From(RandomArray(random, 4 * 6, 2f), [4, 6], device);
        float[] reference, fused;
        using (Autograd.NoGrad())
        {
            fused = x.ScaleMaskSoftmax(0.7f, mask).ToArray();
        }

        reference = (x * 0.7f + mask).Softmax().ToArray();
        AssertClose(reference, fused, 1e-5f, "scale + mask + softmax");

        var norm = new LayerNorm(6, device: device);
        norm.Gamma.Load(RandomArray(random, 6, 2f));
        norm.Beta.Load(RandomArray(random, 6));
        reference = norm.Forward(x).ToArray();                  // autograd on: the three-kernel path
        using (Autograd.NoGrad())
        {
            fused = norm.Forward(x).ToArray();                  // inference: the fused kernel
        }

        AssertClose(reference, fused, 1e-4f, "fused LayerNorm");

        using var bias = Tensor.From(RandomArray(random, 6), device);
        reference = (x + bias).Gelu().ToArray();
        using (Autograd.NoGrad())
        {
            fused = x.BiasGelu(bias).ToArray();
        }

        AssertClose(reference, fused, 1e-5f, "bias + GELU");
    }

    private static void CachedDecodingMatchesFull(Device device)
    {
        const int T = 10, V = 11;
        using var model = TinyGpt(device);
        var ids = new float[T];
        var random = new Random(52);
        for (int t = 0; t < T; t++)
        {
            ids[t] = random.Next(V);
        }

        using (Autograd.NoGrad())
        {
            using var sequence = Tensor.From(ids, [1, T], device);
            var full = model.Forward(sequence).ToArray();                                   // [1, T, V]

            using var context = new DecodingContext(device, 1, 12);
            using var prompt = Tensor.From(ids.AsSpan(0, 4), [1, 4], device);
            var prefill = model.ForwardCached(prompt, context).ToArray();                   // positions 0-3 at once
            AssertClose(full[..(4 * V)], prefill, 1e-4f, "prefill logits");
            for (int t = 4; t < T; t++)
            {
                using var next = Tensor.From(ids.AsSpan(t, 1), [1, 1], device);
                var step = model.ForwardCached(next, context).ToArray();                    // one new position
                AssertClose(full[(t * V)..((t + 1) * V)], step, 1e-4f, $"cached step {t}");
            }

            Check(context.Length == T, "context length");
        }
    }

    private static void BatchedDecoding(Device device)
    {
        const int B = 3, T = 7, V = 11;
        using var model = TinyGpt(device);
        var random = new Random(53);
        var ids = new float[B * T];
        for (int i = 0; i < ids.Length; i++)
        {
            ids[i] = random.Next(V);
        }

        using (Autograd.NoGrad())
        {
            using var batchContext = new DecodingContext(device, B, 12);
            var batched = new List<float[]>();
            using (var prompt = Tensor.From([.. Enumerable.Range(0, B).SelectMany(b => ids.Skip(b * T).Take(2))], [B, 2], device))
            {
                model.ForwardCached(prompt, batchContext);
            }

            for (int t = 2; t < T; t++)
            {
                using var next = Tensor.From([.. Enumerable.Range(0, B).Select(b => ids[b * T + t])], [B, 1], device);
                batched.Add(model.ForwardCached(next, batchContext).ToArray());
            }

            for (int b = 0; b < B; b++)
            {
                using var context = new DecodingContext(device, 1, 12);
                using (var prompt = Tensor.From(ids.AsSpan(b * T, 2), [1, 2], device))
                {
                    model.ForwardCached(prompt, context);
                }

                for (int t = 2; t < T; t++)
                {
                    using var next = Tensor.From(ids.AsSpan(b * T + t, 1), [1, 1], device);
                    var single = model.ForwardCached(next, context).ToArray();
                    AssertClose(single, batched[t - 2][(b * V)..((b + 1) * V)], 1e-4f, $"sequence {b} step {t}");
                }
            }
        }
    }

    private static void GraphReplay(Device device)
    {
        const int B = 2, V = 11, Steps = 8;
        using var model = TinyGpt(device);
        int[][] Run(bool useGraph)
        {
            using var context = new DecodingContext(device, B, 12);
            using var sampler = new TokenSampler(device, B, V, Steps + 1) { Temperature = 0.9f, Seed = 7 };
            using (Autograd.NoGrad())
            {
                using (var scope = new TensorScope())
                {
                    sampler.Sample(model.ForwardCached(Tensor.From([1f, 2f, 3f, 4f], [B, 2], device), context));
                }

                void Step() => sampler.Sample(model.ForwardCached(sampler.Ids.Reshape(B, 1), context));
                using var graph = useGraph ? context.CaptureStep(Step) : null;
                for (int s = 0; s < Steps; s++)
                {
                    if (graph is not null)
                    {
                        context.ReplayStep(graph);
                    }
                    else
                    {
                        using var scope = new TensorScope();
                        Step();
                    }
                }

                if (graph is not null && device.Type == DeviceType.Cuda)
                {
                    Check(graph.IsRecorded, $"CUDA graph recording failed: {graph.FailureReason}");
                }
            }

            return [.. sampler.Read(0, Steps + 1).Select(step => step.Select(t => t.Id).ToArray())];
        }

        var direct = Run(useGraph: false);
        var replayed = Run(useGraph: true);
        for (int s = 0; s < direct.Length; s++)
        {
            Check(direct[s].SequenceEqual(replayed[s]), $"step {s}: direct [{string.Join(",", direct[s])}] vs graph [{string.Join(",", replayed[s])}]");
        }
    }

    private static void SamplerBehaviour(Device device)
    {
        const int Rows = 2000, V = 5;
        float[] logits = [2f, 1f, 0f, -1f, 0.5f];
        var all = new float[Rows * V];
        for (int r = 0; r < Rows; r++)
        {
            logits.CopyTo(all, r * V);
        }

        using var x = Tensor.From(all, [Rows, V], device);
        float[] Frequencies(TokenSampler sampler)
        {
            sampler.Sample(x);
            var counts = new float[V];
            foreach (var t in sampler.Read(0, 1)[0])
            {
                counts[t.Id]++;
            }

            return [.. counts.Select(c => c / Rows)];
        }

        using var plain = new TokenSampler(device, Rows, V, 1) { Temperature = 1f, Seed = 3 };
        var expected = logits.Select(MathF.Exp).ToArray();
        float sum = expected.Sum();
        var observed = Frequencies(plain);
        for (int j = 0; j < V; j++)
        {
            Check(MathF.Abs(observed[j] - expected[j] / sum) < 0.035f, $"token {j}: frequency {observed[j]:F3}, probability {expected[j] / sum:F3}");
        }

        var stats = plain.Read(0, 1)[0][0];
        AssertClose([expected[stats.Id] / sum], [stats.Probability], 1e-4f, "reported probability");
        double entropy = -expected.Sum(e => e / sum * Math.Log2(e / sum));
        AssertClose([(float)entropy], [stats.Entropy], 1e-3f, "reported entropy");
        Check(stats.Alternatives[0].Id == 0 && stats.Alternatives[1].Id == 1 && stats.Alternatives[2].Id == 4, "top alternatives in order");

        using var topTwo = new TokenSampler(device, Rows, V, 1) { TopK = 2, Seed = 3 };
        var restricted = Frequencies(topTwo);
        Check(restricted[2] == 0 && restricted[3] == 0 && restricted[4] == 0, "top-k must exclude all but the two best tokens");
        Check(MathF.Abs(restricted[0] - MathF.E / (MathF.E + 1)) < 0.035f, "top-k renormalizes");

        using var cold = new TokenSampler(device, Rows, V, 1) { Temperature = 0.05f, Seed = 3 };
        Check(Frequencies(cold)[0] == 1f, "low temperature is greedy");

        using var again = new TokenSampler(device, Rows, V, 1) { Temperature = 1f, Seed = 3 };
        Check(Frequencies(again).SequenceEqual(observed), "same seed, same samples");

        if (device.Type != DeviceType.Cpu)
        {
            using var cpuX = Tensor.From(all, [Rows, V], Device.Cpu);
            using var cpu = new TokenSampler(Device.Cpu, Rows, V, 1) { Temperature = 1f, Seed = 3 };
            cpu.Sample(cpuX);
            var cpuIds = cpu.Read(0, 1)[0].Select(t => t.Id).ToArray();
            var deviceIds = plain.Read(0, 1)[0].Select(t => t.Id).ToArray();
            int differences = cpuIds.Zip(deviceIds).Count(p => p.First != p.Second);
            Check(differences <= Rows / 200, $"{differences} of {Rows} samples differ between CPU and {device} (rounding only)");
        }
    }
}

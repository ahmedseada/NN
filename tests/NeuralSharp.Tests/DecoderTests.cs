using NeuralSharp;
using NeuralSharp.Generation;
using NeuralSharp.Inference;
using NeuralSharp.Layers;
using NeuralSharp.Optimizers;

// Decoder-only language models described by DecoderSpec, checked against a plain-loop reference implementation.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] Decoder =
    [
        ("decoder: RMSNorm and rotary embedding gradients (finite differences)", DecoderGradients),
        ("decoder: tiled and decoding attention match a direct computation (offsets, groups, odd head sizes)", AttentionKernels),
        ("decoder: DecoderSpec variants match a plain reference implementation (GQA, q/k norm, biases, rope, tied, post-norms, parallel)", DecoderMatchesReference),
        ("decoder: cached decoding (float32 and int8 KV) and int8 weights match the full pass; generation", DecoderCachedAndInt8),
        ("decoder: LoRA by layer name trains; JSON and package round trip", DecoderLoraAndPackage),
    ];

    private static void DecoderGradients(Device device)
    {
        GradCheck(device, [3, 6], x => (x.RmsNormalize(1e-6f) * x).Sum(), tolerance: 3e-2f);
        var r = new Random(60);
        var freq = Enumerable.Range(0, 3).Select(i => 1.0 / Math.Pow(100, 2.0 * i / 6)).ToArray();
        var cos = new float[8 * 3];
        var sin = new float[8 * 3];
        for (int p = 0; p < 8; p++)
        {
            for (int i = 0; i < 3; i++)
            {
                cos[p * 3 + i] = (float)Math.Cos(p * freq[i]);
                sin[p * 3 + i] = (float)Math.Sin(p * freq[i]);
            }
        }

        using var tc = Tensor.From(cos, [8, 3], device);
        using var ts = Tensor.From(sin, [8, 3], device);
        using var positions = Tensor.From([2f, 5f, 7f], [3], device);
        using var weights = Tensor.From([.. Enumerable.Range(0, 2 * 3 * 2 * 8).Select(_ => (float)r.NextDouble())], [2, 3, 2, 8], device);
        foreach (bool interleaved in new[] { false, true })
        {
            GradCheck(device, [2, 3, 2, 8], x => (x.Rope(tc, ts, positions, 3, interleaved) * weights).Sum());
        }

        // Every dimension rotated (no pass-through copy).
        var cos4 = new float[8 * 4];
        var sin4 = new float[8 * 4];
        for (int p = 0; p < 8; p++)
        {
            for (int i = 0; i < 4; i++)
            {
                cos4[p * 4 + i] = (float)Math.Cos(p * (i + 1) * 0.3);
                sin4[p * 4 + i] = (float)Math.Sin(p * (i + 1) * 0.3);
            }
        }

        using var tc4 = Tensor.From(cos4, [8, 4], device);
        using var ts4 = Tensor.From(sin4, [8, 4], device);
        GradCheck(device, [2, 3, 2, 8], x => (x.Rope(tc4, ts4, positions, 4, false) * weights).Sum());

        // Fused act(gate) · up: gradients through both inputs, and the same values as the separate operations.
        using var other = Tensor.From([.. Enumerable.Range(0, 15).Select(_ => (float)(r.NextDouble() * 2 - 1))], [3, 5], device);
        using var mix = Tensor.From([.. Enumerable.Range(0, 15).Select(_ => (float)r.NextDouble())], [3, 5], device);
        for (int kind = 0; kind < 3; kind++)
        {
            int k = kind;
            GradCheck(device, [3, 5], x => (Tensor.GatedActivation(x, other, k) * mix).Sum(), avoidZero: k == 2);
            GradCheck(device, [3, 5], x => (Tensor.GatedActivation(other, x, k) * mix).Sum());
            using var g = Tensor.From([.. Enumerable.Range(0, 15).Select(i => (i - 7) * 0.4f)], [3, 5], device);
            var separate = (k switch { 0 => g * g.Sigmoid(), 1 => g.Gelu(), _ => g.Relu() }) * other;
            AssertClose(separate.ToArray(), Tensor.GatedActivation(g, other, k).ToArray(), 1e-5f, $"gated activation {k}");
        }

        // Fused RMS normalization with a gain (and an offset, as Gemma stores it).
        using var gain = Tensor.From([.. Enumerable.Range(0, 6).Select(i => 0.5f + i * 0.1f)], [6], device);
        using var rows = Tensor.From([.. Enumerable.Range(0, 18).Select(i => MathF.Sin(i * 0.7f))], [3, 6], device);
        foreach (float offset in new[] { 0f, 1f })
        {
            var normalized = rows.RmsNormalize(1e-6f).ToArray();
            var g = gain.ToArray();
            var expected = normalized.Select((v, i) => v * (g[i % 6] + offset)).ToArray();
            AssertClose(expected, rows.RmsNormAffine(gain, 1e-6f, offset).ToArray(), 1e-5f, $"fused RMS norm, offset {offset}");
        }
    }

    private static void AttentionKernels(Device device)
    {
        var r = new Random(61);
        foreach (var (heads, rowsPerHead, steps, capacity, dim, offset) in new[] { (3, 80, 40, 96, 100, 7), (2, 64, 64, 64, 128, 0), (4, 6, 3, 50, 32, 20), (2, 33, 11, 60, 6, 5) })
        {
            float scale = 1f / MathF.Sqrt(dim);
            var q = Enumerable.Range(0, heads * rowsPerHead * dim).Select(_ => (float)(r.NextDouble() * 2 - 1)).ToArray();
            var k = Enumerable.Range(0, heads * capacity * dim).Select(_ => (float)(r.NextDouble() * 2 - 1)).ToArray();
            var v = Enumerable.Range(0, heads * capacity * dim).Select(_ => (float)(r.NextDouble() * 2 - 1)).ToArray();
            var expected = new float[heads * rowsPerHead * dim];
            for (int h = 0; h < heads; h++)
            {
                for (int i = 0; i < rowsPerHead; i++)
                {
                    int count = Math.Min(offset + i % steps, capacity - 1) + 1;
                    var scores = new double[count];
                    for (int c = 0; c < count; c++)
                    {
                        double dot = 0;
                        for (int d = 0; d < dim; d++)
                        {
                            dot += q[(h * rowsPerHead + i) * dim + d] * k[(h * capacity + c) * dim + d];
                        }

                        scores[c] = dot * scale;
                    }

                    double max = scores.Max(), sum = scores.Sum(x => Math.Exp(x - max));
                    for (int d = 0; d < dim; d++)
                    {
                        double value = 0;
                        for (int c = 0; c < count; c++)
                        {
                            value += Math.Exp(scores[c] - max) / sum * v[(h * capacity + c) * dim + d];
                        }

                        expected[(h * rowsPerHead + i) * dim + d] = (float)value;
                    }
                }
            }

            using var tq = Tensor.From(q, [heads, rowsPerHead, dim], device);
            using var tk = Tensor.From(k, [heads, capacity, dim], device);
            using var tv = Tensor.From(v, [heads, capacity, dim], device);
            using var position = Tensor.From([(float)offset], [1], device);
            string name = $"{heads}×{rowsPerHead} rows, steps {steps}, capacity {capacity}, dim {dim}, offset {offset}";
            AssertClose(expected, Tensor.AttentionTiled(tq, tk, tv, position, steps, scale).ToArray(), 2e-4f, "tiled: " + name);
            using var cache = new KeyValueCache(heads, capacity, dim, device, KeyValueFormat.Float32);
            cache.Keys.Load(k);
            cache.Values.Load(v);
            AssertClose(expected, Tensor.AttentionDecode(tq, cache, position, steps, scale).ToArray(), 2e-4f, "decoding: " + name);
        }
    }

    // Random named weights in NeuralSharp's layout, shared by DecoderSpec.Build and the reference implementation.
    private sealed class RandomWeights(int seed) : IWeightSource
    {
        private readonly Random _random = new(seed);

        public Dictionary<string, float[]> Values { get; } = [];

        public float[] Read(string name, IReadOnlyList<int> shape)
        {
            if (!Values.TryGetValue(name, out var values))
            {
                int count = shape.Aggregate(1, (a, b) => a * b);
                bool gain = name.EndsWith("norm.weight", StringComparison.Ordinal);
                Values[name] = values = [.. Enumerable.Range(0, count).Select(_ => gain ? 0.8f + 0.4f * _random.NextSingle() : (_random.NextSingle() * 2 - 1) * 0.5f)];
            }

            return values;
        }
    }

    // The model written out with loops, following the published definitions of these architectures.
    private static float[] ReferenceDecoder(DecoderSpec s, RandomWeights w, int[] ids)
    {
        int T = ids.Length, D = s.Dim, hd = s.HeadDim, H = s.Heads, KV = s.KvHeads, G = H / KV;
        float[] W(string name) => w.Values[name];
        float[] MatVec(float[] x, float[] m, int inputs, int outputs, float[]? bias)
        {
            var y = new float[outputs];
            for (int j = 0; j < outputs; j++)
            {
                double sum = bias?[j] ?? 0;
                for (int i = 0; i < inputs; i++)
                {
                    sum += x[i] * m[i * outputs + j];
                }

                y[j] = (float)sum;
            }

            return y;
        }

        float[] Norm(float[] x, string name, int offset = 0, int length = -1)
        {
            length = length < 0 ? x.Length : length;
            var y = (float[])x.Clone();
            var g = W($"{name}.weight");
            if (s.Norm == DecoderNorm.Rms)
            {
                double ms = 0;
                for (int i = 0; i < length; i++)
                {
                    ms += (double)x[offset + i] * x[offset + i];
                }

                double inv = 1 / Math.Sqrt(ms / length + s.NormEpsilon);
                for (int i = 0; i < length; i++)
                {
                    y[offset + i] = (float)(x[offset + i] * inv * (g[i] + s.NormOffset));
                }
            }
            else
            {
                var b = W($"{name}.bias");
                double mean = x.Skip(offset).Take(length).Average(v => (double)v);
                double variance = x.Skip(offset).Take(length).Average(v => (v - mean) * (v - mean));
                for (int i = 0; i < length; i++)
                {
                    y[offset + i] = (float)((x[offset + i] - mean) / Math.Sqrt(variance + s.NormEpsilon) * g[i] + b[i]);
                }
            }

            return y;
        }

        float Act(float v) => s.Activation switch
        {
            FeedForwardActivation.Silu => v / (1 + MathF.Exp(-v)),
            FeedForwardActivation.Gelu => 0.5f * v * (1 + MathF.Tanh(0.7978845608f * (v + 0.044715f * v * v * v))),
            _ => MathF.Max(0, v),
        };

        void Rotate(float[] x, int offset, int position)
        {
            if (s.Rope is not { } rope)
            {
                return;
            }

            var freq = rope.Frequencies(hd);
            int half = freq.Length;
            var original = x.AsSpan(offset, hd).ToArray();
            for (int i = 0; i < half; i++)
            {
                double c = Math.Cos(position * freq[i]), sn = Math.Sin(position * freq[i]);
                int a = rope.Interleaved ? 2 * i : i, b = rope.Interleaved ? 2 * i + 1 : i + half;
                x[offset + a] = (float)(original[a] * c - original[b] * sn);
                x[offset + b] = (float)(original[b] * c + original[a] * sn);
            }
        }

        var embed = W("embed.weight");
        var h = Enumerable.Range(0, T).Select(t => embed.AsSpan(ids[t] * D, D).ToArray().Select(v => v * (s.EmbeddingScale ?? 1f)).ToArray()).ToList();
        for (int layer = 0; layer < s.Layers; layer++)
        {
            string p = $"layers.{layer}";
            var normed = h.Select(x => Norm(x, $"{p}.attn_norm")).ToList();
            float[]? Bias(string name) => w.Values.GetValueOrDefault(name);
            var q = normed.Select(x => MatVec(x, W($"{p}.attn.q.weight"), D, H * hd, Bias($"{p}.attn.q.bias"))).ToList();
            var k = normed.Select(x => MatVec(x, W($"{p}.attn.k.weight"), D, KV * hd, Bias($"{p}.attn.k.bias"))).ToList();
            var v = normed.Select(x => MatVec(x, W($"{p}.attn.v.weight"), D, KV * hd, Bias($"{p}.attn.v.bias"))).ToList();
            for (int t = 0; t < T; t++)
            {
                for (int head = 0; head < H; head++)
                {
                    if (s.QkNorm)
                    {
                        q[t] = Norm(q[t], $"{p}.attn.q_norm", head * hd, hd);
                    }

                    Rotate(q[t], head * hd, t);
                }

                for (int head = 0; head < KV; head++)
                {
                    if (s.QkNorm)
                    {
                        k[t] = Norm(k[t], $"{p}.attn.k_norm", head * hd, hd);
                    }

                    Rotate(k[t], head * hd, t);
                }
            }

            var attended = new List<float[]>();
            for (int t = 0; t < T; t++)
            {
                var context = new float[H * hd];
                for (int head = 0; head < H; head++)
                {
                    int kvHead = head / G;
                    var scores = new double[t + 1];
                    for (int c = 0; c <= t; c++)
                    {
                        double dot = 0;
                        for (int d = 0; d < hd; d++)
                        {
                            dot += q[t][head * hd + d] * k[c][kvHead * hd + d];
                        }

                        scores[c] = dot / Math.Sqrt(hd);
                    }

                    double max = scores.Max(), total = scores.Sum(v => Math.Exp(v - max));
                    for (int c = 0; c <= t; c++)
                    {
                        double weight = Math.Exp(scores[c] - max) / total;
                        for (int d = 0; d < hd; d++)
                        {
                            context[head * hd + d] += (float)(weight * v[c][kvHead * hd + d]);
                        }
                    }
                }

                var o = MatVec(context, W($"{p}.attn.o.weight"), H * hd, D, Bias($"{p}.attn.o.bias"));
                attended.Add(s.PostNorms ? Norm(o, $"{p}.post_attn_norm") : o);
            }

            float[] FeedForward(float[] x)
            {
                var up = MatVec(x, W($"{p}.mlp.up.weight"), D, s.FfDim, Bias($"{p}.mlp.up.bias"));
                if (s.Gated)
                {
                    var gate = MatVec(x, W($"{p}.mlp.gate.weight"), D, s.FfDim, Bias($"{p}.mlp.gate.bias"));
                    up = [.. up.Select((u, i) => Act(gate[i]) * u)];
                }
                else
                {
                    up = [.. up.Select(Act)];
                }

                var down = MatVec(up, W($"{p}.mlp.down.weight"), s.FfDim, D, Bias($"{p}.mlp.down.bias"));
                return s.PostNorms ? Norm(down, $"{p}.post_mlp_norm") : down;
            }

            for (int t = 0; t < T; t++)
            {
                if (s.ParallelBlocks)
                {
                    var f = FeedForward(normed[t]);
                    h[t] = [.. h[t].Select((x, i) => x + attended[t][i] + f[i])];
                }
                else
                {
                    var x = h[t].Select((value, i) => value + attended[t][i]).ToArray();
                    var f = FeedForward(Norm(x, $"{p}.mlp_norm"));
                    h[t] = [.. x.Select((value, i) => value + f[i])];
                }
            }
        }

        var headWeights = s.TieEmbeddings ? null : W("head.weight");
        var logits = new List<float>();
        foreach (var x in h)
        {
            var normed = Norm(x, "norm");
            for (int token = 0; token < s.Vocabulary; token++)
            {
                double sum = s.HeadBias ? w.Values["head.bias"][token] : 0;
                for (int i = 0; i < D; i++)
                {
                    sum += normed[i] * (headWeights is null ? embed[token * D + i] : headWeights[i * s.Vocabulary + token]);
                }

                logits.Add((float)sum);
            }
        }

        return [.. logits];
    }

    private static DecoderSpec SmallSpec => new()
    {
        Vocabulary = 23, Dim = 16, Layers = 2, Heads = 4, KvHeads = 2, HeadDim = 6, FfDim = 24, MaxPositions = 32,
        Rope = new RopeSettings(500f), NormEpsilon = 1e-5f,
    };

    private static void DecoderMatchesReference(Device device)
    {
        var variants = new (string Name, DecoderSpec Spec)[]
        {
            ("grouped-query, head size 6", SmallSpec),
            ("q/k norm, qkv biases, untied", SmallSpec with { QkNorm = true, QkvBias = true }),
            ("tied, partial interleaved rope, scale, offset", SmallSpec with
            {
                TieEmbeddings = true, EmbeddingScale = 4f, NormOffset = 1f, Rope = new RopeSettings(100f, RotaryDim: 4, Interleaved: true),
            }),
            ("post-norms, GELU, llama3 rope scaling", SmallSpec with
            {
                PostNorms = true, Activation = FeedForwardActivation.Gelu,
                Rope = new RopeSettings(500f, Scaling: new RopeScaling("llama3", 8f, 1f, 4f, 8)),
            }),
            ("parallel, layer norm, plain feed-forward, all biases", SmallSpec with
            {
                ParallelBlocks = true, Norm = DecoderNorm.Layer, Gated = false, QkvBias = true, OutputBias = true, FeedForwardBias = true,
                HeadBias = true, KvHeads = 4,
            }),
        };
        int[] ids = [3, 17, 5, 5, 22, 0, 9];
        foreach (var (name, spec) in variants)
        {
            var weights = new RandomWeights(61);
            using var model = spec.Build(weights, new DecoderBuildOptions { Device = device });
            var reference = ReferenceDecoder(spec, weights, ids);
            using var input = Tensor.From([.. ids.Select(i => (float)i)], [1, ids.Length], device);
            using var logits = model.Predict(input);
            AssertClose(reference, logits.ToArray(), 1e-3f, name);
            model.Train();
            using (var scope = new TensorScope())
            {
                AssertClose(reference, model.Forward(input).ToArray(), 1e-3f, $"{name} (training path)");
            }
        }
    }

    private static void DecoderCachedAndInt8(Device device)
    {
        var spec = SmallSpec with { QkNorm = true };
        using var model = spec.Build(new RandomWeights(62), new DecoderBuildOptions { Device = device });
        model.Eval();
        int[] ids = [1, 4, 9, 16, 2, 7, 11, 3, 8, 20];
        int T = ids.Length, V = spec.Vocabulary;
        using var sequence = Tensor.From([.. ids.Select(i => (float)i)], [1, T], device);
        var full = model.Predict(sequence).ToArray();
        float range = full.Max(MathF.Abs);
        foreach (var format in new[] { KeyValueFormat.Float32, KeyValueFormat.Int8 })
        {
            float tolerance = format == KeyValueFormat.Float32 ? 1e-4f : 0.03f * range;
            using var context = new DecodingContext(device, 1, 16, format);
            using (Autograd.NoGrad())
            {
                using var prompt = Tensor.From([.. ids.Take(4).Select(i => (float)i)], [1, 4], device);
                AssertClose(full[..(4 * V)], model.ForwardCached(prompt, context).ToArray(), tolerance, $"{format} prefill");
                for (int t = 4; t < T; t++)
                {
                    using var next = Tensor.From([(float)ids[t]], [1, 1], device);
                    AssertClose(full[(t * V)..((t + 1) * V)], model.ForwardCached(next, context).ToArray(), tolerance, $"{format} step {t}");
                }
            }
        }

        using var quantized = spec.Build(new RandomWeights(62), new DecoderBuildOptions { Device = device, Int8 = true });
        Check(quantized.Descendants().OfType<Linear>().All(l => l.Int8 is not null), "every projection is int8");
        var int8 = quantized.Predict(sequence).ToArray();
        float change = int8.Zip(full, (a, b) => MathF.Abs(a - b)).Max();
        Check(change < 0.05f * range, $"int8 logit change {change} (range {range})");

        var generator = new TextGenerator(model, new CharTokenizer("abcdefghijklmnopqrstuvw"), 16);
        var (text, _, stats) = generator.Generate("abc", new GenerationOptions { Temperature = 0f, TopK = 1, NumPredict = 8 });
        Check(stats.GeneratedTokens == 8 && text.Length == 8, $"generated '{text}'");
    }

    private static void DecoderLoraAndPackage(Device device)
    {
        var spec = SmallSpec;
        using var model = spec.Build(options: new DecoderBuildOptions { Device = device, Seed = 63 });
        int adapters = model.AddLora(rank: 2, alpha: 4, targets: l => l.Name is "q" or "v", freezeBase: true, random: new Random(64));
        Check(adapters == 2 * spec.Layers, $"{adapters} adapters on the q and v projections");
        var random = new Random(65);
        var ids = Enumerable.Range(0, 4 * 9).Select(_ => (float)random.Next(spec.Vocabulary)).ToArray();
        using var input = Tensor.From(ids, [4, 9], device);
        using var targets = Tensor.OneHot(Tensor.From([.. ids.Select((_, i) => ids[(i / 9) * 9 + (i % 9 + 1) % 9])], [4, 9], device), spec.Vocabulary);
        using var optimizer = new Adam(model.TrainableParameters(), 0.05f);
        model.Train();
        float first = 0, last = 0;
        for (int step = 0; step < 30; step++)
        {
            using var scope = new TensorScope();
            var loss = Losses.CrossEntropy(model.Forward(input), targets);
            optimizer.ZeroGrad();
            loss.Backward();
            optimizer.Step();
            (first, last) = step == 0 ? (loss.Item(), loss.Item()) : (first, loss.Item());
        }

        Check(last < first * 0.8f, $"LoRA loss {first:F3} → {last:F3}");
        model.MergeLora();
        model.Eval();

        Check(DecoderSpec.FromJson(spec.ToJson()) == spec with { }, "JSON round trip");
        var withRope = spec with { Rope = new RopeSettings(1e6f, 4, true, new RopeScaling("linear", 2f)) };
        Check(DecoderSpec.FromJson(withRope.ToJson()).Rope == withRope.Rope, "rope settings round trip");

        string path = Path.Combine(Path.GetTempPath(), $"ns-{Guid.NewGuid():N}.nsm");
        try
        {
            ModelPackage.Create(path).Architecture(ModelPackage.DefaultModelName, spec.ToJson()).Weights(model).Save();
            using var package = ModelPackage.Open(path);
            using var reloaded = package.BuildModel(device: device);
            using var probe = Tensor.From([1f, 2f, 3f], [1, 3], device);
            AssertClose(model.Predict(probe).ToArray(), reloaded.Predict(probe).ToArray(), 1e-5f, "package round trip");
        }
        finally
        {
            File.Delete(path);
        }
    }
}

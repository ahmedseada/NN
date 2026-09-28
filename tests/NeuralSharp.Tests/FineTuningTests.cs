using System.Text.Json.Nodes;
using NeuralSharp;
using NeuralSharp.Generation;
using NeuralSharp.Layers;
using NeuralSharp.Pretrained;

internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] FineTuning =
    [
        ("fine-tuning: chunked token cross-entropy (all rows or trained rows only) and the frozen-transpose product and the fused LoRA term match dense results and gradients", TokenLoss),
        ("fine-tuning: LoRA terms inside the tensor-core products (float32, bfloat16 and 4-bit bases, one layer or merged) and rank-16 products match the separate computation, with gradients", LoraInsideProducts),
        ("fine-tuning: packed sequences (several per row, rotary or learned positions) give each sequence the logits and gradients it gets alone; packing fills rows first-fit", PackedSequencesMatch),
        ("fine-tuning: a training step recorded as a CUDA graph and replayed gives the losses and adapters of ordinary steps", GraphTraining),
        ("memory: a full GPU raises a clear error, or with offloading places tensors in system memory the kernels still use", HostOffload),
        ("fine-tuning: activation checkpointing gives the same loss and gradients (adapters and input)", CheckpointingGradients),
        ("models: a Hugging Face model id is downloaded (only the files the library reads, sharded weights) into the cache, loads, and works offline", ModelDownload),
        ("fine-tuning: agent transcripts through the chat template, assistant-only tokens, LoRA and QLoRA training, PEFT adapters, merged export", AgentFineTuning),
    ];

    private static void HostOffload(Device device)
    {
        if (device.Type != DeviceType.Cuda)
        {
            return;                                                              // the CPU's memory is system memory already
        }

        var r = new Random(95);
        float[] Values(int n) => [.. Enumerable.Range(0, n).Select(_ => (float)(r.NextDouble() * 2 - 1))];
        var (av, bv) = (Values(512 * 512), Values(512 * 512));
        using var ca = Tensor.From(av, [512, 512], Device.Cpu);
        using var cb = Tensor.From(bv, [512, 512], Device.Cpu);
        var expected = ca.MatMul(cb).ToArray();
        long? limit = ComputeResources.GpuMemoryLimit;
        bool offload = ComputeResources.OffloadToHostMemory;
        ComputeResources.ReleaseCachedMemory(device);
        try
        {
            ComputeResources.GpuMemoryLimit = ComputeResources.GetMemoryUsage(device).InUse + (256L << 10);   // less than one 1 MiB tensor
            ComputeResources.OffloadToHostMemory = false;
            try
            {
                using var tooBig = Tensor.From(av, [512, 512], device);
                Check(false, "a tensor beyond the limit is refused");
            }
            catch (ResourceLimitExceededException)
            {
            }

            ComputeResources.OffloadToHostMemory = true;
            using (var scope = new TensorScope())
            {
                var a = Tensor.From(av, [512, 512], device);
                var b = Tensor.From(bv, [512, 512], device);
                var c = a.MatMul(b) + 0f;
                Check(ComputeResources.GetMemoryUsage(device).Offloaded >= 3L * 512 * 512 * 4, $"offloaded: {ComputeResources.GetMemoryUsage(device)}");
                AssertClose(expected, c.ToArray(), 1e-3f, "a product computed in system memory");
            }

            Check(ComputeResources.GetMemoryUsage(device).Offloaded == 0, "offloaded blocks are returned");
        }
        finally
        {
            ComputeResources.GpuMemoryLimit = limit;
            ComputeResources.OffloadToHostMemory = offload;
            ComputeResources.ReleaseCachedMemory(device);
        }
    }

    private static void CheckpointingGradients(Device device)
    {
        var spec = SmallSpec with { QkNorm = true };
        var r = new Random(92);
        var inputValues = Enumerable.Range(0, 2 * 7 * spec.Dim).Select(_ => (float)(r.NextDouble() * 2 - 1)).ToArray();
        var weightValues = Enumerable.Range(0, 2 * 7 * spec.Dim).Select(_ => (float)(r.NextDouble() * 2 - 1)).ToArray();
        (float Loss, float[] Input, float[] Adapters) Run(bool checkpointed)
        {
            using var model = spec.Build(new RandomWeights(93), new DecoderBuildOptions { Device = device });
            model.AddLora(rank: 2, alpha: 4, targets: l => l.Name is "q" or "v" or "down", freezeBase: true, random: new Random(94));
            foreach (var adapter in model.Descendants().OfType<Linear>().Select(l => l.Adapter).OfType<LoraAdapter>())
            {
                adapter.B.Load([.. Enumerable.Range(0, adapter.B.Size).Select(i => MathF.Sin(i))]);
            }

            model.Train();
            using var scope = new TensorScope();
            var x = Tensor.From(inputValues, [2, 7, spec.Dim], device, requiresGrad: true);
            var hidden = x;
            foreach (var block in model.OfType<DecoderBlock>())
            {
                hidden = checkpointed ? block.ForwardCheckpointed(hidden) : block.Forward(hidden);
            }

            var loss = (hidden * Tensor.From(weightValues, [2, 7, spec.Dim], device)).Sum();
            loss.Backward();
            var adapters = model.TrainableParameters().SelectMany(p => p.Grad!.ToArray()).ToArray();
            return (loss.Item(), x.Grad!.ToArray(), adapters);
        }

        var plain = Run(false);
        var checkpointed = Run(true);
        AssertClose([plain.Loss], [checkpointed.Loss], 1e-4f, "loss");
        AssertClose(plain.Input, checkpointed.Input, 1e-4f, "input gradient");
        AssertClose(plain.Adapters, checkpointed.Adapters, 1e-4f, "adapter gradients");
    }

    // A tiny Qwen3-style model folder: byte-level tokenizer with ChatML tokens, Qwen3's chat template, random weights.
    private static void ModelDownload(Device device)
    {
        var spec = new DecoderSpec
        {
            Vocabulary = 260, Dim = 32, Layers = 2, Heads = 4, KvHeads = 2, HeadDim = 8, FfDim = 64, MaxPositions = 256,
            Rope = new RopeSettings(10000f), NormEpsilon = 1e-6f, QkNorm = true, TieEmbeddings = true,
        };
        string source = WriteChatModel(spec);
        string cache = Path.Combine(Path.GetTempPath(), "ns-models-" + Guid.NewGuid().ToString("N"));
        try
        {
            // Re-save the weights as two shards with an index, as larger models ship.
            foreach (var file in Directory.GetFiles(source, "*.safetensors"))
            {
                File.Delete(file);
            }

            var weights = new RandomWeights(91);
            using (spec.Build(weights, new DecoderBuildOptions { Device = Device.Cpu }))
            {
            }

            WriteCheckpoint(source, spec, weights, "Qwen3ForCausalLM", SafeTensorType.F32, sharded: true);
            var served = Directory.GetFiles(source).Select(Path.GetFileName).ToList();
            Check(served.Contains("model.safetensors.index.json") && served.Count(f => f!.EndsWith(".safetensors", StringComparison.Ordinal)) >= 2, "sharded source");

            const string hf = "https://huggingface.co", commit = "abcdefabcdefabcdefabcdefabcdefabcdefabcd";
            var web = new FakeRouter();
            web.Json($"{hf}/api/models/org/tiny/revision/main", $"{{\"sha\":\"{commit}\"}}");
            var listing = new JsonArray();
            foreach (var name in served.Concat(["pytorch_model.bin", "README.md", "onnx/model.onnx", "original/consolidated.safetensors"]))
            {
                listing.Add((JsonNode)new JsonObject { ["type"] = "file", ["path"] = name, ["size"] = 10 });
            }

            web.Json($"{hf}/api/models/org/tiny/tree/{commit}?recursive=true", listing.ToJsonString());
            foreach (var name in served)
            {
                web.Bytes($"{hf}/org/tiny/resolve/{commit}/{name}", File.ReadAllBytes(Path.Combine(source, name!)), requireToken: "hf_model");
            }

            var downloader = new NeuralSharp.Datasets.Downloader(new HttpClient(web), cache) { Attempts = 1 };
            string folder = ModelSource.DownloadAsync("org/tiny", token: "hf_model", downloader: downloader).GetAwaiter().GetResult();
            Check(folder == Path.Combine(cache, "huggingface", "models", "org", "tiny", commit[..12]), $"model folder {folder}");
            Check(Directory.GetFiles(folder).Select(Path.GetFileName).Order().SequenceEqual(served.Order()), "only the files the library reads");

            using (var original = PretrainedModel.Load(source, new PretrainedOptions { Device = device }))
            using (var downloaded = PretrainedModel.Load(folder, new PretrainedOptions { Device = device }))
            {
                int[] ids = [.. downloaded.Tokenizer!.Encode("hello there, general kenobi")];
                TrainingSequence[] probe = [new TrainingSequence(ids, [.. ids.Select(_ => true)])];
                AssertClose([FineTuner.Evaluate(original, probe)], [FineTuner.Evaluate(downloaded, probe)], 0f, "same loss as the original folder");
            }

            // Without a network, the downloaded copy is used.
            var offline = new NeuralSharp.Datasets.Downloader(new HttpClient(new Unreachable()), cache) { Attempts = 1 };
            Check(ModelSource.Resolve("org/tiny", downloader: offline) == folder, "offline: the cached copy");
            Check(ModelSource.IsModelId("Qwen/Qwen3-0.6B") && !ModelSource.IsModelId(source) && !ModelSource.IsModelId("a/b/c"), "ids and folders");
        }
        finally
        {
            Directory.Delete(source, true);
            if (Directory.Exists(cache))
            {
                Directory.Delete(cache, true);
            }
        }
    }

    private sealed class Unreachable : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) =>
            throw new HttpRequestException("No such host is known.");
    }

    private static string WriteChatModel(DecoderSpec spec)
    {
        string folder = TempFolder();
        var weights = new RandomWeights(91);
        using (spec.Build(weights, new DecoderBuildOptions { Device = Device.Cpu }))
        {
        }

        WriteCheckpoint(folder, spec, weights, "Qwen3ForCausalLM", SafeTensorType.F32, sharded: false);
        var printable = Enumerable.Range('!', '~' - '!' + 1).Concat(Enumerable.Range('¡', '¬' - '¡' + 1)).Concat(Enumerable.Range('®', 'ÿ' - '®' + 1)).ToHashSet();
        var vocab = new JsonObject();
        for (int b = 0, n = 0; b < 256; b++)
        {
            vocab[((char)(printable.Contains(b) ? b : 256 + n++)).ToString()] = b;
        }

        var tokenizer = new JsonObject
        {
            ["added_tokens"] = new JsonArray(
                new JsonObject { ["id"] = 256, ["content"] = "<|im_start|>", ["special"] = true },
                new JsonObject { ["id"] = 257, ["content"] = "<|im_end|>", ["special"] = true },
                new JsonObject { ["id"] = 258, ["content"] = "<|endoftext|>", ["special"] = true }),
            ["model"] = new JsonObject { ["type"] = "BPE", ["vocab"] = vocab, ["merges"] = new JsonArray() },
            ["pre_tokenizer"] = new JsonObject { ["type"] = "ByteLevel", ["add_prefix_space"] = false, ["use_regex"] = true },
            ["decoder"] = new JsonObject { ["type"] = "ByteLevel" },
        };
        File.WriteAllText(Path.Combine(folder, "tokenizer.json"), tokenizer.ToJsonString());
        File.WriteAllText(Path.Combine(folder, "tokenizer_config.json"), new JsonObject
        {
            ["chat_template"] = Qwen3Template, ["eos_token"] = "<|im_end|>", ["bos_token"] = null,
        }.ToJsonString());
        return folder;
    }

    private static void AgentFineTuning(Device device)
    {
        var spec = new DecoderSpec
        {
            Vocabulary = 260, Dim = 32, Layers = 2, Heads = 4, KvHeads = 2, HeadDim = 8, FfDim = 64, MaxPositions = 1024,
            Rope = new RopeSettings(10000f), NormEpsilon = 1e-6f, QkNorm = true, TieEmbeddings = true,
        };
        string folder = WriteChatModel(spec), data = Path.Combine(folder, "train.jsonl"), adapter = Path.Combine(folder, "adapter"),
            exported = Path.Combine(folder, "merged");
        try
        {
            File.WriteAllLines(data,
            [
                """{"messages": [{"role": "system", "content": "You fix code."}, {"role": "user", "content": "Fix a.py"}, {"role": "assistant", "content": "", "reasoning_content": "Read it.", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{\"path\": \"a.py\"}"}}]}, {"role": "tool", "tool_call_id": "c1", "content": "x = 1/0"}, {"role": "assistant", "content": "Divide by one."}], "tools": [{"type": "function", "function": {"name": "read", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]}""",
                "",
                """{"messages": [{"role": "user", "content": [{"type": "text", "text": "Say hi"}]}, {"role": "assistant", "content": "hi"}]}""",
                """{"conversations": [{"from": "human", "value": "2+2?"}, {"from": "gpt", "value": "4"}]}""",
            ]);
            var transcripts = ChatTranscript.ReadJsonLines(data).ToList();
            Check(transcripts.Count == 3 && transcripts[0].Tools.Count == 1 && transcripts[0].Messages[2].ToolCalls![0].Name == "read"
                  && (string?)transcripts[0].Messages[2].ToolCalls![0].Arguments["path"] == "a.py" && transcripts[0].Messages[3].ToolName == "read"
                  && transcripts[0].Messages[2].Thinking == "Read it." && transcripts[1].Messages[0].Content == "Say hi" && transcripts[2].Messages[1].Role == "assistant",
                "transcripts parse (tools, string arguments, tool_call_id, reasoning, content parts, ShareGPT)");

            using var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
            var encoder = new ChatTranscriptEncoder(model.ChatTemplate!, model.Tokenizer!);
            Check(encoder.AssistantHeader == "<|im_start|>assistant\n" && encoder.AssistantEnd == "<|im_end|>",
                $"assistant markers '{encoder.AssistantHeader}' / '{encoder.AssistantEnd}'");
            var sequences = transcripts.Select(t => encoder.Encode(t, 1000)!).ToList();
            var tokenizer = model.Tokenizer!;
            string Trained(TrainingSequence s) => tokenizer.Decode(s.Tokens.Where((_, i) => s.Trained[i]));
            string first = Trained(sequences[0]);
            Check(first.Contains("<tool_call>") && first.Contains("\"path\": \"a.py\"") && first.Contains("Divide by one.<|im_end|>")
                  && first.Contains("Read it.") && !first.Contains("Fix a.py") && !first.Contains("x = 1/0") && !first.Contains("You fix code"),
                $"only the assistant's turns are trained: {first}");
            Check(Trained(sequences[1]).EndsWith("hi<|im_end|>") && sequences.All(s => s.Tokens.Length <= 1001), "short transcripts");
            Check(encoder.Encode(transcripts[0], 20) is null || encoder.Encode(transcripts[0], 20)!.Tokens.Length == 21, "cut to the maximum length");

            // Rows from datasets: conversations as before; plain text trains every token, long texts in chunks.
            var row = JsonNode.Parse(File.ReadLines(data).First())!.AsObject();
            Check(encoder.EncodeRow(row, 1000).Single().Tokens.SequenceEqual(sequences[0].Tokens), "a conversation row encodes as its transcript");
            string text = string.Concat(Enumerable.Repeat("the quick brown fox jumps over the lazy dog. ", 6));
            var whole = encoder.EncodeText(text, 10_000).Single();
            var chunks = encoder.EncodeRow(new JsonObject { ["text"] = text }, 16).ToList();
            Check(whole.Trained.All(t => t) && chunks.Count > 1 && chunks.All(c => c.Tokens.Length <= 17 && c.Trained.All(t => t))
                  && chunks.Sum(c => c.TrainedTokens) == whole.TrainedTokens && chunks[0].Tokens.Concat(chunks[1].Tokens.Skip(1)).SequenceEqual(whole.Tokens.Take(chunks[0].Tokens.Length + chunks[1].Tokens.Length - 1)),
                $"text rows: {chunks.Count} chunks predict each of the {whole.TrainedTokens} tokens once");
            Check(!encoder.EncodeRow(new JsonObject { ["label"] = 3 }, 100).Any(), "rows without text or messages give nothing");

            // LoRA: the transcripts are learned (loss falls), and the adapters round-trip through the PEFT files.
            var options = new FineTuningOptions { Rank = 4, Alpha = 8, LearningRate = 1e-2f, Epochs = 40, BatchTokens = 1024, WarmupFraction = 0f, Seed = 5 };
            float before = FineTuner.Evaluate(model, sequences);
            var progress = new List<FineTuningProgress>();
            FineTuner.Train(model, sequences, null, options, adapter, new Progress<FineTuningProgress>(progress.Add));
            float after = FineTuner.Evaluate(model, sequences);
            Check(after < before * 0.5f, $"loss {before:F3} → {after:F3}");
            Check(File.Exists(Path.Combine(adapter, "adapter_model.safetensors")) && File.Exists(Path.Combine(adapter, "adapter_config.json")), "PEFT files");
            var peft = JsonNode.Parse(File.ReadAllText(Path.Combine(adapter, "adapter_config.json")))!;
            Check((int)peft["r"]! == 4 && (float)peft["lora_alpha"]! == 8f && peft["target_modules"]!.AsArray().Count == 7, $"adapter config {peft}");
            using (var reader = SafeTensorsReader.Open(Path.Combine(adapter, "adapter_model.safetensors")))
            {
                Check(reader.Tensors["base_model.model.model.layers.1.self_attn.q_proj.lora_A.weight"].Shape.SequenceEqual([4, 32])
                      && reader.Tensors["base_model.model.model.layers.1.mlp.down_proj.lora_B.weight"].Shape.SequenceEqual([32, 4]), "PEFT names and layouts");
            }

            using (var reloaded = PretrainedModel.Load(folder, new PretrainedOptions { Device = device }))
            {
                Check(reloaded.LoadAdapter(adapter) == 7 * spec.Layers, "every adapter loads");
                AssertClose([after], [FineTuner.Evaluate(reloaded, sequences)], 1e-4f, "loss with the reloaded adapters");
                reloaded.SaveHuggingFace(exported, SafeTensorType.F32);
            }

            using (var merged = PretrainedModel.Load(exported, new PretrainedOptions { Device = device }))
            {
                Check(!merged.Network.Descendants().OfType<Linear>().Any(l => l.Adapter is not null), "the export has no adapters");
                AssertClose([after], [FineTuner.Evaluate(merged, sequences)], 1e-3f, "loss of the merged export");
            }

            using (var merged = PretrainedModel.Load(folder, new PretrainedOptions { Device = device, MergeAdapter = adapter }))
            {
                Check(!merged.Network.Descendants().OfType<Linear>().Any(l => l.Adapter is not null)
                      && merged.Notes.Any(n => n.Contains($"merged into {7 * spec.Layers} weights")), "merged while loading");
                AssertClose([after], [FineTuner.Evaluate(merged, sequences)], 1e-3f, "loss with the adapter merged while loading");
            }

            using (var merged = PretrainedModel.Load(folder, new PretrainedOptions { Device = device, MergeAdapter = adapter, Int8 = true }))
            {
                float loss = FineTuner.Evaluate(merged, sequences);
                Check(loss < before * 0.5f, $"int8 weights with the adapter merged while loading: {loss:F3}");
            }

            // QLoRA: adapters on a 4-bit base also learn.
            using var quantized = PretrainedModel.Load(folder, new PretrainedOptions { Device = device, Int4 = true });
            float q0 = FineTuner.Evaluate(quantized, sequences);
            FineTuner.Train(quantized, sequences, sequences, options with { Epochs = 25 });
            float q1 = FineTuner.Evaluate(quantized, sequences);
            Check(q1 < q0 * 0.6f, $"QLoRA loss {q0:F3} → {q1:F3}");
            Check(progress.Count > 0 && progress[^1].Step == progress[^1].TotalSteps && progress.All(p => float.IsFinite(p.Loss)), "progress reports");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void TokenLoss(Device device)
    {
        const int Rows = 11, Dim = 6, Vocabulary = 37;
        var r = new Random(81);
        float[] Random(int n) => [.. Enumerable.Range(0, n).Select(_ => (float)(r.NextDouble() * 2 - 1))];
        var hiddenValues = Random(Rows * Dim);
        var ids = Enumerable.Range(0, Rows).Select(_ => (float)r.Next(Vocabulary)).ToArray();
        var weights = Enumerable.Range(0, Rows).Select(i => i % 4 == 0 ? 0f : 1f).ToArray();
        float normalizer = weights.Sum();
        var adapterB = Random(2 * Vocabulary);

        float[] Run(int mode, out float[] hiddenGrad, out float[] adapterGrad)
        {
            using var head = new Linear(Dim, Vocabulary, bias: true, device, new Random(82));
            head.AddLora(rank: 2, alpha: 2, targets: _ => true, freezeBase: true, random: new Random(83));
            head.Adapter!.B.Load(adapterB);
            using var scope = new TensorScope();
            var hidden = Tensor.From(hiddenValues, [Rows, Dim], device, requiresGrad: true);
            Tensor loss;
            if (mode == 2)
            {
                // Only the rows with a weight: the head never sees the others.
                int[] rows = [.. Enumerable.Range(0, Rows).Where(i => weights[i] != 0f)];
                loss = Losses.TokenCrossEntropyRows(hidden, h => head.Forward(h), rows, [.. rows.Select(i => ids[i])], [.. rows.Select(i => weights[i])],
                    normalizer, chunkRows: 3);
            }
            else if (mode == 1)
            {
                using var targets = Tensor.From(ids, [Rows], device);
                using var w = Tensor.From(weights, [Rows], device);
                loss = Losses.TokenCrossEntropy(hidden, h => head.Forward(h), targets, w, normalizer, chunkRows: 4);
            }
            else
            {
                var weighted = new float[Rows * Vocabulary];
                for (int i = 0; i < Rows; i++)
                {
                    weighted[i * Vocabulary + (int)ids[i]] = weights[i];
                }

                var onehot = Tensor.From(weighted, [Rows, Vocabulary], device);
                loss = (head.Forward(hidden).LogSoftmax() * onehot).Sum() * (-1f / normalizer);
            }

            loss.Backward();
            hiddenGrad = hidden.Grad!.ToArray();
            adapterGrad = head.Adapter.A.Grad!.ToArray();
            return loss.ToArray();
        }

        var expected = Run(0, out var expectedHidden, out var expectedAdapter);
        var actual = Run(1, out var actualHidden, out var actualAdapter);
        AssertClose(expected, actual, 1e-4f, "loss");
        AssertClose(expectedHidden, actualHidden, 1e-4f, "hidden gradient");
        AssertClose(expectedAdapter, actualAdapter, 1e-4f, "adapter gradient");
        var rowsOnly = Run(2, out var rowsHidden, out var rowsAdapter);
        AssertClose(expected, rowsOnly, 1e-4f, "loss on trained rows only");
        AssertClose(expectedHidden, rowsHidden, 1e-4f, "hidden gradient, trained rows only");
        AssertClose(expectedAdapter, rowsAdapter, 1e-4f, "adapter gradient, trained rows only");

        // A frozen matrix used through a transposed copy: same product and input gradient as x · Wᵀ.
        var tableValues = Random(Vocabulary * Dim);
        float[] Tied(bool cached, out float[] inputGrad)
        {
            using var scope = new TensorScope();
            var table = Tensor.From(tableValues, [Vocabulary, Dim], device);
            var x = Tensor.From(hiddenValues, [Rows, Dim], device, requiresGrad: true);
            using var transposed = Tensor.TransposedCopy(table);
            var y = cached ? Tensor.MatMulFrozenTransposed(x, table, transposed) : x.MatMul(table, transposeB: true);
            (y * y).Sum().Backward();
            inputGrad = x.Grad!.ToArray();
            return y.ToArray();
        }

        AssertClose(Tied(false, out var plainGrad), Tied(true, out var cachedGrad), 1e-4f, "frozen transposed product");
        AssertClose(plainGrad, cachedGrad, 1e-4f, "frozen transposed product: input gradient");

        // The fused LoRA term: same output and gradients (input, A, B, base weight) as product + x·A·B·scale.
        var w0 = Random(Dim * Vocabulary);
        var a0 = Random(Dim * 3);
        var b0 = Random(3 * Vocabulary);
        float[][] Lora(bool fused)
        {
            using var scope = new TensorScope();
            var x = Tensor.From(hiddenValues, [Rows, Dim], device, requiresGrad: true);
            var w = Tensor.From(w0, [Dim, Vocabulary], device, requiresGrad: true);
            var a = Tensor.From(a0, [Dim, 3], device, requiresGrad: true);
            var b = Tensor.From(b0, [3, Vocabulary], device, requiresGrad: true);
            var product = x.MatMul(w);
            var y = fused ? Tensor.AddLowRank(product, x, a, b, 0.75f) : product + x.MatMul(a).MatMul(b) * 0.75f;
            (y * y).Sum().Backward();
            return [y.ToArray(), x.Grad!.ToArray(), w.Grad!.ToArray(), a.Grad!.ToArray(), b.Grad!.ToArray()];
        }

        var plainLora = Lora(false);
        var fusedLora = Lora(true);
        string[] parts = ["output", "input gradient", "base weight gradient", "A gradient", "B gradient"];
        for (int i = 0; i < parts.Length; i++)
        {
            AssertClose(plainLora[i], fusedLora[i], 1e-3f, $"fused LoRA: {parts[i]}");
        }
    }

    private static void LoraInsideProducts(Device device)
    {
        const int Batch = 2, Steps = 100, In = 96, Out = 384, Rank = 16;
        var random = new Random(71);
        float[] Values(int n) => RandomArray(random, n);
        if (device.Type != DeviceType.Cuda || MixedPrecision.TensorCoresUnavailable(device) is not null)
        {
            // Devices without the fused kernels decline, and callers compute the term separately.
            using var scope = new TensorScope();
            var layer = new Linear(In, Out, bias: false, device: device, random: random);
            layer.Adapter = new LoraAdapter(Tensor.From(Values(In * Rank), [In, Rank], device), Tensor.From(Values(Rank * Out), [Rank, Out], device), Rank, 2f);
            Check(Tensor.LoraProducts(Tensor.From(Values(Steps * In), [Steps, In], device), [layer]) is null, "no fused LoRA products on this device");
            return;
        }

        static double RelativeError(float[] expected, float[] actual)
        {
            double difference = 0, norm = 0;
            for (int i = 0; i < expected.Length; i++)
            {
                difference += (double)(expected[i] - actual[i]) * (expected[i] - actual[i]);
                norm += (double)expected[i] * expected[i];
            }

            return Math.Sqrt(difference / Math.Max(norm, 1e-30));
        }

        void Close(float[] expected, float[] actual, string what, double tolerance = 2e-2)
        {
            Check(expected.Length == actual.Length, $"{what}: length {actual.Length}, expected {expected.Length}");
            double error = RelativeError(expected, actual);
            Check(error < tolerance, $"{what}: relative error {error:G3}");
        }

        using var precision = MixedPrecision.BFloat16();

        // Skinny products (one side 16) on tensor cores: as the CPU computes them, in every layout.
        foreach (var (m, n, k, ta, tb) in new[] { (300, 16, 2000, false, false), (2000, 16, 300, true, false), (300, 16, 2000, false, true),
                     (16, 2000, 300, true, false), (300, 2000, 16, false, false), (300, 2000, 16, false, true) })
        {
            var av = Values(m * k);
            var bv = Values(k * n);
            var cv = Values(m * n);
            float[] Product(Device on, float beta)
            {
                using var a = Tensor.From(av, ta ? [k, m] : [m, k], on);
                using var b = Tensor.From(bv, tb ? [n, k] : [k, n], on);
                using var c = Tensor.From(cv, [m, n], on);
                on.Backend.BatchedMatMul(a.Storage, b.Storage, c.Storage, 1, m, n, k, ta, tb, beta);
                return c.ToArray();
            }

            foreach (float beta in new[] { 0f, 1f })
            {
                Close(Product(Device.Cpu, beta), Product(device, beta), $"{m}x{n}x{k} {(ta ? 't' : 'n')}{(tb ? 't' : 'n')} beta {beta}");
            }
        }

        bool fp8Available = device.Backend.Float8PaddedK(In) > 0;
        foreach (string format in new[] { "float32", "bfloat16", "int4", "bfloat16 + fp8", "float32 + fp8" })
        {
            bool fp8 = format.EndsWith("fp8", StringComparison.Ordinal);
            if (fp8 && !fp8Available)
            {
                continue;
            }

            foreach (int count in format.StartsWith("float32", StringComparison.Ordinal) ? new[] { 1 } : new[] { 1, 2, 3 })
            {
                var weights = Enumerable.Range(0, count).Select(_ => Values(In * Out)).ToArray();
                var aValues = Enumerable.Range(0, count).Select(_ => Values(In * Rank)).ToArray();
                var bValues = Enumerable.Range(0, count).Select(_ => Values(Rank * Out)).ToArray();
                var coefficients = Enumerable.Range(0, count).Select(_ => Values(Batch * Steps * Out)).ToArray();
                var xValues = Values(Batch * Steps * In);
                float[][] Run(bool fused)
                {
                    using var scope = new TensorScope();
                    var layers = new Linear[count];
                    for (int j = 0; j < count; j++)
                    {
                        layers[j] = Linear.FromWeights(Tensor.From(weights[j], [In, Out], device));
                        if (format.StartsWith("bfloat16", StringComparison.Ordinal))
                        {
                            layers[j].ToBFloat16();
                        }
                        else if (format == "int4")
                        {
                            layers[j].QuantizeInt4();
                        }

                        layers[j].Adapter = new LoraAdapter(Tensor.From(aValues[j], [In, Rank], device, requiresGrad: true),
                            Tensor.From(bValues[j], [Rank, Out], device, requiresGrad: true), Rank, 1.5f);
                    }

                    var x = Tensor.From(xValues, [Batch, Steps, In], device, requiresGrad: true);
                    Tensor[] ys;
                    if (fused)
                    {
                        if (fp8)
                        {
                            Check(layers.All(l => l.AttachFloat8()), $"{format}: FP8 copies of the weights");
                        }

                        ys = Tensor.LoraProducts(x, layers) ?? throw new Exception($"{format} × {count}: no fused LoRA products on {device}");
                        foreach (var layer in layers)
                        {
                            layer.DetachFloat8();
                        }
                    }
                    else
                    {
                        ys = [.. layers.Select(l =>
                        {
                            var product = l.BFloat16 is { } h ? x.MatMulBFloat16(h) : l.Int4 is { } q ? x.MatMulInt4(q) : x.MatMul(l.Weight);
                            return product + x.MatMul(l.Adapter!.A).MatMul(l.Adapter.B) * l.Adapter.Scale;
                        })];
                    }

                    var loss = ys.Select((y, j) => (y * Tensor.From(coefficients[j], [Batch, Steps, Out], device)).Sum()).Aggregate((p, q) => p + q);
                    loss.Backward();
                    return [.. ys.Select(y => y.ToArray()), x.Grad!.ToArray(), .. layers.Select(l => l.Adapter!.A.Grad!.ToArray()),
                        .. layers.Select(l => l.Adapter!.B.Grad!.ToArray())];
                }

                var expected = Run(false);
                var actual = Run(true);
                for (int i = 0; i < expected.Length; i++)
                {
                    string part = i < count ? $"output {i}" : i == count ? "input gradient" : i <= 2 * count ? $"A gradient {i - count - 1}" : $"B gradient {i - 2 * count - 1}";
                    Close(expected[i], actual[i], $"{format} × {count}: {part}", fp8 && i < count ? 1e-1 : 2e-2);
                }
            }
        }
    }

    private static void PackedSequencesMatch(Device device)
    {
        if (device.Type == DeviceType.Cuda && MixedPrecision.TensorCoresUnavailable(device) is not null)
        {
            return;                                                              // packed attention runs on tensor cores
        }

        using var precision = MixedPrecision.Use(device.Type == DeviceType.Cuda ? MatMulPrecision.BFloat16 : MatMulPrecision.Float32);
        int[][] rows = [[37, 50, 20], [70, 30]];
        const int Length = 110;
        var random = new Random(97);
        var tokens = rows.Select(r => r.Select(n => Enumerable.Range(0, n).Select(_ => (float)random.Next(50)).ToArray()).ToArray()).ToArray();
        var coefficients = rows.Select(r => r.Select(n => RandomArray(random, n * 50)).ToArray()).ToArray();
        foreach (bool rotary in new[] { true, false })
        {
            var spec = new DecoderSpec
            {
                Vocabulary = 50, Dim = 128, Layers = 2, Heads = 4, KvHeads = 2, HeadDim = 64, FfDim = 128, MaxPositions = 256,
                Rope = rotary ? new RopeSettings(10000f) : null, LearnedPositions = !rotary,
            };
            using var model = spec.Build(new RandomWeights(98), new DecoderBuildOptions { Device = device });
            model.AddLora(rank: 2, alpha: 4, targets: _ => true, freezeBase: true, random: new Random(99));
            foreach (var adapter in model.Descendants().OfType<Linear>().Select(l => l.Adapter).OfType<LoraAdapter>())
            {
                adapter.B.Load([.. Enumerable.Range(0, adapter.B.Size).Select(i => 0.1f * MathF.Sin(i))]);
            }

            Check(PackedSequences.Supports(model), "the model runs packed batches");
            model.Train();

            // Packed: both rows in one [2, 110] batch (the ends of the rows are padding).
            var packedLogits = new List<float[]>();
            float[] packedGradients;
            float packedLoss;
            using (var scope = new TensorScope())
            {
                var values = new float[rows.Length * Length];
                var weights = new float[rows.Length * Length * 50];
                for (int r = 0; r < rows.Length; r++)
                {
                    int offset = 0;
                    for (int j = 0; j < rows[r].Length; j++)
                    {
                        tokens[r][j].CopyTo(values, r * Length + offset);
                        coefficients[r][j].CopyTo(weights, (r * Length + offset) * 50);
                        offset += rows[r][j];
                    }
                }

                using var packing = PackedSequences.Create(rows, Length, device);
                using (packing.Use())
                {
                    var logits = model.Forward(Tensor.From(values, [rows.Length, Length], device));
                    var loss = (logits * Tensor.From(weights, [rows.Length, Length, 50], device)).Sum();
                    loss.Backward();
                    packedLoss = loss.Item();
                    var all = logits.ToArray();
                    for (int r = 0; r < rows.Length; r++)
                    {
                        int offset = 0;
                        foreach (int n in rows[r])
                        {
                            packedLogits.Add(all[((r * Length + offset) * 50)..((r * Length + offset + n) * 50)]);
                            offset += n;
                        }
                    }
                }

                packedGradients = [.. model.TrainableParameters().SelectMany(p => p.Grad!.ToArray())];
            }

            // Alone: each sequence as its own batch, gradients summed.
            foreach (var parameter in model.TrainableParameters())
            {
                parameter.ZeroGrad();
            }

            var aloneLogits = new List<float[]>();
            float aloneLoss = 0f;
            float[] aloneGradients;
            using (var scope = new TensorScope())
            {
                for (int r = 0; r < rows.Length; r++)
                {
                    for (int j = 0; j < rows[r].Length; j++)
                    {
                        int n = rows[r][j];
                        var logits = model.Forward(Tensor.From(tokens[r][j], [1, n], device));
                        var loss = (logits * Tensor.From(coefficients[r][j], [1, n, 50], device)).Sum();
                        loss.Backward();
                        aloneLoss += loss.Item();
                        aloneLogits.Add(logits.ToArray());
                    }
                }

                aloneGradients = [.. model.TrainableParameters().SelectMany(p => p.Grad!.ToArray())];
            }

            string what = rotary ? "rotary positions" : "learned positions";
            float tolerance = device.Type == DeviceType.Cuda ? 3e-2f : 1e-3f;
            for (int i = 0; i < aloneLogits.Count; i++)
            {
                CloseByNorm(aloneLogits[i], packedLogits[i], tolerance, $"{what}: logits of sequence {i}");
            }

            CloseByNorm([aloneLoss], [packedLoss], tolerance, $"{what}: loss");
            CloseByNorm(aloneGradients, packedGradients, tolerance, $"{what}: adapter gradients");
        }

        // First fit, longest first: 6+4, 5+5, 3+2 in rows of 10.
        var sequences = new[] { 5, 3, 6, 4, 5, 2 }.Select(n => new TrainingSequence(new int[n + 1], new bool[n + 1])).ToList();
        var batches = FineTuner.PackedBatches(sequences, 10, 2, random: null);
        Check(batches.Count == 2 && batches[0].Length == 2 && batches[1].Length == 1, $"packed batches: {batches.Count}");
        var filled = batches.SelectMany(b => b).Select(r => r.Sum(i => sequences[i].Tokens.Length - 1)).ToArray();
        Check(filled.SequenceEqual([10, 10, 5]) && batches.SelectMany(b => b).SelectMany(r => r).Order().SequenceEqual(Enumerable.Range(0, 6)),
            $"rows filled {string.Join(", ", filled)}");
    }

    // ‖expected - actual‖ ≤ tolerance · ‖expected‖ (products in bfloat16 differ element by element, not overall).
    private static void CloseByNorm(float[] expected, float[] actual, float tolerance, string what)
    {
        Check(expected.Length == actual.Length, $"{what}: length {actual.Length}, expected {expected.Length}");
        double difference = 0, norm = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            difference += (double)(expected[i] - actual[i]) * (expected[i] - actual[i]);
            norm += (double)expected[i] * expected[i];
        }

        double error = Math.Sqrt(difference / Math.Max(norm, 1e-30));
        Check(error <= tolerance, $"{what}: relative error {error:G3}");
    }

    private static void GraphTraining(Device device)
    {
        if (device.Type != DeviceType.Cuda || !device.Backend.SupportsGraphs || MixedPrecision.TensorCoresUnavailable(device) is not null)
        {
            return;                                                              // graphs: CUDA; packed attention: tensor cores
        }

        using var precision = MixedPrecision.BFloat16();
        var spec = new DecoderSpec
        {
            Vocabulary = 260, Dim = 128, Layers = 2, Heads = 2, KvHeads = 1, HeadDim = 64, FfDim = 256, MaxPositions = 512,
            Rope = new RopeSettings(10000f), NormEpsilon = 1e-6f, TieEmbeddings = true,
        };
        string folder = WriteChatModel(spec);
        try
        {
            var random = new Random(61);
            var sequences = Enumerable.Range(0, 24).Select(_ =>
            {
                int n = random.Next(20, 120);
                return new TrainingSequence([.. Enumerable.Range(0, n).Select(_ => random.Next(256))], [.. Enumerable.Range(0, n).Select(i => i >= n / 2)]);
            }).ToList();
            (List<float> Losses, float[] Adapters, string Trace) Run(bool graphs)
            {
                using var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
                var options = new FineTuningOptions { Rank = 4, Alpha = 8, LearningRate = 1e-3f, BatchTokens = 256, Seed = 1, CudaGraphs = graphs };
                var losses = new List<float>();
                var trace = new System.Text.StringBuilder();
                FineTuner.Train(model, sequences, null, options, progress: new SynchronousProgress<FineTuningProgress>(p => losses.Add(p.Loss)),
                    trace: line => trace.AppendLine(line));
                return (losses, [.. model.Network.TrainableParameters().SelectMany(p => p.ToArray())], trace.ToString());
            }

            var ordinary = Run(false);
            var replayed = Run(true);
            Check(replayed.Trace.Contains("recorded one training step as a CUDA graph") && replayed.Trace.Contains("replayed"),
                $"the graph was recorded and replayed:\n{replayed.Trace}");
            Check(ordinary.Losses.Count >= 5 && ordinary.Losses.Count == replayed.Losses.Count, $"steps: {ordinary.Losses.Count} and {replayed.Losses.Count}");
            CloseByNorm([.. ordinary.Losses], [.. replayed.Losses], 2e-3f, "losses per step");
            CloseByNorm(ordinary.Adapters, replayed.Adapters, 2e-3f, "trained adapters");
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }

    // Reports on the calling thread (Progress<T> posts to the thread pool, so its reports can arrive late).
    private sealed class SynchronousProgress<T>(Action<T> report) : IProgress<T>
    {
        public void Report(T value) => report(value);
    }
}

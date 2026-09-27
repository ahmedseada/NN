using System.Text.Json.Nodes;
using NeuralSharp;
using NeuralSharp.Generation;
using NeuralSharp.Layers;
using NeuralSharp.Pretrained;

internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] FineTuning =
    [
        ("fine-tuning: chunked token cross-entropy matches the dense loss, its input gradient and a head adapter's gradient", TokenLoss),
        ("memory: a full GPU raises a clear error, or with offloading places tensors in system memory the kernels still use", HostOffload),
        ("fine-tuning: activation checkpointing gives the same loss and gradients (adapters and input)", CheckpointingGradients),
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

        float[] Run(bool chunked, out float[] hiddenGrad, out float[] adapterGrad)
        {
            using var head = new Linear(Dim, Vocabulary, bias: true, device, new Random(82));
            head.AddLora(rank: 2, alpha: 2, targets: _ => true, freezeBase: true, random: new Random(83));
            head.Adapter!.B.Load(adapterB);
            using var scope = new TensorScope();
            var hidden = Tensor.From(hiddenValues, [Rows, Dim], device, requiresGrad: true);
            Tensor loss;
            if (chunked)
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

        var expected = Run(false, out var expectedHidden, out var expectedAdapter);
        var actual = Run(true, out var actualHidden, out var actualAdapter);
        AssertClose(expected, actual, 1e-4f, "loss");
        AssertClose(expectedHidden, actualHidden, 1e-4f, "hidden gradient");
        AssertClose(expectedAdapter, actualAdapter, 1e-4f, "adapter gradient");
    }
}

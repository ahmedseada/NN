// A small GPT: a decoder-only, character-level transformer with causal self-attention that learns to
// continue text. It trains on sentences from a small grammar (or your own text via --corpus), saves the
// model, then generates new text one character at a time.
//
//   dotnet run -c Release --project samples/NeuralSharp.Samples.Transformer                       train, save, generate
//   dotnet run -c Release --project samples/NeuralSharp.Samples.Transformer -- --corpus book.txt  train on your own text
//   dotnet run -c Release --project samples/NeuralSharp.Samples.Transformer -- --predict --input "the old wizard" --temperature 0.8 --length 400
//                                                                                                  generate from the saved model
//   dotnet run -c Release --project samples/NeuralSharp.Samples.Transformer -- --predict --benchmark true
//                                                          compare full recompute, KV cache, CUDA graph and batched sampling
// The Web API sample (NeuralSharp.Samples.GptApi) serves the same model with a browser UI.

using NeuralSharp.Diagnostics;
using NeuralSharp.Samples;
using NeuralSharp.Samples.Gpt;

if (SampleOptions.Parse(args,
        ("corpus", "text file to train on (default: sentences from a built-in grammar)"),
        ("temperature", "sampling temperature: lower is safer, higher is more creative (default 0.7)"),
        ("top-k", "sample only among the k most likely characters (default: all)"),
        ("length", "characters to generate (default 400)"),
        ("samples", "continuations generated together as one batch (default 1)"),
        ("cache", "on|off: incremental decoding with a KV cache (default on)"),
        ("graph", "on|off: record the decoding step as a CUDA graph on GPU (default on)"),
        ("benchmark", "true: compare decoding modes and batch sizes")) is not { } options)
{
    return 0;
}

var device = options.Device;
Console.WriteLine($"Device: {device} - {device.Name}\n");
using var logger = Telemetry.Subscribe(new ConsoleLogger(options.LogLevels));
string modelPath = options.ModelPath("transformer.weights");

CharGpt gpt;
if (options.PredictOnly)
{
    if (!options.RequireModel(modelPath))
    {
        return 1;
    }

    gpt = CharGpt.Load(modelPath, device);
    Console.WriteLine($"Loaded {gpt.Model.ParameterCount:N0} parameters from {modelPath}" +
                      $" (trained {gpt.Config.TrainedEpochs} epochs on {gpt.Config.Corpus}, validation accuracy {gpt.Config.ValidationAccuracy:P1})\n");
}
else
{
    string? corpusPath = options.Get("corpus");
    string corpus = corpusPath is not null ? File.ReadAllText(corpusPath) : Grammar.Corpus(sentences: 6000, seed: 1);
    string vocabulary = new([.. corpus.Distinct().Order()]);
    Console.WriteLine($"Corpus: {corpus.Length:N0} characters, vocabulary of {vocabulary.Length}: \"{vocabulary.Replace("\n", "\\n")}\"");
    Console.WriteLine($"Sample: \"{corpus[..160].Replace("\n", " ")}...\"\n");

    // Small enough to train in a couple of minutes on a CPU; raise Dim / Layers / Context on a GPU.
    gpt = CharGpt.Create(new GptConfig(vocabulary, Context: 64, Dim: 96, Heads: 4, Layers: 3), device);
    var trained = GptTraining.Train(gpt, corpus, options.Epochs ?? 2, options.BatchSize ?? 32, corpusPath ?? "built-in grammar");
    gpt.Save(modelPath, trained);
    Console.WriteLine($"\nSaved the model to {modelPath}\nTest it later with: --predict --input \"the little robot\"\n");
}

using (gpt)
{
    bool On(string name) => options.Get(name, "on") is "on" or "true" or "1";
    var settings = new GenerationSettings(
        Length: int.Parse(options.Get("length", "400")!),
        Temperature: float.Parse(options.Get("temperature", "0.7")!, System.Globalization.CultureInfo.InvariantCulture),
        TopK: int.Parse(options.Get("top-k", "0")!),
        Seed: 6,
        Samples: int.Parse(options.Get("samples", "1")!),
        UseCache: On("cache"),
        UseGraph: On("graph"));

    if (options.Get("benchmark") is "true" or "1")
    {
        Benchmark(gpt, settings);
        return 0;
    }

    string[] prompts = options.Input is { } input ? [input] : ["the little robot ", "every morning, ", "a curious cat "];
    foreach (var (prompt, i) in prompts.Select((p, i) => (p, i)))
    {
        var result = gpt.Generate(prompt.ToLowerInvariant(), settings with { Seed = settings.Seed + i });
        var m = result.Metrics;
        Console.WriteLine($"Prompt \"{prompt}\" (temperature {settings.Temperature}, {m.Mode}):");
        foreach (var sample in result.Samples)
        {
            Console.WriteLine((result.Samples.Count > 1 ? $"  [{sample.Sample + 1}] " : "  ") + (prompt + sample.Text).Replace("\n", "\n  "));
        }

        Console.WriteLine($"  -> {m.TotalTokens} characters in {m.TotalMs:F0} ms ({m.TokensPerSecond:F0}/s, {m.MsPerStep:F2} ms per step), " +
                          $"first token {m.FirstTokenMs:F1} ms, average confidence {m.AverageProbability:P0}, perplexity {m.Perplexity:F2}");
        if (gpt.Config.Corpus == "built-in grammar")
        {
            var (valid, total) = Grammar.Score(string.Join(" ", result.Samples.Select(s => s.Text)));
            Console.WriteLine($"  -> {valid}/{total} generated words are real vocabulary words ({valid / (double)Math.Max(total, 1):P0})");
        }

        Console.WriteLine();
    }
}

return 0;

// Times each decoding mode on the same prompt: characters per second summed over all samples.
static void Benchmark(CharGpt gpt, GenerationSettings settings)
{
    var modes = new (string Name, GenerationSettings Settings)[]
    {
        ("full recompute, 1 sample", settings with { UseCache = false, UseGraph = false, Samples = 1 }),
        ("KV cache, 1 sample", settings with { UseCache = true, UseGraph = false, Samples = 1 }),
        ("KV cache + graph, 1 sample", settings with { UseCache = true, UseGraph = true, Samples = 1 }),
        ("KV cache + graph, 8 samples", settings with { UseCache = true, UseGraph = true, Samples = 8 }),
        ("KV cache + graph, 32 samples", settings with { UseCache = true, UseGraph = true, Samples = 32 }),
    };
    Console.WriteLine($"Benchmark on {gpt.Device} ({gpt.Device.Name}), {settings.Length} characters per sample\n");
    Console.WriteLine("  mode                            chars/s   ms/step   first token   note");
    double? baseline = null;
    foreach (var (name, s) in modes)
    {
        gpt.Generate("the little robot ", s with { Length = 40 });   // warm-up (JIT, memory pool, graph)
        var m = gpt.Generate("the little robot ", s).Metrics;
        baseline ??= m.TokensPerSecond;
        Console.WriteLine($"  {name,-30} {m.TokensPerSecond,8:F0}  {m.MsPerStep,8:F2}  {m.FirstTokenMs,9:F1} ms   {m.TokensPerSecond / baseline:F1}x{(m.GraphNote is { } note ? "  " + note : "")}");
    }
}

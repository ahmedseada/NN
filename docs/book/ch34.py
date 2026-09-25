"""Chapter 34 — GPT Text Generation."""
from gen import *

PART = "VI"


def build():
    return page(
        chapter_open(
            "gpt",
            "A GPT is a decoder-only transformer trained to predict the next token of text; generating means "
            "predicting, sampling and repeating. This project, the repository's <code>NeuralSharp.Samples.Transformer</code> "
            "with the shared <code>CharGpt</code> class, trains a character-level GPT on 333,000 characters of simple "
            "English from a small grammar (or any text file you give it), saves it with its configuration, and "
            "generates text with a KV cache, on-device sampling, CUDA graphs and batched samples.",
            "Model: <code>Embedding</code> → <code>PositionalEncoding</code> → 3 causal <code>TransformerEncoderLayer</code>s → <code>LayerNorm</code> → <code>Linear</code> to the vocabulary (341,309 parameters).",
            "Training data: windows of 64 characters; the target is the same window shifted by one character; <code>SparseCrossEntropy</code> per position.",
            "Result: 89.4% next-character accuracy; every generated word is a real word of the grammar.",
            "Training: 136 s on the book's 4-core CPU; about 16 s on a laptop GeForce RTX 5050.",
            "Generation on the CPU: 230 characters/s by full recompute, 3,429 with the KV cache, 10,081 with 32 samples at once.",
        ),
        h2("34.1 The model and its configuration"),
        snippet("""
            public sealed record GptConfig(string Vocabulary, int Context = 64, int Dim = 96, int Heads = 4, int Layers = 3)
            {
                public int TrainedEpochs { get; init; }
                public double? ValidationLoss { get; init; }
                public double? ValidationAccuracy { get; init; }
                public DateTimeOffset? TrainedAt { get; init; }
                public string? Corpus { get; init; }
                // Save(weightsPath) / Load(weightsPath): JSON next to the weights file
            }

            public static CharGpt Create(GptConfig config, Device device, Random? random = null)
            {
                random ??= new Random(2);
                var model = new Sequential
                {
                    new Embedding(config.Vocabulary.Length, config.Dim, device, random),
                    new PositionalEncoding(config.Context, config.Dim, device),
                };
                for (int layer = 0; layer < config.Layers; layer++)
                    model.Add(new TransformerEncoderLayer(config.Dim, config.Heads, ffDim: 4 * config.Dim,
                                                          dropout: 0.1f, causal: true, device: device, random: random));
                model.Add(new LayerNorm(config.Dim, device: device));
                model.Add(new Linear(config.Dim, config.Vocabulary.Length, device: device, random: random));
                return new CharGpt(config, model, device);
            }
            """, caption="samples/Shared/Gpt/CharGpt.cs (abridged)"),
        para("<code>CharGpt</code>, <code>GptConfig</code>, <code>GptTraining</code> and <code>GenerationSettings</code> are "
             "sample code in <code>samples/Shared/Gpt/CharGpt.cs</code>, built only on the public library API; copy the "
             "file into your own project (or link it) to reuse them. The configuration (vocabulary, sizes, training "
             "record) is saved as JSON next to the weights. Loading "
             "reads the JSON first and builds the matching model, so an inference program never has to repeat the "
             "architecture by hand: a pattern worth copying for any model whose shape is chosen at training time."),
        h2("34.2 Training windows"),
        snippet("""
            public static Dataset Windows(string corpus, GptConfig config, int maxWindows, int seed)
            {
                var index = config.Vocabulary.Select((c, i) => (c, i)).ToDictionary(p => p.c, p => p.i);
                var random = new Random(seed);
                int context = config.Context;
                int windows = Math.Min(maxWindows, Math.Max(1, corpus.Length / 4));
                var features = new float[windows * context];
                var targets = new float[windows * context];
                for (int w = 0; w < windows; w++)
                {
                    int start = random.Next(corpus.Length - context - 1);
                    for (int t = 0; t < context; t++)
                    {
                        features[w * context + t] = index[corpus[start + t]];       // characters t
                        targets[w * context + t] = index[corpus[start + t + 1]];    // the character after each
                    }
                }
                string[] positions = [.. Enumerable.Range(0, context).Select(t => $"t{t}")];
                return Dataset.FromFlat(features, targets, windows, positions, positions);
            }

            // training: AdamW, cosine schedule, clipping; SparseCrossEntropy over [N, 64, V] logits and [N, 64] targets
            var trainer = new Trainer(gpt.Model, optimizer, (logits, next) => Losses.SparseCrossEntropy(logits, next))
            {
                Metrics = { Metric.SparseAccuracy },
                Scheduler = new CosineAnnealing(optimizer, epochs, minLearningRate: 2e-4f),
                MaxGradientNorm = 1f,
            };
            """, caption="Every window teaches 64 predictions at once, thanks to the causal mask"),
        output("""
            Corpus: 333,267 characters, vocabulary of 29: "\\n ,.abcdefghiklmnopqrstuvwxyz"
            Sample: "every morning, my neighbor forgot a wooden boat. the old wizard borrowed a shiny key and then opened a shiny key. last night, our teacher opened a wooden boat. ..."

            Training on cpu (CPU (4 threads, 8-wide SIMD)) | 19,000 samples, 1,000 validation | batch 32, 594 steps/epoch | AdamW lr=0.002 | 341,309 parameters | 4 CPU threads
            Epoch 1/2  loss 0.573884  accuracy 0.7999  val_loss 0.262710  val_accuracy 0.8926  70053.7 ms  271 samples/s  *
            Epoch 2/2  loss 0.266744  accuracy 0.8895  val_loss 0.250425  val_accuracy 0.8938  65444.2 ms  290 samples/s  *
            Finished 2 epochs in 135.51 s | best epoch 2 loss 0.250425
            """, caption="Training output on the CPU"),
        para("A validation loss of 0.25 is far below ln 29 ≈ 3.37, the loss of guessing. The remaining uncertainty "
             "is real: after \"the old \" several nouns are equally valid in this grammar, so no model can reach 100%."),
        h2("34.3 Generating"),
        output("""
            Prompt "the little robot " (temperature 0.7, KV cache):
              the little robot built a tiny garden and then carried the heavy box.
              a quiet student opened the heavy box on the hill. the old wizard sold a shiny key and then watched a shiny key. before dawn, the young pilot sold a tiny garden. ...
              -> 400 characters in 112 ms (3585/s, 0.25 ms per step), first token 12.1 ms, average confidence 91 %, perplexity 1.20
              -> 76/76 generated words are real vocabulary words (100 %)
            """, caption="Generation after training (text shortened)"),
        para("Perplexity 1.20 means the model is, on average, choosing between about 1.2 equally likely characters: "
             "very sure of itself, as a simple grammar allows (glossary <b>Perplexity</b>). The generation loop in "
             "<code>CharGpt.Generate</code> is the one explained in " + ch("generation") + ": prefill the prompt, "
             "then one cached step per character, sampling on the device, reading tokens back in chunks, and "
             "re-reading the last half of the window when the 64-character context is full."),
        snippet("""
            using var gpt = CharGpt.Load("models/transformer.weights", Device.Default);
            var settings = new GenerationSettings(
                Length: 400, Temperature: 0.7f, TopK: 0, Seed: 6,
                Samples: 1, UseCache: true, UseGraph: true);
            var result = gpt.Generate("the little robot ", settings,
                onToken: token => Console.Write(token.Token));            // streams characters as they arrive
            Console.WriteLine($"\\n{result.Metrics.TokensPerSecond:F0} chars/s, perplexity {result.Metrics.Perplexity:F2}");
            """, caption="Using a trained model from your own code"),
        h2("34.4 How fast, and why"),
        output("""
            Benchmark on cpu (CPU (4 threads, 8-wide SIMD)), 400 characters per sample

              mode                            chars/s   ms/step   first token   note
              full recompute, 1 sample            230      4.36        4.1 ms   1.0x
              KV cache, 1 sample                 3429      0.29        2.4 ms   14.9x
              KV cache + graph, 1 sample         4260      0.23        1.8 ms   18.6x  graphs are GPU-only; CPU runs the step directly
              KV cache + graph, 8 samples        6271      1.26        8.7 ms   27.3x  graphs are GPU-only; CPU runs the step directly
              KV cache + graph, 32 samples      10081      3.13       20.4 ms   43.9x  graphs are GPU-only; CPU runs the step directly
            """, caption="--predict --benchmark true on the CPU"),
        para("The cache removes the repeated work (15×); batching samples shares each step's fixed costs (up to 44× in "
             "total characters per second). On the CPU the graph rows differ from the plain cache only by timing noise "
             "and warm-up; on a GPU, recording the step as a CUDA graph removes the launch overhead of the step's "
             "several hundred kernels, which is where most of a small model's GPU time goes."),
        cpugpu("commands",
               """
               dotnet run -c Release --project samples/NeuralSharp.Samples.Transformer -- --cpu
               dotnet run -c Release --project samples/NeuralSharp.Samples.Transformer -- --cpu --predict --input "the old wizard" --temperature 0.8
               dotnet run -c Release --project samples/NeuralSharp.Samples.Transformer -- --cpu --corpus mybook.txt --epochs 5
               """,
               """
               dotnet run -c Release --project samples/NeuralSharp.Samples.Transformer -- --cuda
               dotnet run -c Release --project samples/NeuralSharp.Samples.Transformer -- --cuda --predict --benchmark true
               dotnet run -c Release --project samples/NeuralSharp.Samples.Transformer -- --cuda --predict --samples 8 --top-k 10
               """,
               "The Web API of " + ch("webapi") + " serves the same model with a browser interface and live metrics."),
        h2("34.5 Training on your own text"),
        reftable(["Corpus size", "Suggested configuration", "Notes"], [
            ["~100 KB (a short story)", "Dim 64, 2 layers, context 64, 5–10 epochs", "Learns spelling and common words; expect memorized phrases"],
            ["1–5 MB (a few books)", "Dim 128–192, 4 layers, context 128, GPU", "Plausible sentences in the author's style"],
            ["Larger", "Dim 256+, 6+ layers, context 256; a GPU is essential", "Consider word pieces instead of characters"],
        ], caption="Table 34.1 — Sizing a character GPT"),
        trap("expecting knowledge from a small GPT",
             "<p>A model of a few hundred thousand parameters learns the surface of its corpus (spelling, grammar, style), "
             "not facts or reasoning. It is excellent for text in a fixed format (logs, product names, code-like strings, "
             "templated reports) and for learning how language models work.</p>"),
        practice([
            (1, "Why does one training window give 64 training examples?",
             "The causal mask lets position t see only characters 0…t, so each of the 64 positions is an independent "
             "next-character prediction, all computed in one forward pass."),
            (1, "What does temperature 0.7 do compared with 1.0?",
             "It sharpens the distribution, favouring likely characters: more conservative, more repetitive text (" + ch("generation") + ")."),
            (2, "Train on your own text file and generate with three temperatures (0.5, 0.8, 1.2). Describe the differences.",
             "<code>--corpus file.txt</code>, then <code>--predict --temperature …</code>. Low temperatures repeat frequent phrases; high "
             "temperatures invent words and break grammar; around 0.7–0.9 is usually most readable."),
            (2, "Generate 8 different continuations of the same prompt in one call.",
             "<code>--samples 8</code>, or <code>GenerationSettings(Samples: 8, …)</code>; the result holds 8 sequences, generated as one batch."),
            (3, "Double the model (Dim 192, 6 layers, context 128) and train it on the GPU. What changes in the code?",
             "Only the <code>GptConfig</code> passed to <code>CharGpt.Create</code>; the saved JSON records the new sizes, so "
             "<code>CharGpt.Load</code> and the Web API pick them up without changes."),
        ], PART),
        footer("GPT", "Language model", "Character-level model", "Next-token prediction", "Causal mask",
               "Perplexity", "Temperature", "KV cache", "Context length"),
    )

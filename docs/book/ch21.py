"""Chapter 21 — Fast Generation: KV Cache, Sampling and CUDA Graphs."""
from gen import *

PART = "IV"


def cache_svg():
    w, h = 470, 150
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">']
    p.append(svg_text(110, 16, "without cache: step t recomputes t positions", 8.4, "#b3261e"))
    for t in range(5):
        for k in range(t + 1):
            p.append(f'<rect x="{30 + k * 18}" y="{26 + t * 22}" width="16" height="16" rx="2" fill="#fdecea" stroke="#b3261e" stroke-width="0.7"/>')
        p.append(svg_text(20, 38 + t * 22, f"t{t + 1}", 7.4, "#56606a", "end"))
    p.append(svg_text(355, 16, "with KV cache: step t computes 1 position", 8.4, "#0f6b5c"))
    for t in range(5):
        for k in range(t + 1):
            new = k == t
            p.append(f'<rect x="{270 + k * 18}" y="{26 + t * 22}" width="16" height="16" rx="2" fill="{"#0f6b5c" if new else "#e6f2ef"}" stroke="#0f6b5c" stroke-width="0.7"/>')
    p.append(svg_text(355, 146, "light cells: keys and values read from the cache", 7.4, "#56606a"))
    p.append("</svg>")
    return "".join(p)


def build():
    return page(
        chapter_open(
            "generation",
            "A language model generates text one token at a time, each new token depending on all previous ones. Done "
            "naively, every step reruns the model over the whole text so far. NeuralSharp provides the three standard "
            "remedies: a key–value cache so each step only processes the newest token, a sampler that picks tokens on "
            "the device, and compute graphs that replay a whole decoding step as one GPU launch. This chapter explains "
            "each with a measured example.",
            "<code>DecodingContext</code> holds one <code>KeyValueCache</code> per attention layer; <code>model.ForwardCached(ids, context)</code> processes only new positions.",
            "<code>TokenSampler</code> samples on the device with temperature and top-k, reproducibly (same seed ⇒ same tokens on CPU and GPU).",
            "<code>context.CaptureStep(step)</code> records a decoding step as a <code>ComputeGraph</code>; <code>ReplayStep</code> runs it again. On the GPU this is a CUDA Graph; on the CPU it simply re-runs the step.",
            "Measured on the CPU: 260 tokens/s without cache, 2,690 with it, same tokens.",
            "All of this is inference only: call it under <code>Autograd.NoGrad()</code> with the model in evaluation mode.",
        ),
        h2("21.1 Why a cache"),
        diagram("Figure 21.1 — Work per generated token", cache_svg(),
                "Attention needs every earlier position's key and value. Keeping them removes all repeated work."),
        para("In a causal transformer (" + ch("attention") + "), the keys and values of earlier positions never change "
             "once computed. The cache stores them per attention layer, shaped <code>[batch·heads, capacity, headDim]</code>. "
             "The first call (the <b>prefill</b>) processes the whole prompt at once and fills the cache; every later call "
             "processes one new position per sequence and appends its key and value. Layers implementing "
             "<code>ICachedModule</code> (<code>Sequential</code>, <code>MultiHeadAttention</code>, "
             "<code>TransformerEncoderLayer</code>, <code>PositionalEncoding</code>) take this path; other layers "
             "(embeddings, norms, linear heads) simply run on the new positions."),
        reftable(["Type", "Members"], [
            ["<code>DecodingContext(device, batch, capacity)</code>", "<code>Length</code>, <code>Capacity</code>, <code>Batch</code>, <code>Reset()</code>, <code>CaptureStep(action)</code>, <code>ReplayStep(graph)</code>, <code>Dispose()</code>"],
            ["<code>Sequential.ForwardCached(input, context)</code>", "Runs new positions <code>[batch, newSteps]</code> through the model using and extending the caches"],
            ["<code>TokenSampler(device, rows, vocabulary, maxSteps, historyCapacity)</code>", "<code>Temperature</code>, <code>TopK</code>, <code>TopP</code>, <code>MinP</code>, penalties (" + ch("textgen") + "), <code>Seed</code>, <code>SetHistory</code>, <code>Sample(logits)</code>, <code>Ids</code>, <code>Read(from, to)</code>, <code>Reset()</code>"],
            ["<code>SampledToken</code>", "<code>Id</code>, <code>Probability</code>, <code>Entropy</code> (bits), <code>Alternatives</code> (top 5)"],
            ["<code>ComputeGraph</code>", "<code>Capture(device, action)</code>, <code>Replay()</code>, <code>IsRecorded</code>, <code>FailureReason</code>"],
        ], caption="Table 21.1 — The generation API"),
        h2("21.2 The three modes, measured"),
        mex("the same 100 tokens three ways",
            "A small untrained GPT (2 blocks, width 64, 65-token vocabulary). All three runs use the same sampler seed, "
            "so they must produce the same tokens.",
            """
            const int Vocab = 65, Dim = 64, Context = 128, Steps = 100;
            var r = new Random(2);
            using var gpt = new Sequential
            {
                new Embedding(Vocab, Dim, random: r),
                new PositionalEncoding(Context, Dim),
                new TransformerEncoderLayer(Dim, heads: 4, dropout: 0f, causal: true, random: r),
                new TransformerEncoderLayer(Dim, heads: 4, dropout: 0f, causal: true, random: r),
                new LayerNorm(Dim),
                new Linear(Dim, Vocab, random: r),
            };
            gpt.Eval();
            int[] prompt = [5, 17, 42, 8];

            // (a) recompute the whole text every step
            int[] FullRecompute()
            {
                var ids = new List<int>(prompt);
                using var sampler = new TokenSampler(Device.Default, rows: 1, vocabulary: Vocab, maxSteps: Steps)
                    { Seed = 7, Temperature = 0.8f };
                for (int step = 0; step < Steps; step++)
                {
                    using var scope = new TensorScope();
                    using var noGrad = Autograd.NoGrad();
                    var window = Tensor.From([.. ids.Select(i => (float)i)], [1, ids.Count]);
                    sampler.Sample(gpt.Forward(window));                  // samples from the last position
                    ids.Add((int)sampler.Ids.Item());                     // host read every step
                }
                return [.. ids.Skip(prompt.Length)];
            }

            // (b) KV cache; (c) the same with the step recorded as a graph
            int[] Cached(bool graph)
            {
                using var context = new DecodingContext(Device.Default, batch: 1, capacity: Context);
                using var sampler = new TokenSampler(Device.Default, rows: 1, vocabulary: Vocab, maxSteps: Steps)
                    { Seed = 7, Temperature = 0.8f };
                using var noGrad = Autograd.NoGrad();
                using (var scope = new TensorScope())                    // prefill: the whole prompt at once
                    sampler.Sample(gpt.ForwardCached(Tensor.From([.. prompt.Select(i => (float)i)], [1, prompt.Length]), context));

                void Step() => sampler.Sample(gpt.ForwardCached(sampler.Ids.Reshape(1, 1), context));
                using var recorded = graph ? context.CaptureStep(Step) : null;
                for (int step = 1; step < Steps; step++)
                {
                    if (recorded is not null) context.ReplayStep(recorded);
                    else { using var scope = new TensorScope(); Step(); }
                }
                return [.. sampler.Read(0, Steps).Select(s => s[0].Id)];  // ONE host read at the end
            }

            foreach (var (name, run) in new (string, Func<int[]>)[] {
                ("full recompute", FullRecompute), ("KV cache", () => Cached(false)), ("KV cache + graph", () => Cached(true)) })
            {
                run();                                                    // warm-up
                var clock = Stopwatch.StartNew();
                var tokens = run();
                Console.WriteLine($"{name,-17} {Steps / clock.Elapsed.TotalSeconds,8:N0} tokens/s   " +
                                  $"first ids: {string.Join(" ", tokens.Take(12))}");
            }
            """,
            out="""
            full recompute         260 tokens/s   first ids: 7 17 17 48 27 50 17 4 14 58 3 27
            KV cache             2,690 tokens/s   first ids: 7 17 17 48 27 50 17 4 14 58 3 27
            KV cache + graph     2,425 tokens/s   first ids: 7 17 17 48 27 50 17 4 14 58 3 27
            """,
            after="Timed after a warm-up run on the book's 4-core CPU. The cache gives a 10× speed-up with identical "
                  "output. On the CPU a graph has nothing to save, so it simply re-runs the step (the small difference "
                  "is measurement noise); on the GPU it is where the next large gain comes from."),
        h2("21.3 Sampling on the device"),
        para("<code>TokenSampler.Sample(logits)</code> takes the last position's scores for every row, applies the "
             "temperature (divide the scores; below 1 is more conservative, above 1 more adventurous), keeps only the "
             "<code>TopK</code> most likely tokens if set, and draws one token per row from the resulting distribution "
             "(glossary <b>Temperature</b>, <b>Top-k sampling</b>). The chosen ids stay in <code>sampler.Ids</code> on "
             "the device and feed the next step directly, so the GPU never waits for the CPU between tokens. Statistics "
             "for every step (probability, entropy, five alternatives) are written to a device buffer and read in one "
             "call with <code>Read(from, to)</code>."),
        reftable(["Setting", "Effect"], [
            ["<code>Temperature = 1</code>", "Sample from the model's distribution as is"],
            ["<code>Temperature = 0.5–0.8</code>", "More likely tokens favoured; text more coherent, less varied"],
            ["<code>Temperature = 1.2–1.5</code>", "More surprising text, more mistakes"],
            ["<code>TopK = 1</code>", "Greedy: always the most likely token (deterministic)"],
            ["<code>TopK = 10–50</code>", "Never pick very unlikely tokens"],
            ["<code>Seed</code>", "Same seed, same logits ⇒ same tokens, on CPU and GPU alike"],
        ], caption="Table 21.2 — Sampler settings"),
        h2("21.4 CUDA graphs"),
        para("One decoding step of a transformer is a few hundred tiny kernels. On a GPU, launching each one costs a few "
             "microseconds of CPU time, which can exceed the kernel's own run time; the GPU then sits idle waiting for "
             "launches. <code>context.CaptureStep(step)</code> records the whole step once as a CUDA Graph "
             "(glossary <b>CUDA graph</b>); <code>context.ReplayStep(graph)</code> launches all of it with a single call."),
        deriv("What makes a step replayable", [
            "Everything that changes from step to step lives in device memory: the position counter, the causal mask and "
            "the cache write offset are computed on the device from <code>context.Position</code>, and the next input is "
            "<code>sampler.Ids</code>.",
            "The step reads nothing back to the host and uploads nothing; the sampler's step counter is on the device too.",
            "All tensors the step uses (weights, caches, sampler buffers) stay alive and on the same device while the graph exists.",
            "Kernel sizes are fixed at recording: batch and one new position per step. The prefill (many positions) runs normally.",
        ]),
        cpugpu("the same generation code on both devices",
               """
               Device.Default = Device.Cpu;
               // CaptureStep returns a graph with IsRecorded == false:
               // ReplayStep simply calls the step again
               """,
               """
               Device.Default = Device.Cuda();
               using var graph = context.CaptureStep(Step);
               Console.WriteLine(graph.IsRecorded ? "CUDA graph" : $"fallback: {graph.FailureReason}");
               """,
               "If recording fails for any reason (an old driver, a step that reads back to the host), "
               "<code>FailureReason</code> says why and replays fall back to normal launches: the output is the same."),
        h2("21.5 Batches and long texts"),
        reftable(["Situation", "What to do"], [
            ["Several continuations of one prompt", "<code>DecodingContext(device, batch: n, …)</code> and <code>TokenSampler(…, rows: n, …)</code>; each row samples independently. On the GPU n rows cost little more than one."],
            ["Different prompts in one batch", "Left-pad or process separately; one context holds sequences of equal length"],
            ["Text longer than the context", "When <code>context.Length</code> reaches <code>Capacity</code>, <code>Reset()</code> and prefill again with the last part of the text (the GPT of " + ch("gpt") + " keeps the last half)"],
            ["Streaming to a user", "Call <code>sampler.Read</code> every N steps (N = 8–32) and send those tokens; each read is one synchronization"],
        ], caption="Table 21.3 — Practical generation patterns"),
        trap("dropout or training mode during generation",
             "<p>Call <code>model.Eval()</code> before generating. In training mode dropout changes every step's output "
             "and cached keys no longer match what the full model would compute.</p>"),
        trap("capturing the prefill",
             "<p>Record only the one-token step. The prefill has a different number of positions (the prompt length), and "
             "a recorded graph always repeats exactly the sizes it saw.</p>"),
        practice([
            (1, "Why must the prefill and the first recorded step be separate?",
             "The prefill processes the whole prompt (many positions) and the graph step exactly one; kernel sizes are "
             "frozen at recording, so a graph can only replay the one-position step."),
            (1, "Which sampler settings make generation deterministic?",
             "<code>TopK = 1</code> (greedy), or any settings with a fixed <code>Seed</code>, which reproduce the same tokens."),
            (2, "Estimate the work saved by the cache for 1,000 generated tokens with a 1,000-token context.",
             "Without it, step t processes t positions: about 1,000²/2 = 500,000 position-passes; with it, 1,000. The "
             "attention inside each step still reads all cached positions, so the real speed-up is smaller but grows with length."),
            (2, "Generate 8 continuations of one prompt in one batch.",
             "Create the context and sampler with batch/rows 8, prefill with the prompt repeated in 8 rows "
             "(<code>[8, promptLength]</code>), feed <code>sampler.Ids.Reshape(8, 1)</code> each step, and read "
             "<code>sampler.Read(0, steps)[step][row]</code>."),
            (3, "Stream tokens every 16 steps while using a graph.",
             "Replay the graph each step and every 16 steps call <code>sampler.Read(emitted, produced)</code>, output those "
             "tokens and set <code>emitted = produced</code>; " + ch("gpt") + " and the Web API of " + ch("webapi") + " do exactly this."),
        ], PART),
        footer("KV cache", "Prefill", "Autoregressive generation", "Temperature", "Top-k sampling", "Entropy",
               "CUDA graph", "Kernel launch", "Greedy decoding"),
    )

"""Chapter 11 — Recurrent Layers: LSTM and GRU."""
from gen import *

PART = "II"


def unroll_svg():
    w, h = 470, 140
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">']
    for t in range(4):
        x = 40 + t * 105
        p.append(f'<rect x="{x}" y="50" width="64" height="36" rx="5" fill="#e6f2ef" stroke="#0f6b5c" stroke-width="1.2"/>')
        p.append(svg_text(x + 32, 72, "cell", 9, "#0f6b5c"))
        p.append(f'<line x1="{x + 32}" y1="118" x2="{x + 32}" y2="88" stroke="#56606a" stroke-width="1.1"/>')
        p.append(f'<polygon points="{x + 32},{86} {x + 28},{94} {x + 36},{94}" fill="#56606a"/>')
        p.append(svg_text(x + 32, 132, f"x{t + 1}", 8.6, "#a15c00"))
        p.append(f'<line x1="{x + 32}" y1="50" x2="{x + 32}" y2="24" stroke="#56606a" stroke-width="1.1"/>')
        p.append(f'<polygon points="{x + 32},{22} {x + 28},{30} {x + 36},{30}" fill="#56606a"/>')
        p.append(svg_text(x + 32, 16, f"h{t + 1}", 8.6, "#0f6b5c"))
        if t < 3:
            p.append(f'<line x1="{x + 64}" y1="68" x2="{x + 103}" y2="68" stroke="#0f6b5c" stroke-width="1.4"/>')
            p.append(f'<polygon points="{x + 105},68 {x + 97},64 {x + 97},72" fill="#0f6b5c"/>')
            p.append(svg_text(x + 84, 62, "state", 7.2, "#0f6b5c"))
    p.append(svg_text(450, 72, "…", 12, "#56606a"))
    p.append("</svg>")
    return "".join(p)


def build():
    return page(
        chapter_open(
            "recurrent",
            "A recurrent layer reads a sequence one step at a time and carries a hidden state from each step to the "
            "next, so what it outputs at step t can depend on everything before t. LSTM and GRU add gates that "
            "decide what to keep and what to forget, which lets them carry information across many steps. This "
            "chapter covers both layers, their shapes, stacking, and when to prefer them over attention.",
            "Input <code>[N, T, F]</code> (batch, time steps, features per step).",
            "<code>LSTM(F, H)</code> / <code>GRU(F, H)</code> return the last state <code>[N, H]</code>; with <code>returnSequences: true</code> every state <code>[N, T, H]</code>.",
            "Stack layers by setting <code>returnSequences: true</code> on every layer except the last.",
            "GRU has 3 gates (¾ of the LSTM's parameters) and often trains as well or faster.",
            "Steps run one after another, so long sequences are slower than with attention; the GPU helps most with large batches and hidden sizes.",
        ),
        h2("11.1 Reading a sequence step by step"),
        diagram("Figure 11.1 — A recurrent layer unrolled over time", unroll_svg(),
                "The same cell (same weights) is applied at every step; the state carries information forward."),
        para("At each step the cell combines the current input x<sub>t</sub> with the previous state and produces "
             "the new state h<sub>t</sub>. The <b>LSTM</b> keeps a second, separate cell state and uses four gates "
             "(input, forget, candidate, output); the <b>GRU</b> uses three (reset, update, candidate) and one state. "
             "The gates are sigmoids between 0 and 1 that scale how much of the old state and the new input pass "
             "through (glossary <b>Gate</b>; the Activation Functions volume covers sigmoid and tanh in the gate role)."),
        reftable(["", "LSTM", "GRU"], [
            ["Constructor", "<code>LSTM(inputSize, hiddenSize, returnSequences = false, device, random)</code>", "<code>GRU(inputSize, hiddenSize, returnSequences = false, device, random)</code>"],
            ["Parameters", "4·H·(F + H + 1)", "3·H·(F + H + 1)"],
            ["State", "hidden h and cell c", "hidden h"],
            ["Special initialization", "forget-gate bias starts at 1 (remember by default)", "—"],
            ["Output", "<code>[N, H]</code> or <code>[N, T, H]</code>", "same"],
        ], caption="Table 11.1 — The two recurrent layers"),
        mex("shapes and parameter counts", None,
            """
            using var scope = new TensorScope();
            var sequences = Tensor.Zeros([32, 20, 8]);                 // 32 sequences, 20 steps, 8 features
            using var last = new LSTM(8, 16);
            using var all  = new LSTM(8, 16, returnSequences: true);
            using var gru  = new GRU(8, 16);

            Console.WriteLine($"{last}: [{string.Join(", ", last.Forward(sequences).Shape.ToArray())}]  {last.ParameterCount} params");
            Console.WriteLine($"{all}: [{string.Join(", ", all.Forward(sequences).Shape.ToArray())}]");
            Console.WriteLine($"{gru}: [{string.Join(", ", gru.Forward(sequences).Shape.ToArray())}]  {gru.ParameterCount} params");
            """,
            out="""
            LSTM(8 -> 16): [32, 16]  1600 params
            LSTM(8 -> 16, sequences): [32, 20, 16]
            GRU(8 -> 16): [32, 16]  1200 params
            """,
            after="4·16·(8 + 16 + 1) = 1,600 and 3·16·(8 + 16 + 1) = 1,200. The input projection of all steps is "
                  "computed at once as one large matrix product; only the hidden-to-hidden part runs step by step."),
        h2("11.2 Memory across steps: a test"),
        mex("remember the first number of twenty",
            "Each sample is 20 random numbers; the target is the first one. To answer, the layer must carry the "
            "first value through 19 more steps. A model that forgets scores the variance of the target, about 0.33.",
            """
            var rng = new Random(0);
            const int N = 512, T = 20;
            var xs = new float[N * T];
            var ys = new float[N];
            for (int i = 0; i < N; i++)
            {
                for (int t = 0; t < T; t++) xs[i * T + t] = rng.NextSingle() * 2 - 1;
                ys[i] = xs[i * T];                                     // the first value
            }
            using var x = Tensor.From(xs, [N, T, 1]);
            using var y = Tensor.From(ys, [N, 1]);

            foreach (var (name, make) in new (string, Func<Random, Module>)[] {
                ("LSTM", r => new LSTM(1, 32, random: r)), ("GRU", r => new GRU(1, 32, random: r)) })
            {
                var r = new Random(1);
                using var model = new Sequential { make(r), new Linear(32, 1, random: r) };
                using var opt = new Adam(model.Parameters(), 0.01f);
                for (int step = 1; step <= 300; step++)
                {
                    using var scope = new TensorScope();
                    var loss = Losses.MeanSquaredError(model.Forward(x), y);
                    opt.ZeroGrad(); loss.Backward(); opt.Step();
                    if (step % 100 == 0) Console.Write($"{step}: {loss.Item():F4}  ");
                }
                Console.WriteLine($"({name})");
            }
            """,
            out="""
            100: 0.3055  200: 0.0722  300: 0.0291  (LSTM)
            100: 0.0954  200: 0.0009  300: 0.0005  (GRU)
            """,
            after="Both learn to carry the value; on this task the GRU learns faster. On the 4-core test machine the "
                  "300 steps took 7.2 s (LSTM) and 5.1 s (GRU) on the CPU."),
        h2("11.3 Stacking and heads"),
        cpugpu("a two-layer LSTM classifier for token sequences",
               """
               Device.Default = Device.Cpu;
               var r = new Random(1);
               using var model = new Sequential
               {
                   new Embedding(vocabularySize, 64, random: r),               // [N, T] -> [N, T, 64]
                   new LSTM(64, 128, returnSequences: true, random: r),        // -> [N, T, 128]
                   new Dropout(0.2f, r),
                   new LSTM(128, 128, random: r),                              // -> [N, 128] (last state)
                   new Linear(128, classes, random: r),                        // -> [N, classes]
               };
               """,
               """
               Device.Default = Device.Cuda();
               // same model. Each time step is a few small kernels, so the GPU pays off
               // with batches of 64+ sequences and hidden sizes of 128+
               """),
        reftable(["Task", "Recurrent output", "Head"], [
            ["Classify a whole sequence (sentiment, activity)", "last state <code>[N, H]</code>", "<code>Linear(H, K)</code>"],
            ["Predict the next value of a time series", "last state <code>[N, H]</code>", "<code>Linear(H, 1)</code> (or H → horizon)"],
            ["Label every step (tagging)", "all states <code>[N, T, H]</code>", "<code>Linear(H, K)</code> applied per step"],
            ["Pool the whole sequence", "all states <code>[N, T, H]</code>", "<code>Lambda(x =&gt; x.Mean(1))</code> then <code>Linear</code>"],
        ], caption="Table 11.2 — Using the recurrent output"),
        trap("forgetting returnSequences when stacking",
             "<p>A recurrent layer needs <code>[N, T, F]</code> input. If the layer before it returns only the last state "
             "(<code>[N, H]</code>), the second layer throws <i>LSTM expects [batch, time, 128], got [32, 128]</i>. Every recurrent "
             "layer except the last needs <code>returnSequences: true</code>.</p>"),
        trap("unscaled time series",
             "<p>LSTM and GRU gates saturate on large inputs. Scale each feature (" + ch("data") + ") so values are "
             "roughly within [−3, 3] before training; predict scaled targets and unscale the predictions.</p>"),
        h2("11.4 Recurrent layers or attention?"),
        reftable(["Consideration", "LSTM / GRU", "Transformer (" + ch("attention") + ")"], [
            ["Speed on long sequences", "T sequential steps", "all positions at once"],
            ["Memory for long sequences", "linear in T", "grows with T² (attention weights)"],
            ["Small data", "often better (fewer parameters, built-in order)", "needs more data"],
            ["Streaming, one step at a time", "natural", "possible with a KV cache (" + ch("generation") + ")"],
            ["Typical uses in this book", "sentiment, time series (" + ch("sentiment") + ", " + ch("timeseries") + ")", "text generation (" + ch("gpt") + ")"],
        ], caption="Table 11.3 — Choosing a sequence model"),
        practice([
            (1, "How many parameters does <code>GRU(32, 64)</code> have?",
             "3·64·(32 + 64 + 1) = 18,624."),
            (1, "What must change to feed an LSTM with sequences of 50 steps of 3 sensor readings each?",
             "Nothing but the sizes: input <code>[N, 50, 3]</code> and <code>new LSTM(3, H)</code>."),
            (2, "Make the memory test harder (T = 60). Does the LSTM still learn within 300 steps? What helps?",
             "In our run both still learned: after 300 steps the loss was 0.0102 (LSTM) and 0.0138 (GRU), against 0.33 for a "
             "model that forgets, but the GRU now started more slowly (0.20 at step 100) and each step took about three times "
             "longer (60 sequential steps instead of 20). If a model stalls, train longer, or give it more hidden units."),
            (2, "Label every time step of a sequence with one of 5 tags.",
             "Use <code>returnSequences: true</code> on the last recurrent layer and follow it with <code>Linear(H, 5)</code>, "
             "which applies per step and gives <code>[N, T, 5]</code>. Train with <code>SparseCrossEntropy</code> on tags of "
             "shape <code>[N, T]</code> (" + ch("losses") + ")."),
            (3, "Compare an LSTM, a GRU and a bag-of-words model on a sentence task where word order matters (for example \"not good\" versus \"good\").",
             "The bag-of-words mean cannot tell \"not good\" from \"good not\"; the recurrent models can. "
             + ch("sentiment") + " runs exactly this comparison, including a transformer."),
        ], PART),
        footer("Recurrent layer", "Hidden state", "LSTM", "GRU", "Gate", "Cell state", "Sequence", "Time step",
               "Return sequences", "Vanishing gradient"),
    )

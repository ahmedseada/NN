"""Chapter 12 — Attention and Transformers."""
from gen import *

PART = "II"


def block_svg():
    w, h = 470, 200
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">']

    def box(x, y, bw, label, fill="#ffffff"):
        p.append(f'<rect x="{x}" y="{y}" width="{bw}" height="24" rx="4" fill="{fill}" stroke="#0f6b5c" stroke-width="1.1"/>')
        p.append(svg_text(x + bw / 2, y + 16, label, 8.4, "#0f6b5c"))

    def up(x, y1, y2):
        p.append(f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2 + 2}" stroke="#56606a" stroke-width="1.1"/>')
        p.append(f'<polygon points="{x},{y2} {x - 4},{y2 + 7} {x + 4},{y2 + 7}" fill="#56606a"/>')

    cx = 160
    box(cx - 60, 170, 120, "x  [N, T, D]", "#e6f2ef")
    up(cx, 170, 146)
    box(cx - 60, 122, 120, "LayerNorm")
    up(cx, 122, 98)
    box(cx - 60, 74, 120, "MultiHeadAttention")
    up(cx, 74, 50)
    p.append(f'<circle cx="{cx}" cy="38" r="10" fill="#fff4e2" stroke="#a15c00"/>')
    p.append(svg_text(cx, 42, "+", 11, "#a15c00"))
    p.append(f'<path d="M {cx - 60} 182 C {cx - 95} 182, {cx - 95} 38, {cx - 12} 38" fill="none" stroke="#a15c00" stroke-width="1.1"/>')
    p.append(svg_text(cx - 98, 110, "residual", 7.4, "#a15c00", "end"))
    cx2 = 350
    box(cx2 - 60, 170, 120, "x′", "#e6f2ef")
    up(cx2, 170, 146)
    box(cx2 - 60, 122, 120, "LayerNorm")
    up(cx2, 122, 98)
    box(cx2 - 60, 74, 120, "Linear D→4D, GELU, 4D→D")
    up(cx2, 74, 50)
    p.append(f'<circle cx="{cx2}" cy="38" r="10" fill="#fff4e2" stroke="#a15c00"/>')
    p.append(svg_text(cx2, 42, "+", 11, "#a15c00"))
    p.append(f'<path d="M {cx2 + 60} 182 C {cx2 + 95} 182, {cx2 + 95} 38, {cx2 + 12} 38" fill="none" stroke="#a15c00" stroke-width="1.1"/>')
    p.append(f'<line x1="{170}" y1="30" x2="{290}" y2="182" stroke="#b7c3c8" stroke-width="1" stroke-dasharray="3 3"/>')
    p.append(svg_text(160, 14, "1. attention sub-block", 8.4, "#1a1f23"))
    p.append(svg_text(350, 14, "2. feed-forward sub-block", 8.4, "#1a1f23"))
    p.append("</svg>")
    return "".join(p)


def build():
    return page(
        chapter_open(
            "attention",
            "Attention lets every position of a sequence look directly at every other position and take a weighted "
            "mix of what it finds, with weights computed from the data. Transformers stack attention with small "
            "feed-forward networks, normalization and residual connections; they are the architecture behind modern "
            "language models. This chapter covers <code>MultiHeadAttention</code>, <code>TransformerEncoderLayer</code> "
            "and <code>PositionalEncoding</code>, the causal mask, and how to assemble classifiers and GPT-style models.",
            "<code>MultiHeadAttention(dim, heads, causal, dropout)</code>: <code>[N, T, D]</code> → <code>[N, T, D]</code>; D must be divisible by heads.",
            "<code>TransformerEncoderLayer(dim, heads, ffDim = 4·dim, dropout = 0.1, causal)</code>: one pre-norm block with two residual sub-blocks.",
            "<code>PositionalEncoding(maxLength, dim)</code> adds position information; attention itself ignores order.",
            "<code>causal: true</code> hides future positions: required for next-token prediction (GPT).",
            "Under <code>NoGrad</code>, attention and LayerNorm switch to fused inference kernels automatically.",
        ),
        h2("12.1 Attention in one paragraph"),
        para("For every position the layer computes three vectors from its input: a <b>query</b> (what am I looking "
             "for?), a <b>key</b> (what do I contain?) and a <b>value</b> (what do I pass on?). The score between two "
             "positions is the dot product of one's query with the other's key, scaled by 1/√d; a softmax over each "
             "row turns scores into weights, and the output is the weighted sum of the values (glossary <b>Scaled "
             "dot-product attention</b>). <b>Multi-head</b> attention runs several of these in parallel on slices of "
             "the vector, so different heads can attend to different things, then mixes the results with a final "
             "<code>Linear</code>."),
        reftable(["Step", "Shape (N batch, T positions, D width, H heads, d = D/H)"], [
            ["Input", "<code>[N, T, D]</code>"],
            ["One <code>Linear(D, 3D)</code> gives queries, keys and values", "<code>[N, T, 3D]</code>"],
            ["Split into heads", "3 × <code>[N·H, T, d]</code>"],
            ["Scores Q·Kᵀ / √d (+ causal mask), softmax per row", "<code>[N·H, T, T]</code>"],
            ["Weights · V", "<code>[N·H, T, d]</code>"],
            ["Merge heads, <code>Linear(D, D)</code>", "<code>[N, T, D]</code>"],
        ], caption="Table 12.1 — Inside MultiHeadAttention"),
        para("<code>MultiHeadAttention(16, 4)</code> therefore has 16·48 + 48 + 16·16 + 16 = 1,088 parameters, "
             "whatever the sequence length. The T×T score matrix is why attention's memory grows with the square of "
             "the sequence length."),
        h2("12.2 The causal mask"),
        para("A model that predicts the next token must not see the future. With <code>causal: true</code> a mask "
             "adds a large negative number to every score above the diagonal before the softmax, so position t "
             "attends only to positions 0…t."),
        mex("changing the last position: who notices?",
            "Two inputs that differ only at the last of five positions. With full attention the first position's "
            "output changes; with causal attention it cannot.",
            """
            var r = new Random(2);
            using var full   = new MultiHeadAttention(dim: 16, heads: 4, random: r);
            using var causal = new MultiHeadAttention(dim: 16, heads: 4, causal: true, random: r);
            using var scope = new TensorScope();

            var a = Tensor.Normal([1, 5, 16], random: new Random(1));
            var values = a.ToArray();
            for (int i = 4 * 16; i < 5 * 16; i++) values[i] += 1f;      // alter position 4 only
            var b = Tensor.From(values, [1, 5, 16]);

            float FirstPositionChange(Module m)
            {
                using var noGrad = Autograd.NoGrad();
                return (m.Forward(a) - m.Forward(b)).Narrow(1, 0, 1).Abs().Sum().Item();
            }
            Console.WriteLine($"position 0 changes by {FirstPositionChange(full):F4} (full) " +
                              $"and {FirstPositionChange(causal):F4} (causal)");
            """,
            out="""
            position 0 changes by 0.8457 (full) and 0.0000 (causal)
            """),
        h2("12.3 The transformer block"),
        diagram("Figure 12.1 — TransformerEncoderLayer (pre-norm)", block_svg(),
                "x′ = x + Attention(LayerNorm(x)); output = x′ + FeedForward(LayerNorm(x′)). Dropout (if any) is applied to each sub-block's output."),
        reftable(["Argument", "Default", "Meaning"], [
            ["<code>dim</code>", "—", "Model width D (the embedding size)"],
            ["<code>heads</code>", "—", "Attention heads; D must be divisible by it (4–8 for small models)"],
            ["<code>ffDim</code>", "4·dim", "Hidden width of the feed-forward sub-block"],
            ["<code>dropout</code>", "0.1", "Dropout on attention weights and sub-block outputs"],
            ["<code>causal</code>", "false", "Mask future positions (true for GPT-style models)"],
            ["<code>device</code>, <code>random</code>", "Default, Shared", "Placement and initialization"],
        ], caption="Table 12.2 — new TransformerEncoderLayer(dim, heads, ffDim, dropout, causal, device, random)"),
        defbox("PositionalEncoding",
               "<p>Attention treats its input as a set: shuffling the positions shuffles the outputs the same way. "
               "<code>PositionalEncoding(maxLength, dim)</code> adds a fixed pattern of sines and cosines of different "
               "frequencies to each position, so the model can tell positions apart (glossary <b>Positional "
               "encoding</b>; sines and cosines are in the Pre-Calc volume). It has no parameters; sequences longer "
               "than <code>maxLength</code> throw.</p>"),
        h2("12.4 Assembling models"),
        mex("a small GPT: embeddings, two causal blocks, a head",
            "The same pattern the " + ch("gpt") + " model uses, at a size that trains on a laptop in minutes.",
            """
            const int Vocab = 65, Dim = 64, Context = 128;
            var r = new Random(2);
            using var gpt = new Sequential
            {
                new Embedding(Vocab, Dim, random: r),                   // [N, T] ids -> [N, T, 64]
                new PositionalEncoding(Context, Dim),
                new TransformerEncoderLayer(Dim, heads: 4, dropout: 0.1f, causal: true, random: r),
                new TransformerEncoderLayer(Dim, heads: 4, dropout: 0.1f, causal: true, random: r),
                new LayerNorm(Dim),
                new Linear(Dim, Vocab, random: r),                      // next-token scores per position
            };
            Console.WriteLine(gpt.Summary());

            using var scope = new TensorScope();
            Console.WriteLine(string.Join(",", gpt.Forward(Tensor.Zeros([8, 32])).Shape.ToArray()));
            """,
            out="""
            Sequential(6 layers)
              Embedding(65 -> 64)  [4,160 params]
              PositionalEncoding(128 x 64)
              TransformerEncoderLayer(dim 64)
                LayerNorm(64)  [128 params]
                MultiHeadAttention(dim 64, 4 heads, causal)
                  Linear(64 -> 192)  [12,480 params]
                  Linear(64 -> 64)  [4,160 params]
                  Dropout(p=0.1)
                LayerNorm(64)  [128 params]
                Linear(64 -> 256)  [16,640 params]
                Linear(256 -> 64)  [16,448 params]
                Dropout(p=0.1)
              TransformerEncoderLayer(dim 64)
                LayerNorm(64)  [128 params]
                MultiHeadAttention(dim 64, 4 heads, causal)
                  Linear(64 -> 192)  [12,480 params]
                  Linear(64 -> 64)  [4,160 params]
                  Dropout(p=0.1)
                LayerNorm(64)  [128 params]
                Linear(64 -> 256)  [16,640 params]
                Linear(256 -> 64)  [16,448 params]
                Dropout(p=0.1)
              LayerNorm(64)  [128 params]
              Linear(64 -> 65)  [4,225 params]
            Total trainable parameters: 108,481
            8,32,65
            """),
        cpugpu("a transformer sentence classifier",
               """
               Device.Default = Device.Cpu;
               var r = new Random(1);
               using var model = new Sequential
               {
                   new Embedding(vocabularySize, 32, random: r),
                   new PositionalEncoding(maxLength, 32),
                   new TransformerEncoderLayer(32, heads: 4, ffDim: 64, dropout: 0f, random: r),
                   new TransformerEncoderLayer(32, heads: 4, ffDim: 64, dropout: 0f, random: r),
                   new LayerNorm(32),
                   new Lambda(x => x.Mean(1), "MeanOverWords"),       // not causal: pool all positions
                   new Linear(32, classes, random: r),
               };
               """,
               """
               Device.Default = Device.Cuda();
               // same model; transformers are the biggest GPU win in this book:
               // the GPT of Chapter 34 trains about 8x faster on a laptop RTX 5050
               """.replace("Chapter 34", ch("gpt")),
               "This is the transformer variant of the sentiment models compared in " + ch("sentiment") + "."),
        reftable(["Model", "Blocks", "Causal", "Head"], [
            ["Sequence classifier", "<code>Embedding</code>, <code>PositionalEncoding</code>, 1–4 blocks, <code>LayerNorm</code>", "no", "mean over positions, <code>Linear(D, K)</code>"],
            ["Language model (GPT)", "same, 2–12 blocks", "yes", "<code>Linear(D, vocabulary)</code> at every position"],
            ["Per-token tagging", "same", "no", "<code>Linear(D, K)</code> at every position"],
            ["Time-series forecaster", "<code>Linear(F, D)</code> instead of <code>Embedding</code>, then as above", "yes (to predict ahead)", "<code>Linear(D, 1)</code> on the last position"],
        ], caption="Table 12.3 — Transformer model families"),
        h2("12.5 Inference and generation"),
        para("When autograd is off (<code>NoGrad</code> or <code>Predict</code>), attention computes scale, mask and "
             "softmax in one fused kernel, LayerNorm in one kernel, and the feed-forward bias and GELU in one kernel. "
             "Nothing changes in your code. For generating text token by token, the layers also implement a cached "
             "path that keeps each position's keys and values (a <b>KV cache</b>), so each new token costs one "
             "position's work instead of the whole sequence's; " + ch("generation") + " explains it and "
             + ch("gpt") + " uses it."),
        trap("dim not divisible by heads",
             "<p><code>new MultiHeadAttention(10, 4)</code> throws <i>dim (10) must be divisible by heads (4).</i> "
             "Choose D as a multiple of the head count (e.g. 64 with 4 or 8 heads).</p>"),
        trap("sequences longer than PositionalEncoding's maxLength",
             "<p>Training with 64-token windows and then feeding 200 tokens throws. Set <code>maxLength</code> to the "
             "longest context you will ever use, and cut inputs to it (a GPT keeps only the last maxLength tokens).</p>"),
        trap("dropout left on in generation",
             "<p>Sampling in training mode applies dropout to every step and degrades the text. Generation code must "
             "call <code>Eval()</code> (the helpers of " + ch("generation") + " do).</p>"),
        practice([
            (1, "How many parameters does one <code>TransformerEncoderLayer(128, heads: 8)</code> have?",
             "Two LayerNorms 2·256 = 512; attention 128·384 + 384 + 128·128 + 128 = 66,048; feed-forward 128·512 + 512 + "
             "512·128 + 128 = 131,712. Total 198,272 (heads do not change the count)."),
            (1, "Why does a sentence classifier not need <code>causal: true</code>?",
             "It sees the whole sentence at once and predicts one label for it, so every word may look at every other. "
             "Causality matters only when predicting the future from the past."),
            (2, "Estimate the memory of the attention weights for a batch of 32 sequences of 512 tokens with 8 heads.",
             "32·8·512·512 floats = 67,108,864 × 4 bytes ≈ 256 MiB per layer, stored for the backward pass. Halving the "
             "context divides it by four."),
            (2, "Turn the small GPT of Section 12.4 into a character-level model for your own text: which sizes change?",
             "Only <code>Vocab</code> (the number of distinct characters) and optionally <code>Context</code>, "
             "<code>Dim</code> and the number of blocks. " + ch("gpt") + " builds the vocabulary, the training windows and "
             "the sampling loop."),
            (3, "Build a transformer forecaster: inputs <code>[N, T, F]</code> of F sensor values, target the next value of sensor 0.",
             "<code>Linear(F, D)</code>, <code>PositionalEncoding(T, D)</code>, two causal <code>TransformerEncoderLayer(D, 4)</code>, "
             "<code>LayerNorm(D)</code>, then <code>Lambda(x =&gt; x.Narrow(1, T - 1, 1).Flatten())</code> to keep the last "
             "position and <code>Linear(D, 1)</code>. Train with MSE on scaled data (" + ch("timeseries") + ")."),
        ], PART),
        footer("Attention", "Query, key, value", "Scaled dot-product attention", "Multi-head attention",
               "Causal mask", "Transformer", "Positional encoding", "Feed-forward network", "Residual connection",
               "Pre-norm", "Context length", "KV cache"),
    )

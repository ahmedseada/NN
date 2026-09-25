"""Chapter 9 — Embeddings."""
from gen import *

PART = "II"


def lookup_svg():
    w, h = 470, 130
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">']
    ids = [3, 0, 3]
    p.append(svg_text(45, 20, "ids [3]", 8.6, "#56606a"))
    for i, v in enumerate(ids):
        p.append(f'<rect x="{25}" y="{30 + i * 28}" width="40" height="24" rx="3" fill="#fff4e2" stroke="#a15c00"/>')
        p.append(svg_text(45, 46 + i * 28, str(v), 10, "#a15c00"))
    p.append(svg_text(200, 20, "table Weight [5, 4]", 8.6, "#56606a"))
    for r in range(5):
        on = r in (0, 3)
        p.append(f'<rect x="150" y="{26 + r * 20}" width="100" height="18" rx="2" fill="{"#0f6b5c" if on else "#e6f2ef"}" stroke="#0f6b5c" stroke-width="0.8"/>')
        p.append(svg_text(140, 39 + r * 20, str(r), 8, "#56606a", "end"))
    p.append(svg_text(390, 20, "output [3, 4]", 8.6, "#56606a"))
    for i, v in enumerate(ids):
        p.append(f'<rect x="340" y="{30 + i * 28}" width="100" height="24" rx="3" fill="#0f6b5c" stroke="#0f6b5c"/>')
        p.append(svg_text(390, 46 + i * 28, f"row {v}", 8.6, "#ffffff"))
    for i, v in enumerate(ids):
        p.append(f'<line x1="65" y1="{42 + i * 28}" x2="150" y2="{35 + v * 20}" stroke="#a15c00" stroke-width="0.9"/>')
        p.append(f'<line x1="250" y1="{35 + v * 20}" x2="340" y2="{42 + i * 28}" stroke="#0f6b5c" stroke-width="0.9"/>')
    p.append("</svg>")
    return "".join(p)


def build():
    return page(
        chapter_open(
            "embedding",
            "Words, characters, product ids, user ids and categories are not numbers a network can multiply. An "
            "embedding gives each such id a learned vector: a row in a table that training adjusts so that ids used "
            "in similar ways end up with similar vectors. This chapter covers the <code>Embedding</code> layer, how to "
            "turn text and categories into ids, and how to pool a sequence of vectors into one.",
            "<code>Embedding(vocabulary, dim)</code> maps integer ids (stored as floats) of any shape <code>[...]</code> to vectors <code>[..., dim]</code>.",
            "Ids must lie in [0, vocabulary): the CPU throws on a bad id, the GPU clamps it, so validate ids when you build the data.",
            "Pool a sequence of vectors with <code>Lambda(x =&gt; x.Mean(1))</code>, an <code>LSTM</code>/<code>GRU</code>, or attention.",
            "Typical sizes: 8–32 for categories, 32–256 for words and characters.",
            "An embedding is equivalent to a one-hot input times a <code>Linear</code> without bias, but far cheaper.",
        ),
        h2("9.1 The lookup"),
        diagram("Figure 9.1 — An embedding is a table lookup", lookup_svg(),
                "Each id selects one row of the table. Repeated ids select the same row; in the backward pass their gradients add up."),
        mex("shapes, a single lookup and a bad id", None,
            """
            using var emb = new Embedding(vocabulary: 5, dim: 3, random: new Random(4));
            using var scope = new TensorScope();

            var ids = Tensor.From(new float[,] { { 0, 2, 2 }, { 4, 1, 0 } });   // [2 sentences, 3 tokens]
            Console.WriteLine(string.Join(",", emb.Forward(ids).Shape.ToArray()));
            Console.WriteLine(emb.Forward(Tensor.From([2f])));
            Console.WriteLine(emb);

            try { emb.Forward(Tensor.From([7f])); }
            catch (Exception e) { Console.WriteLine(e.GetType().Name + ": " + e.Message); }
            """,
            out="""
            2,3,3
            Tensor(shape=[1, 3], device=cpu, requiresGrad)
            [[-0.9328, -1.7177, -1.2379]]
            Embedding(5 -> 3)
            IndexOutOfRangeException: Embedding index 7 is not an integer in [0, 5).
            """),
        reftable(["Member", "Meaning"], [
            ["<code>new Embedding(vocabulary, dim, device, random)</code>", "A table of <code>vocabulary</code> vectors of size <code>dim</code>, initialized from N(0, 1)"],
            ["<code>Weight</code>", "The <code>[vocabulary, dim]</code> table (a parameter)"],
            ["<code>Vocabulary</code>, <code>Dim</code>", "The two sizes"],
            ["Input", "Any shape of integer-valued floats (exact up to 16,777,216)"],
            ["Output", "Input shape + <code>[dim]</code>"],
        ], caption="Table 9.1 — The Embedding layer"),
        trap("ids out of range on the GPU",
             "<p>On the GPU an id of −1 or ≥ vocabulary is silently clamped to the nearest valid row rather than raising "
             "an error, because checking would need a round trip to the CPU. Validate ids once when you build the "
             "dataset, and reserve id 0 for padding and 1 for unknown tokens (Section 9.2).</p>"),
        h2("9.2 From text and categories to ids"),
        snippet("""
            // A vocabulary with reserved ids: 0 = padding, 1 = unknown word.
            sealed class Vocabulary
            {
                private readonly Dictionary<string, int> _ids = new(StringComparer.OrdinalIgnoreCase);
                public const int Pad = 0, Unknown = 1;

                public Vocabulary(IEnumerable<string> texts, int minCount = 1)
                {
                    var counts = texts.SelectMany(Tokenize).CountBy(w => w, StringComparer.OrdinalIgnoreCase);
                    foreach (var (word, count) in counts.OrderByDescending(c => c.Value))
                        if (count >= minCount) _ids[word] = _ids.Count + 2;
                }

                public int Size => _ids.Count + 2;

                public static IEnumerable<string> Tokenize(string text) =>
                    text.Split([' ', ',', '.', '!', '?', ';', ':'], StringSplitOptions.RemoveEmptyEntries);

                /// Ids for one text, cut or padded to exactly `length` positions.
                public float[] Encode(string text, int length)
                {
                    var ids = new float[length];                           // zeros = padding
                    int i = 0;
                    foreach (var word in Tokenize(text).Take(length))
                        ids[i++] = _ids.TryGetValue(word, out int id) ? id : Unknown;
                    return ids;
                }
            }
            """, caption="A small word vocabulary you can reuse"),
        para("For categorical columns in a table (country, product type, weekday) the vocabulary is simply the list "
             "of distinct values, and each row contributes one id per categorical column. " + ch("recommender") +
             " builds a complete tabular model that mixes embeddings with numeric features, and " + ch("gpt") +
             " uses one id per character."),
        reftable(["Kind of id", "Vocabulary size", "Suggested dim"], [
            ["Weekday, month, small category", "7–50", "4–8"],
            ["Product type, country", "50–1,000", "8–32"],
            ["User or item ids", "10³–10⁶", "32–128"],
            ["Characters", "50–300", "32–128"],
            ["Words", "10⁴–10⁵", "64–300"],
        ], caption="Table 9.2 — Rule-of-thumb embedding sizes"),
        h2("9.3 Pooling: from many vectors to one"),
        para("A sentence of T tokens becomes <code>[N, T, dim]</code> after the embedding. A classifier needs one "
             "vector per sample, so the T vectors must be combined."),
        reftable(["Pooling", "Code", "Keeps word order?"], [
            ["Mean", "<code>new Lambda(x =&gt; x.Mean(1), \"MeanPool\")</code>", "No (bag of words)"],
            ["Sum", "<code>new Lambda(x =&gt; x.Sum(1), \"SumPool\")</code>", "No"],
            ["Last hidden state of a recurrent layer", "<code>new LSTM(dim, h)</code> / <code>new GRU(dim, h)</code>", "Yes (" + ch("recurrent") + ")"],
            ["Transformer + mean", "<code>PositionalEncoding</code>, <code>TransformerEncoderLayer</code>, …, mean", "Yes (" + ch("attention") + ")"],
        ], caption="Table 9.3 — Pooling options"),
        cpugpu("a bag-of-words sentiment model",
               """
               Device.Default = Device.Cpu;
               var r = new Random(1);
               using var model = new Sequential
               {
                   new Embedding(vocabulary.Size, 32, random: r),     // [N, T] -> [N, T, 32]
                   new Lambda(x => x.Mean(1), "MeanPool"),           // -> [N, 32]
                   new Linear(32, 2, random: r),                      // -> [N, 2] class scores
               };
               """,
               """
               Device.Default = Device.Cuda();
               // same model; the token-id tensors must be created on the GPU as well:
               using var ids = Tensor.From(encodedBatch, device: Device.Default);
               """),
        honestbox("Padding and the mean",
                  "<p>Padding positions (id 0) are embedded and averaged like real tokens. With a trainable pad row "
                  "this works well in practice, since the model learns a harmless pad vector, but short sentences are "
                  "then diluted by padding. Keep sequence lengths close to the typical sentence length, or use a "
                  "recurrent or attention model, which can learn to ignore padding.</p>"),
        h2("9.4 What embeddings learn"),
        mex("similar words end up close",
            "Six words, each labelled positive or negative, trained through a 4-dimensional embedding and one "
            "<code>Linear</code>. Afterwards the cosine similarity (glossary <b>Cosine similarity</b>) of the learned "
            "vectors shows the grouping the task required.",
            """
            string[] words = ["good", "great", "fine", "bad", "awful", "poor"];
            float[] label = [1, 1, 1, 0, 0, 0];
            var r = new Random(5);
            using var model = new Sequential
            {
                new Embedding(words.Length, 4, random: r), new Flatten(), new Linear(4, 1, random: r),
            };
            using var x = Tensor.From(new float[,] { { 0 }, { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });
            using var y = Tensor.From(label, [6, 1]);
            using var opt = new Adam(model.Parameters(), 0.05f);
            for (int i = 0; i < 200; i++)
            {
                using var s = new TensorScope();
                var loss = Losses.BinaryCrossEntropyWithLogits(model.Forward(x), y);
                opt.ZeroGrad(); loss.Backward(); opt.Step();
            }

            var table = ((Embedding)model[0]).Weight.ToArray2D();
            float Cos(int a, int b)
            {
                float dot = 0, na = 0, nb = 0;
                for (int k = 0; k < 4; k++)
                {
                    dot += table[a, k] * table[b, k];
                    na += table[a, k] * table[a, k];
                    nb += table[b, k] * table[b, k];
                }
                return dot / MathF.Sqrt(na * nb);
            }
            Console.WriteLine($"good~great {Cos(0, 1):F2}  good~bad {Cos(0, 3):F2}  awful~poor {Cos(4, 5):F2}");
            """,
            out="""
            good~great 0.87  good~bad -0.55  awful~poor 0.93
            """),
        para("The same property powers recommenders: users and items that interact similarly get similar vectors, "
             "and the dot product of a user vector and an item vector predicts interest (" + ch("recommender") + ")."),
        practice([
            (1, "What is the output shape of <code>new Embedding(1000, 64)</code> applied to ids of shape <code>[32, 50]</code>?",
             "<code>[32, 50, 64]</code>: one 64-dimensional vector per id."),
            (1, "How many parameters does <code>Embedding(50_000, 128)</code> have, and how many bytes do they take?",
             "6,400,000 parameters, 25.6 MB as float32 (plus the same again for the gradient and twice more for Adam's state during training)."),
            (2, "Encode the sentence \"The movie was GREAT!\" with a vocabulary built from [\"the movie was great\", \"awful movie\"], using length 6.",
             "Ids are given by frequency, ties in order of first appearance, after the two reserved ids: movie = 2, "
             "the = 3, was = 4, great = 5, awful = 6 (vocabulary size 7). The result is [3, 2, 4, 5, 0, 0]: \"GREAT\" "
             "matches because the dictionary ignores case, \"!\" is removed by the tokenizer, and the last two "
             "positions are padding."),
            (2, "Show that <code>Embedding(V, d)</code> gives the same result as <code>Linear(V, d, bias: false)</code> applied to one-hot rows.",
             "A one-hot row times the weight matrix selects exactly one row of it. Compute "
             "<code>Tensor.OneHot(ids, V).MatMul(emb.Weight)</code> and compare with <code>emb.Forward(ids)</code>: they "
             "are equal, but the embedding avoids building and multiplying the V-wide one-hot matrix."),
            (3, "Build a tabular model with one numeric input block of 10 features and one categorical column with 200 values: embed the category (dim 8), concatenate with the numeric features and feed an MLP.",
             "Write a composite module holding an <code>Embedding(200, 8)</code> and an MLP whose first layer is "
             "<code>Linear(18, 64)</code>. Its input can be <code>[N, 11]</code> with the category id in column 10: "
             "use <code>Narrow(1, 0, 10)</code> for the numbers and <code>Narrow(1, 10, 1)</code> for the id, embed and "
             "<code>Flatten</code> the id, then <code>Tensor.Concat([numbers, embedded], 1)</code>. " + ch("recommender") +
             " develops this fully."),
        ], PART),
        footer("Embedding", "Vocabulary", "Token", "Padding", "Unknown token", "Pooling", "Bag of words",
               "Cosine similarity", "Categorical feature"),
    )

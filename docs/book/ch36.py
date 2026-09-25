"""Chapter 36 — Search Re-Ranking with a Cross-Encoder."""
from gen import *

PART = "VI"


def stages_svg():
    w, h = 470, 120
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">']
    items = [(10, "question", "#56606a", "#f2f4f5"), (120, "BM25 over 1,600", "#56606a", "#f2f4f5"),
             (240, "cross-encoder", "#0f6b5c", "#e6f2ef"), (360, "ranked answers", "#0f6b5c", "#e6f2ef")]
    for x, t, stroke, fill in items:
        p.append(f'<rect x="{x}" y="34" width="100" height="36" rx="4" fill="{fill}" stroke="{stroke}" stroke-width="0.8"/>')
        p.append(svg_text(x + 50, 56, t, 8.2, stroke))
    for x, label in ((110, ""), (230, "20 candidates"), (350, "20 scores")):
        p.append(f'<path d="M{x} 52 L{x + 6} 52" stroke="#1a1f23" stroke-width="1"/><path d="M{x + 6} 49 L{x + 9} 52 L{x + 6} 55 Z" fill="#1a1f23"/>')
        if label:
            p.append(svg_text(x + 4, 26, label, 7.2, "#56606a"))
    p.append(svg_text(170, 88, "fast: word matching, whole collection", 7.4, "#56606a"))
    p.append(svg_text(290, 100, "slow but accurate: reads question and passage together", 7.4, "#0f6b5c"))
    p.append("</svg>")
    return "".join(p)


SCORER = """
    // "<cls> question <sep> passage <pad>..." (28 ids) → one relevance score, read at the <cls> position.
    Sequential Scorer(int vocabulary, Random r) => new()
    {
        new Embedding(vocabulary, Dim, device, r),                     // Dim = 64
        new PositionalEncoding(Length, Dim, device),                   // Length = 28
        new TransformerEncoderLayer(Dim, heads: 4, ffDim: 2 * Dim, dropout: 0.1f, device: device, random: r),
        new TransformerEncoderLayer(Dim, heads: 4, ffDim: 2 * Dim, dropout: 0.1f, device: device, random: r),
        new LayerNorm(Dim, device: device),
        new Lambda(x => x.Narrow(1, 0, 1).Reshape(-1, Dim), "ClsToken"),   // [N, 28, 64] → [N, 64]
        new Linear(Dim, 1, device: device, random: r),
    };
"""

GROUPS = """
    // One training row = one question with 8 passages (answer at a random position + 7 BM25 hard negatives),
    // flattened to 8 × 28 ids; the label is the answer's position (a class out of 8).
    var scorer = Scorer(tokenizer.VocabularySize, new Random(2));
    using var groupModel = new Sequential
    {
        new Lambda(x => x.Reshape(-1, Length), "PairsOfGroup"),       // [B, 8·28] → [B·8, 28]
        scorer,                                                        // → [B·8, 1]
        new Lambda(x => x.Reshape(-1, Group), "ScoresOfGroup"),       // → [B, 8]: logits over the group
    };
    using var optimizer = new AdamW(groupModel.Parameters(), learningRate: 0.002f, weightDecay: 0.01f);
    var trainer = new Trainer(groupModel, optimizer, (logits, targets) => Losses.CrossEntropy(logits, targets))
    {
        Metrics = { Metric.Accuracy },                                 // = answer ranked first within its group
        Scheduler = new CosineAnnealing(optimizer, epochs, minLearningRate: 1e-4f, warmupEpochs: 1),
        MaxGradientNorm = 1f,
    };
    trainer.Fit(new DataLoader(trainSet, 32, shuffle: true, device: device, seed: 5), epochs,
        validation: new DataLoader(validation, 200, device: device));
    scorer.Save(modelPath);                                            // only the pair scorer is needed later
"""

PLACEHOLDERS = """
    // Vocabulary: words used for at least two training towns; a town's own name is not in it.
    // Unknown words become <w0>, <w1>, ... numbered by first appearance in each (question, passage) pair.
    var placeholders = new Dictionary<string, int>();
    int Id(string word) => words[word] != words["<unk>"] ? words[word]
        : placeholders.Count < Placeholders || placeholders.ContainsKey(word)
            ? words[$"<w{(placeholders.TryGetValue(word, out int k) ? k : placeholders[word] = placeholders.Count)}>"]
            : words["<unk>"];
    var ids = new List<int> { words["<cls>"] };
    ids.AddRange(WordTokenizer.Split(question).Select(Id));
    ids.Add(words["<sep>"]);
    ids.AddRange(WordTokenizer.Split(passage).Select(Id));
"""

RESULTS = """
    1600 passages about 160 towns; 2880 training questions (120 towns), 960 test questions (40 unseen towns); vocabulary 222 (+ placeholders for town names)
    Example: "how many people live in tormor" -> "roughly 37 thousand inhabitants call tormor home ."

    BM25 alone:            Hit@1 27.1 %   MRR@10 0.429   Recall@20 94.9 %

    Cross-encoder: 2 layers, dim 64, 81,345 parameters; groups of 8 (1 answer + 7 hard negatives), 12 epochs
    Epoch 12/12  loss 0.000171  accuracy 1.0000  val_loss 0.000006  val_accuracy 1.0000  14020.0 ms  195 samples/s  *
    Trained in 180 s

    BM25 + cross-encoder:  Hit@1 86.6 %   MRR@10 0.895   Recall@20 94.9 %
    The re-ranker can only order what the first stage found: of the 94.9 % of questions whose answer is among the 20 candidates, it puts 91.2 % first.
    Re-ranking cost: 6.90 ms per question (20 pairs in one batch, cpu)

    Hit@1 per aspect (unseen towns)      BM25   re-ranked
      population                          0 %      67 %
      founded                            48 %     100 %
      food                               26 %      72 %
      river                              34 %      79 %
      climate                            17 %      95 %
      sport                              37 %      80 %
      mayor                              40 %     100 %
      transport                          14 %     100 %

    "how many people live in tormor"
      BM25 top:      many people visit tormor to see the old town .
      re-ranked top: roughly 37 thousand inhabitants call tormor home .
    "which river runs through tormor"
      BM25 top:      the ash river flows through pelby .
      re-ranked top: the stone river flows through lorwick .
    "is there public transport in tormor"
      BM25 top:      summers in tormor are mild and winters are wet .
      re-ranked top: tormor has a metro service across the city .
"""


def build():
    return page(
        chapter_open(
            "reranker",
            "Search usually runs in two stages. A fast first stage matches words over the whole collection and returns a "
            "few dozen candidates; a slower, more accurate model then reads the question together with each candidate "
            "and re-orders them. This project, <code>NeuralSharp.Samples.ReRanker</code>, builds both: BM25 as the first "
            "stage and a small transformer <b>cross-encoder</b> as the re-ranker, trained listwise on hard negatives and "
            "tested on towns it never saw during training.",
            "First stage: BM25 word matching; the answer is among its top 20 for 94.9 % of the questions, but first for only 27.1 %.",
            "Re-ranker: 2 transformer layers read \"&lt;cls&gt; question &lt;sep&gt; passage\" and output one score (81,345 parameters).",
            "Training: one answer + 7 hard negatives per question, softmax cross-entropy over the 8 scores, with the ordinary <code>Trainer</code>.",
            "Rare words (town names) become placeholders <code>&lt;w0&gt;</code>, <code>&lt;w1&gt;</code>…: without them 57.5 % first on unseen towns, with them 86.6 %.",
            "Cost: 6.9 ms per question to score 20 candidates on the CPU; training 180 s.",
        ),
        h2("36.1 The task and the data"),
        diagram("Figure 36.1 — Two-stage search", stages_svg(),
                "The first stage must be fast because it looks at every passage; the second may be slow because it looks at 20."),
        para("The collection (<code>Collection.cs</code>) describes 160 invented towns. Each town has one passage for each of "
             "eight aspects (population, founding, food, river, climate, sport, mayor, transport) and two general passages. "
             "Each aspect has three question wordings. The data is built to be hard for word matching: questions and "
             "answers use different words, while the general passages share the questions' words without answering them."),
        reftable(["Question", "Answer passage", "Tempting wrong passage"], [
            ["how many people live in tormor", "roughly 37 thousand <b>inhabitants</b> call tormor home .", "many <b>people</b> visit tormor to see the old town ."],
            ["which <b>river</b> runs through tormor", "tormor sits on the banks of the reed .", "the ash <b>river</b> flows through pelby ."],
            ["is there public transport in tormor", "tormor has a metro service across the city .", "summers <b>in</b> tormor are mild ... (shares \"in\", \"tormor\")"],
        ], caption="Table 36.1 — Why word matching struggles"),
        para("The towns are split: questions about the first 120 are used for training, questions about the other 40 "
             "(960 questions) only for testing. A re-ranker that merely memorized towns would fail the test."),
        trap("testing on what was trained",
             "<p>Splitting the <i>questions</i> randomly would put questions about the same town in both sets; the model "
             "could then score well by remembering which passages belong to which town. Split by the unit you will meet "
             "new in production (here the town; in real search, new documents and new users).</p>"),
        h2("36.2 The first stage: BM25"),
        para("BM25 (<code>Bm25</code> in the sample) scores a passage by the question words it contains. Rare words count "
             "more than common ones (a logarithm of how few passages contain the word, the inverse document frequency), "
             "repeated words count with diminishing returns, and long passages are penalized slightly (glossary <b>BM25</b>; "
             "logarithms are in the Pre-Calc volume). It needs no training and scans 1,600 passages in well under a "
             "millisecond."),
        snippet("""
            // score(passage) = Σ over question words w in the passage of
            //     idf(w) · tf · (k1 + 1) / (tf + k1 · (1 − b + b · length / averageLength))
            //   idf(w) = log(1 + (N − n + 0.5) / (n + 0.5)),  n = passages containing w,  k1 = 1.2,  b = 0.75
            var bm25 = new Bm25(collection.Passages.Select(p => p.Text));
            int[] candidates = bm25.Search("how many people live in tormor", k: 20);   // passage ids, best first
            """, caption="The first stage"),
        para("BM25 finds the right passage somewhere in its top 20 for 94.9 % of the test questions (Recall@20) but "
             "ranks it first for only 27.1 % (Hit@1), and never for the population questions: \"people\" and \"live\" "
             "point to the general passages, not to \"inhabitants\" and \"residents\"."),
        h2("36.3 The cross-encoder and listwise training"),
        para("A cross-encoder reads the question and the passage as one sequence, so attention can relate every question "
             "word to every passage word (\"live\" ↔ \"inhabitants\", the town in the question ↔ the town in the passage). "
             "It is the same kind of transformer as the sentiment model of " + ch("sentiment") + ", with a single score "
             "as output."),
        snippet(SCORER, caption="The pair scorer"),
        para("Training shows the model one question with eight passages: the answer and seven <b>hard negatives</b>, the "
             "passages BM25 ranks highest that are not the answer. The eight scores are treated as the logits of an "
             "8-class problem whose correct class is the answer's position, so the ordinary cross-entropy loss teaches the "
             "model to score the answer above its closest competitors (a <b>listwise loss</b>). Two <code>Lambda</code> "
             "layers reshape a batch of groups into pairs and the scores back into groups, so <code>Trainer</code>, "
             "<code>DataLoader</code> and <code>Metric.Accuracy</code> work unchanged."),
        snippet(GROUPS, caption="Training on groups with the standard Trainer"),
        trap("easy negatives",
             "<p>With random passages as negatives, the model only needs to check that the town matches, and it reaches "
             "high training accuracy without learning the aspect words. Negatives must be the mistakes the first stage "
             "actually makes: take them from its top results.</p>"),
        h2("36.4 Words the model has never seen"),
        para("The first version of this project put every word, town names included, in the vocabulary. It reached 99.3 % "
             "within-group accuracy on training towns, but only 57.5 % Hit@1 on unseen towns: an unseen name is an "
             "untrained embedding, so the model could not tell whether the town in a passage was the town in the question, "
             "and river passages of other towns won. The fix is a standard one for rare words: words that occur for only "
             "one town (their names) are left out of the vocabulary and encoded, per pair, as placeholders numbered by "
             "first appearance."),
        snippet(PLACEHOLDERS, caption="Placeholder encoding of rare words"),
        para("\"how many people live in tormor\" paired with \"roughly 37 thousand inhabitants call tormor home .\" becomes "
             "<code>&lt;cls&gt; how many people live in &lt;w0&gt; &lt;sep&gt; roughly 37 thousand inhabitants call &lt;w0&gt; home .</code>, "
             "while a passage about lorwick gets <code>&lt;w1&gt;</code>. The model learns \"same placeholder = same "
             "entity\", which holds for any name. Hit@1 on unseen towns rose from 57.5 % to 86.6 %, with a smaller "
             "vocabulary (222 words instead of 550)."),
        h2("36.5 Results"),
        output(RESULTS, caption="dotnet run -c Release --project samples/NeuralSharp.Samples.ReRanker -- --cpu"),
        para("Re-ranking lifts Hit@1 from 27.1 % to 86.6 % and MRR@10 from 0.43 to 0.90 (glossary <b>MRR</b>). The "
             "re-ranker cannot recover an answer the first stage missed, so 94.9 % is its ceiling; of the reachable "
             "answers it ranks 91.2 % first. The remaining errors are mostly questions whose answer is missing from the 20 "
             "candidates (the river question above: its answer, \"tormor sits on the banks of the reed .\", has no "
             "\"river\", and BM25 leaves it out of its top 20), so the next improvement is a larger candidate list or a better first stage, "
             "not a bigger re-ranker."),
        reftable(["Candidates re-ranked", "Effect"], [
            ["More (50–100)", "Higher ceiling (recall); cost grows linearly (6.9 ms per 20 here)"],
            ["Fewer (5–10)", "Cheaper; the re-ranker can only fix small ordering mistakes"],
            ["Batch all candidates in one call", "One <code>Predict</code> of [20, 28] ids instead of 20 calls"],
        ], caption="Table 36.2 — Tuning the second stage"),
        cpugpu("training and inference commands",
               """
               dotnet run -c Release --project samples/NeuralSharp.Samples.ReRanker -- --cpu
               dotnet run -c Release --project samples/NeuralSharp.Samples.ReRanker -- --cpu --predict \\
                   --input "how many people live in armor"
               """,
               """
               dotnet run -c Release --project samples/NeuralSharp.Samples.ReRanker -- --cuda
               dotnet run -c Release --project samples/NeuralSharp.Samples.ReRanker -- --cuda --predict \\
                   --input "is it cold?|winters are icy and long .|the shop opens at nine ."
               """,
               note="With <code>|</code> the input is a question followed by your own passages to re-rank. On the GPU, "
                    "all 20 pairs are one batch of small kernels; the gain over the CPU grows with the number of "
                    "candidates and the size of the model."),
        practice([
            (1, "Why is BM25 run over the whole collection but the cross-encoder only over 20 passages?",
             "BM25 costs microseconds per passage and needs no model; the cross-encoder runs a transformer per pair "
             "(6.9 ms per 20 here), far too slow for every passage of a large collection."),
            (1, "What does Recall@20 of 94.9 % limit?",
             "The best possible Hit@1 of the two-stage system: the re-ranker only orders the 20 candidates."),
            (2, "Change the sample to re-rank 50 candidates. What do you expect for Recall, Hit@1 and cost?",
             "Set <code>Candidates = 50</code>: Recall rises (the ceiling), Hit@1 rises if the re-ranker keeps its accuracy on "
             "the extra, mostly easy candidates, and the cost per question rises about 2.5×."),
            (2, "Why must the group model's output be [B, 8] and not [B·8, 1] for the loss?",
             "Cross-entropy normalizes over the last dimension; the softmax must run over the 8 passages of one question, "
             "so the scores of a group have to be one row."),
            (3, "Replace the cross-encoder by a bi-encoder: encode questions and passages separately into vectors and score by dot product. What do you gain and lose?",
             "Passage vectors can be computed once and searched quickly (it can even replace BM25), but question and "
             "passage words no longer attend to each other, so fine distinctions (which town, which aspect) are usually "
             "less accurate; bi-encoders are typical first stages, cross-encoders re-rankers."),
        ], PART),
        footer("Re-ranking", "Two-stage retrieval", "BM25", "Cross-encoder", "Hard negative", "Listwise loss",
               "MRR", "Recall@k", "Placeholder token"),
    )

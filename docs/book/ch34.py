"""Chapter 34 — Sequence Classification and Sentiment."""
from gen import *

PART = "VI"


def build():
    return page(
        chapter_open(
            "sentiment",
            "Is this review positive or negative? The answer depends on word order: \"not good\" is negative, \"not bad\" "
            "positive. This project, the repository's <code>NeuralSharp.Samples.Sequences</code>, trains four models on "
            "the same sentences, a bag of words that ignores order, an LSTM, a GRU and a transformer, and measures "
            "exactly where order matters. It is the template for any text or event-sequence classifier: support "
            "tickets, intents, log lines, click streams.",
            "Task type: <b>sequence classification</b>. Input: token ids <code>[N, T]</code>; output: class scores <code>[N, K]</code>.",
            "Every model starts with <code>Embedding</code>; they differ in how they combine the T word vectors.",
            "Result: bag of words 70.1% (57.3% on sentences with \"not\"); LSTM 100%, GRU 99.9%, transformer 99.5%.",
            "The transformer needed 40 epochs against 15 for the recurrent models, a typical pattern on small data.",
            "The <code>--predict</code> mode compares all saved models on your sentences.",
        ),
        h2("34.1 Vocabulary and data"),
        snippet("""
            string[] positive = ["good", "great", "excellent", "love", "wonderful", "fun"];
            string[] negative = ["bad", "awful", "terrible", "hate", "boring", "dull"];
            string[] neutral = ["the", "movie", "was", "a", "plot", "acting", "really", "very", "it", "this", "i", "at", "all", "and"];
            string[] vocabulary = ["<pad>", "<unk>", "not", .. positive, .. negative, .. neutral];
            var ids = vocabulary.Select((word, id) => (word, id)).ToDictionary(p => p.word, p => p.id);
            const int MaxLength = 12;                                      // sentences padded/cut to 12 tokens
            """, caption="A 29-word vocabulary with padding and unknown-word ids"),
        para("The sample generates 6,000 training and 1,500 test sentences of 5–12 words mixing sentiment words, "
             "neutral words and \"not\", which flips the word after it; the label follows from the words. Real text "
             "uses the <code>Vocabulary</code> helper of " + ch("embedding") + " to build ids from a corpus."),
        h2("34.2 Four models"),
        snippet("""
            const int Dim = 32;
            var models = new (string Name, int Epochs, Func<Random, Module> Create)[]
            {
                ("Bag of words (order-blind baseline)", 15, r => new Sequential
                {
                    new Embedding(vocabulary.Length, Dim, device, r),
                    new Lambda(x => x.Mean(1), "MeanOverWords"),
                    new Linear(Dim, 2, device: device, random: r),
                }),
                ("LSTM", 15, r => new Sequential
                {
                    new Embedding(vocabulary.Length, Dim, device, r),
                    new LSTM(Dim, 64, device: device, random: r),
                    new Linear(64, 2, device: device, random: r),
                }),
                ("GRU", 15, r => new Sequential
                {
                    new Embedding(vocabulary.Length, Dim, device, r),
                    new GRU(Dim, 64, device: device, random: r),
                    new Linear(64, 2, device: device, random: r),
                }),
                ("Transformer", 40, r => new Sequential
                {
                    new Embedding(vocabulary.Length, Dim, device, r),
                    new PositionalEncoding(MaxLength, Dim, device),
                    new TransformerEncoderLayer(Dim, heads: 4, ffDim: 64, dropout: 0f, device: device, random: r),
                    new TransformerEncoderLayer(Dim, heads: 4, ffDim: 64, dropout: 0f, device: device, random: r),
                    new LayerNorm(Dim, device: device),
                    new Lambda(x => x.Mean(1), "MeanOverWords"),
                    new Linear(Dim, 2, device: device, random: r),
                }),
            };
            """, caption="Same embedding, four ways to read the sentence"),
        snippet("""
            foreach (var (name, epochs, create) in models)
            {
                var model = create(new Random(2));
                var optimizer = new AdamW(model.Parameters(), learningRate: 0.003f, weightDecay: 1e-4f);
                var trainer = new Trainer(model, optimizer, (logits, targets) => Losses.CrossEntropy(logits, targets))
                {
                    Metrics = { Metric.Accuracy },
                    Scheduler = new CosineAnnealing(optimizer, epochs, warmupEpochs: 1),
                    MaxGradientNorm = 1f,                                  // recurrent layers: clip
                };
                trainer.Fit(new DataLoader(train, 64, shuffle: true, device: device, seed: 3), epochs);
                double accuracy = trainer.Evaluate(new DataLoader(test, 500, device: device)).Metrics["accuracy"];
                double negated = trainer.Evaluate(new DataLoader(testNegated, 500, device: device)).Metrics["accuracy"];
                model.Save(ModelFile(name));
            }
            """, caption="One training loop for all four"),
        output("""
            === Bag of words (order-blind baseline)
            Epoch 15/15  loss 0.566507  accuracy 0.7072  14.1 ms  424,680 samples/s  *
            === LSTM
            Epoch 15/15  loss 0.000262  accuracy 1.0000  400.9 ms  14,966 samples/s  *
            === GRU
            Epoch 15/15  loss 0.000185  accuracy 1.0000  308.2 ms  19,471 samples/s  *
            === Transformer
            Epoch 20/40  loss 0.005096  accuracy 0.9987  620.7 ms  9,667 samples/s  *
            Epoch 40/40  loss 0.000065  accuracy 1.0000  562.8 ms  10,661 samples/s  *

            Model                                   test accuracy   sentences with "not"
              Bag of words (order-blind baseline)        70.1 %               57.3 %
              LSTM                                      100.0 %              100.0 %
              GRU                                        99.9 %               99.9 %
              Transformer                                99.5 %               99.3 %
            """, caption="Selected training lines and the comparison (CPU)"),
        reftable(["Model", "Parameters", "Training time (CPU)", "Reads order by"], [
            ["Bag of words", "994", "0.6 s", "— (averages word vectors)"],
            ["LSTM", "25,890", "7.2 s", "carrying a state through the words"],
            ["GRU", "19,682", "4.7 s", "carrying a state (fewer gates)"],
            ["Transformer", "18,146", "24.1 s (40 epochs)", "positional encodings + attention"],
        ], caption="Table 34.1 — The four models compared"),
        para("The bag of words gets the easy sentences right but is little better than a coin toss (57%) when \"not\" "
             "decides the answer: averaging throws away which word \"not\" was next to. All order-aware models solve "
             "the task. On such a small dataset the recurrent models learn fastest; transformers overtake them on "
             "longer texts and larger datasets (" + ch("recurrent") + ", Table 11.3)."),
        h2("34.3 Using the models"),
        snippet("""
            float[] PositiveProbabilities(Module model, string[] texts)
            {
                var encoded = new float[texts.Length, MaxLength];                // zeros = <pad>
                for (int s = 0; s < texts.Length; s++)
                {
                    var words = texts[s].ToLowerInvariant().Split(' ', StringSplitOptions.RemoveEmptyEntries);
                    for (int t = 0; t < Math.Min(words.Length, MaxLength); t++)
                        encoded[s, t] = ids.GetValueOrDefault(words[t], ids["<unk>"]);
                }
                using var input = Tensor.From(encoded, device);
                using var logits = model.Predict(input);
                using var probabilities = logits.Softmax();
                var p = probabilities.ToArray();
                return [.. Enumerable.Range(0, texts.Length).Select(s => p[s * 2 + 1])];   // column 1 = positive
            }
            """, caption="Tokenize, pad, predict, softmax"),
        output("""
            "the movie was not good"
                Bag of words (order-blind baseline)  positive (67 %)
                LSTM                                 negative (100 %)
                GRU                                  negative (100 %)
                Transformer                          negative (100 %)
            "not bad at all"
                Bag of words (order-blind baseline)  negative (60 %)
                LSTM                                 positive (100 %)
                GRU                                  positive (100 %)
                Transformer                          positive (100 %)
            """, caption="--predict --input \"the movie was not good;not bad at all\""),
        cpugpu("commands",
               """
               dotnet run -c Release --project samples/NeuralSharp.Samples.Sequences -- --cpu
               dotnet run -c Release --project samples/NeuralSharp.Samples.Sequences -- --cpu --predict --input "i hate it;not boring"
               """,
               """
               dotnet run -c Release --project samples/NeuralSharp.Samples.Sequences -- --cuda --batch-size 256
               """),
        h2("34.4 Real text"),
        reftable(["Need", "How"], [
            ["Build the vocabulary", "Count words in the training texts; keep those seen at least 2–5 times; reserve pad and unknown ids"],
            ["Longer documents", "Raise <code>MaxLength</code> (and <code>PositionalEncoding</code>'s length); consider truncating from the start"],
            ["More classes (topics, intents)", "<code>Linear(h, K)</code> and K-class labels"],
            ["Several labels per text", "Sigmoid outputs with <code>BinaryCrossEntropyWithLogits</code> (" + ch("binary") + ")"],
            ["Tag each word (names, dates)", "<code>returnSequences: true</code> and a per-step <code>Linear</code> (" + ch("recurrent") + ")"],
            ["Misspellings and rare words", "Character-level input (as in " + ch("gpt") + ") or sub-word pieces"],
        ], caption="Table 34.2 — From the sample to real text"),
        trap("unknown words in production",
             "<p>Words not in the training vocabulary map to <code>&lt;unk&gt;</code>. If many important words are unknown, "
             "predictions degrade silently. Log the share of unknown tokens per request and retrain with a larger "
             "vocabulary when it grows.</p>"),
        practice([
            (1, "Why can the bag-of-words model not tell \"not good\" from \"good not\"?",
             "The mean of the word vectors is the same for both orders."),
            (1, "What is the input shape for a batch of 64 sentences?",
             "<code>[64, 12]</code> token ids; the embedding turns it into <code>[64, 12, 32]</code>."),
            (2, "Add a third class \"neutral\" for sentences without sentiment words.",
             "Generate or label neutral sentences, make the targets 3-class, and change every model's last layer to "
             "<code>Linear(…, 3)</code>; the probability code then reads three columns."),
            (2, "Make the transformer train faster on this data.",
             "Try a higher learning rate with warm-up, one block instead of two, or a smaller <code>ffDim</code>; measure epochs to "
             "99% accuracy rather than time per epoch."),
            (3, "Classify real support tickets into 8 categories from a CSV with <code>text</code> and <code>category</code> columns.",
             "Read the file with your own CSV code (text columns), build a <code>Vocabulary</code> from the training texts, encode to "
             "<code>[N, T]</code> ids, build the dataset with <code>FromClassLabels(ids, labels, 8)</code> and <code>WithFeatureShape(T)</code>, "
             "use the GRU model with <code>Linear(64, 8)</code>, and report per-class accuracy."),
        ], PART),
        footer("Sequence classification", "Sentiment analysis", "Token", "Vocabulary", "Bag of words", "LSTM",
               "GRU", "Transformer", "Padding"),
    )

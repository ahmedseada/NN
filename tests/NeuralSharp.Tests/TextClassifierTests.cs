using NeuralSharp;
using NeuralSharp.Text;

internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] TextClassifierGroup =
    [
        ("text classifier: trains on Arabic and English intents, scores held-out texts, saves and loads to the same predictions; split keeps copies together; CSV with quoted line breaks", TextClassification),
    ];

    private static void TextClassification(Device device)
    {
        var random = new Random(12);
        string[] people = ["Dr. Heba", "Dr. Omar", "Dr. Sara", "د. هبة", "د. عمر", "د. سارة"];
        string[] days = ["Tuesday", "Monday", "Friday", "الخميس", "الاحد", "السبت"];
        string Pick(string[] values) => values[random.Next(values.Length)];
        var templates = new (string Label, Func<string> Make)[]
        {
            ("book", () => random.Next(2) == 0 ? $"book me with {Pick(people)} on {Pick(days)}" : $"عايز احجز مع {Pick(people)} يوم {Pick(days)}"),
            ("ask", () => random.Next(2) == 0 ? $"what are the available days for {Pick(people)}?" : $"ايه المواعيد المتاحة مع {Pick(people)}"),
            ("greet", () => Pick(["hi", "hello", "hey", "هلا", "مرحبا", "اهلا"]) + " " + Pick(["there", "friend", "team", "يا صديقي", "صاحبي", "حبيبي"])
                            + " " + Pick(["", "good morning", "صباح الخير", "!", "how are you"])),
        };
        var examples = Enumerable.Range(0, 600).Select(i => templates[i % 3]).Select(t => new LabeledText(t.Make(), t.Label)).ToList();

        var (train, test) = TextClassifier.Split(examples, 0.25, seed: 3);
        var trainTexts = train.Select(e => e.Text.ToLowerInvariant()).ToHashSet();
        Check(test.Count > 0 && test.All(e => !trainTexts.Contains(e.Text.ToLowerInvariant())), "no test text is also a training text");

        using var classifier = TextClassifier.Train(train, new TextClassifierOptions { Device = device, Epochs = 12, Buckets = 4096, Hidden = 64, Seed = 1 });
        var report = classifier.Evaluate(test);
        Check(report.Accuracy >= 0.95 && report.Labels.SequenceEqual(["ask", "book", "greet"]), $"held-out accuracy:\n{report}");
        var prediction = classifier.Predict("please book me with Dr. Omar on Monday");
        Check(prediction.Label == "book" && prediction.Probabilities.Count == 3 && Math.Abs(prediction.Probabilities.Sum(p => p.Probability) - 1f) < 1e-4f,
            $"prediction {prediction.Label} {prediction.Confidence}");

        string path = Path.Combine(Path.GetTempPath(), $"ns-text-{Guid.NewGuid():N}.nsm");
        try
        {
            classifier.Save(path);
            using var loaded = TextClassifier.Load(path, device);
            var texts = test.Take(40).Select(e => e.Text).ToList();
            var before = classifier.Predict(texts);
            var after = loaded.Predict(texts);
            Check(before.Zip(after).All(p => p.First.Label == p.Second.Label && Math.Abs(p.First.Confidence - p.Second.Confidence) < 1e-5f),
                "a loaded model predicts as the saved one");

            // Predictions from several threads at once.
            var parallel = new TextPrediction[texts.Count];
            Parallel.For(0, texts.Count, i => parallel[i] = loaded.Predict(texts[i]));
            Check(parallel.Zip(after).All(p => p.First.Label == p.Second.Label), "concurrent predictions");
        }
        finally
        {
            File.Delete(path);
        }

        string csv = Path.Combine(Path.GetTempPath(), $"ns-text-{Guid.NewGuid():N}.csv");
        try
        {
            File.WriteAllText(csv, "﻿raw_question,other,intent\n\"line one\nline, two with \"\"quotes\"\"\",x,retrieve\nهلا,y,direct_reply\n,z,retrieve\n");
            var rows = TextClassifier.ReadCsv(csv, "raw_question", "intent");
            Check(rows.Count == 2 && rows[0].Text == "line one\nline, two with \"quotes\"" && rows[0].Label == "retrieve" && rows[1].Text == "هلا",
                $"CSV rows: {string.Join(" | ", rows)}");
        }
        finally
        {
            File.Delete(csv);
        }
    }
}

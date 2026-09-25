"""Chapter 30 — Anomaly Detection with Autoencoders."""
from gen import *

PART = "V"


def ae_svg():
    w, h = 470, 120
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">']
    layers = [(8, 40), (16, 130), (2, 235), (16, 340), (8, 430)]
    for n, x in layers:
        hgt = 6 + n * 5.5
        p.append(f'<rect x="{x - 14}" y="{60 - hgt / 2}" width="28" height="{hgt}" rx="4" fill="{"#a15c00" if n == 2 else "#e6f2ef"}" stroke="#0f6b5c"/>')
        p.append(svg_text(x, 60 + hgt / 2 + 12, str(n), 8, "#56606a"))
    for (_, x1), (_, x2) in zip(layers, layers[1:]):
        p.append(f'<line x1="{x1 + 14}" y1="60" x2="{x2 - 14}" y2="60" stroke="#56606a"/>')
    p.append(svg_text(40, 14, "readings", 7.8, "#0f6b5c"))
    p.append(svg_text(235, 14, "bottleneck", 7.8, "#a15c00"))
    p.append(svg_text(430, 14, "rebuilt readings", 7.8, "#0f6b5c"))
    p.append(svg_text(235, 116, "anomaly score = mean squared difference between input and output", 7.4, "#56606a"))
    p.append("</svg>")
    return "".join(p)


PROGRAM = """
    using NeuralSharp;
    using NeuralSharp.Data;
    using NeuralSharp.Layers;
    using NeuralSharp.Optimizers;
    using NeuralSharp.Training;

    Device.Default = args.Contains("--cuda") ? Device.Cuda() : Device.Cpu;

    // ---- machine sensors: 8 readings driven by 2 hidden factors (load, temperature) + noise
    var rng = new Random(9);
    float[] Reading(bool faulty)
    {
        float load = rng.NextSingle() * 2 - 1, temp = rng.NextSingle() * 2 - 1;
        float[] r =
        [
            load, 0.8f * load + 0.2f * temp, temp, 0.5f * load - 0.5f * temp,
            load * temp, 0.3f * load + 0.7f * temp, -load, 0.6f * temp,
        ];
        for (int k = 0; k < 8; k++) r[k] += 0.05f * (rng.NextSingle() - 0.5f);
        if (faulty)                                             // one sensor goes wrong
        {
            int k = rng.Next(8);
            r[k] += (rng.Next(2) == 0 ? -1 : 1) * (0.6f + rng.NextSingle());
        }
        return r;
    }
    float[,] Rows(int n, Func<int, bool> faulty, out bool[] labels)
    {
        var m = new float[n, 8];
        labels = new bool[n];
        for (int i = 0; i < n; i++)
        {
            labels[i] = faulty(i);
            var r = Reading(labels[i]);
            for (int k = 0; k < 8; k++) m[i, k] = r[k];
        }
        return m;
    }

    // Train on NORMAL data only; the targets are the inputs themselves.
    var normal = Rows(4000, _ => false, out _);
    var (train, validation) = Dataset.FromArrays(normal, normal).Split(0.9, seed: 1);
    var testRows = Rows(2000, i => i % 50 == 0, out bool[] isFaulty);          // 2% faulty

    static Sequential Build(Random? r = null) => new()
    {
        new Linear(8, 16, random: r), new Tanh(),
        new Linear(16, 2, random: r),                            // the bottleneck
        new Linear(2, 16, random: r), new Tanh(),
        new Linear(16, 8, random: r),
    };
    using var model = Build(new Random(2));
    using var optimizer = new Adam(model.Parameters(), 3e-3f);
    var trainer = new Trainer(model, optimizer, Losses.MeanSquaredError) { EarlyStoppingPatience = 10 };
    var history = trainer.Fit(new DataLoader(train, 64, shuffle: true, seed: 1), epochs: 200,
                              validation: new DataLoader(validation, 512));
    Console.WriteLine($"trained {history.Epochs.Count} epochs, best validation reconstruction MSE {history.BestLoss:F5}");
"""

SCORE = """
    // ---- anomaly score: reconstruction error per row
    float[] Errors(float[,] rows)
    {
        using var input = Tensor.From(rows);
        using var output = model.Predict(input);
        var o = output.ToArray2D();
        var e = new float[rows.GetLength(0)];
        for (int i = 0; i < e.Length; i++)
            for (int k = 0; k < 8; k++)
                e[i] += (o[i, k] - rows[i, k]) * (o[i, k] - rows[i, k]) / 8;
        return e;
    }

    // Threshold: the 99th percentile of the errors on normal validation data.
    var validationErrors = Errors(validation.FeaturesToArray()).Order().ToArray();
    float threshold = validationErrors[(int)(0.99 * validationErrors.Length)];
    Console.WriteLine($"threshold (99th percentile of normal errors) {threshold:F5}");

    var errors = Errors(testRows);
    int tp = 0, fp = 0, fn = 0;
    for (int i = 0; i < errors.Length; i++)
    {
        bool flagged = errors[i] > threshold;
        if (flagged && isFaulty[i]) tp++; else if (flagged) fp++; else if (isFaulty[i]) fn++;
    }
    Console.WriteLine($"test: {isFaulty.Count(f => f)} faulty of {errors.Length}; caught {tp}, missed {fn}, false alarms {fp}");
"""


def build():
    return page(
        chapter_open(
            "anomaly",
            "Often you have plenty of normal data and almost no examples of failures, fraud or defects, and the next "
            "anomaly may look like nothing seen before. An autoencoder learns to compress and rebuild normal data; "
            "anything it rebuilds badly is unusual. This project detects faulty sensor readings that way, without a "
            "single faulty example during training.",
            "Task type: <b>unsupervised anomaly detection</b>. Train with the inputs as targets (<code>FromArrays(x, x)</code>) and MSE.",
            "A narrow bottleneck forces the model to learn the structure of normal data.",
            "Score = reconstruction error; threshold = a high percentile (e.g. 99th) of scores on normal validation data.",
            "Result: all 40 faulty readings caught; 37 false alarms among 1,960 normal readings (1.9%).",
            "Works for sensors, transactions, network traffic, log statistics: any fixed-length numeric record.",
        ),
        diagram("Figure 30.1 — The autoencoder", ae_svg(),
                "Eight readings are squeezed through two numbers. Normal readings, which really depend on two factors, pass almost unchanged."),
        h2("30.1 Training on normal data"),
        snippet(PROGRAM, caption="Program.cs, part 1: data and training"),
        output("""
            trained 72 epochs, best validation reconstruction MSE 0.00023
            """),
        h2("30.2 Scoring and the threshold"),
        snippet(SCORE, caption="Program.cs, part 2: reconstruction errors and detection"),
        output("""
            threshold (99th percentile of normal errors) 0.00063
            test: 40 faulty of 2000; caught 40, missed 0, false alarms 37
            """),
        para("The typical (median) error is 0.00021 for normal readings and 0.10 for faulty ones: faults stand out by "
             "a factor of about 500. The 99th-percentile threshold accepts that about 1% of normal readings are flagged; "
             "here 37 of 1,960 (1.9%). Raise the percentile (99.9th) for fewer false alarms at the risk of missing "
             "subtle faults."),
        cpugpu("scoring a stream of readings in a service",
               """
               using var detector = Build();
               detector.Load("sensors.weights");
               detector.Eval();
               float threshold = float.Parse(File.ReadAllText("sensors.threshold"));
               float Score(float[] reading)
               {
                   using var input = Tensor.From(reading, [1, 8]);
                   using var output = detector.Predict(input);
                   var o = output.ToArray();
                   return o.Select((v, k) => (v - reading[k]) * (v - reading[k])).Sum() / 8;
               }
               bool IsAnomaly(float[] reading) => Score(reading) > threshold;
               """,
               """
               using var detector = Build();
               detector.To(Device.Cuda());
               detector.Load("sensors.weights");
               // score readings in batches of thousands per Predict call for throughput
               """,
               "Save the threshold with the model: it is part of the detector. If the readings need scaling, save the scaler too."),
        h2("30.3 Making it work on real data"),
        reftable(["Issue", "Remedy"], [
            ["Features on different scales", "Standardize with a scaler fitted on normal training data"],
            ["Some training data is already anomalous", "Usually fine if rare; or train, remove the top 1% of scores, and retrain"],
            ["Too many false alarms", "Higher percentile; a larger bottleneck; more training data covering all normal operating modes"],
            ["Anomalies slip through", "A narrower bottleneck; per-feature errors (which sensor is off?)"],
            ["Normal behaviour drifts (seasons, wear)", "Retrain regularly on recent normal data; monitor the score distribution"],
            ["Sequences rather than single readings", "Autoencode windows (flattened, or with LSTM encoder/decoder)"],
            ["A few labelled anomalies exist", "Use them to choose the threshold (maximize recall at acceptable precision, " + ch("binary") + ")"],
        ], caption="Table 30.1 — Practical issues"),
        trap("a bottleneck that is too wide",
             "<p>A bottleneck as wide as the input lets the network learn to copy any input, faults included, so errors "
             "shrink for everything and the gap between normal and faulty data narrows (Practice 4 measures it). Start "
             "with a bottleneck close to the number of factors that really drive the data and widen only until normal "
             "data is rebuilt well.</p>"),
        practice([
            (1, "Why does the dataset use the features as targets?",
             "An autoencoder learns to reproduce its input; the reconstruction error then measures how typical an input is."),
            (1, "What fraction of normal readings should a 99.5th-percentile threshold flag?",
             "About 0.5%."),
            (2, "Report which sensor caused each alarm.",
             "Keep the per-feature squared errors instead of averaging them; the sensor with the largest error is the likely culprit."),
            (2, "Widen the bottleneck to 8 units and rerun. What happens?",
             "Measured: normal readings are rebuilt better (median error 0.00003), but faults are rebuilt better too "
             "(median 0.010 instead of 0.10). The detector caught 38 of 40 faults with 44 false alarms, against 40 and 37 "
             "with the 2-unit bottleneck: a wider bottleneck starts to copy faults as well."),
            (3, "Adapt the detector to credit-card transactions (amount, hour, merchant category, distance from home, …).",
             "Scale numeric columns, embed or one-hot the merchant category (" + ch("recommender") + "), train on transactions "
             "believed legitimate, choose the threshold with the few known fraud cases, and review flagged transactions "
             "by hand before acting."),
        ], PART),
        footer("Anomaly detection", "Autoencoder", "Bottleneck", "Reconstruction error", "Percentile",
               "Unsupervised learning", "False alarm"),
    )

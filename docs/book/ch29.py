"""Chapter 29 — Time-Series Forecasting."""
from gen import *

PART = "V"


def windows_svg():
    w, h = 470, 110
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">']
    for i in range(22):
        x = 12 + i * 20
        fill = "#fff4e2" if i < 16 else "#e6f2ef"
        p.append(f'<rect x="{x}" y="20" width="18" height="18" rx="2" fill="{fill}" stroke="#b7c3c8"/>')
    for k, y in enumerate((50, 72)):
        x0 = 12 + k * 20
        p.append(f'<rect x="{x0}" y="{y}" width="{7 * 20 - 2}" height="14" rx="2" fill="#0f6b5c" opacity="0.85"/>')
        p.append(f'<rect x="{x0 + 7 * 20}" y="{y}" width="18" height="14" rx="2" fill="#a15c00"/>')
        p.append(svg_text(x0 + 7 * 20 + 32, y + 11, "target", 7.4, "#a15c00", "start"))
    p.append(svg_text(12 + 3.5 * 20, 104, "window of T past values", 7.4, "#0f6b5c"))
    p.append(svg_text(12 + 8 * 20, 14, "training period", 7.6, "#a15c00"))
    p.append(svg_text(12 + 19 * 20, 14, "test period", 7.6, "#0f6b5c"))
    p.append("</svg>")
    return "".join(p)


PROGRAM = """
    using NeuralSharp;
    using NeuralSharp.Data;
    using NeuralSharp.Diagnostics;
    using NeuralSharp.Layers;
    using NeuralSharp.Optimizers;
    using NeuralSharp.Training;

    Device.Default = args.Contains("--cuda") ? Device.Cuda() : Device.Cpu;

    // ---- a daily series: trend + weekly pattern + yearly season + noise (e.g. store visitors)
    var rng = new Random(5);
    float[] series = new float[1200];
    for (int d = 0; d < series.Length; d++)
        series[d] = 200 + 0.05f * d + 30 * MathF.Sin(2 * MathF.PI * d / 7)
                  + 50 * MathF.Sin(2 * MathF.PI * d / 365) + 8 * (rng.NextSingle() - 0.5f) * 2;

    // ---- split BY TIME: the first 1000 days train, the last 200 test
    const int T = 28;                                            // look back four weeks
    int split = 1000;
    var scale = StandardScaler.Fit(series.AsSpan(0, split), 1);   // training period only
    float[] scaled = (float[])series.Clone();
    scale.Transform(scaled, 1);

    Dataset Windows(int from, int to)                             // windows whose TARGET day is in [from, to)
    {
        int count = to - Math.Max(from, T);
        var x = new float[count, T];
        var y = new float[count, 1];
        for (int i = 0; i < count; i++)
        {
            int target = Math.Max(from, T) + i;
            for (int k = 0; k < T; k++) x[i, k] = scaled[target - T + k];
            y[i, 0] = scaled[target];
        }
        return Dataset.FromArrays(x, y).WithFeatureShape(T, 1);  // batches: [N, T, 1]
    }
    var train = Windows(0, split);
    var test = Windows(split, series.Length);
    Console.WriteLine($"{train.Count} training windows, {test.Count} test windows");

    static Sequential Build(Random? r = null) => new()
    {
        new GRU(1, 32, random: r),                                // reads the window step by step
        new Linear(32, 1, random: r),                             // next value
    };
    using var model = Build(new Random(1));
    using var optimizer = new Adam(model.Parameters(), 3e-3f);
    var trainer = new Trainer(model, optimizer, Losses.MeanSquaredError)
    {
        EarlyStoppingPatience = 8,
        MaxGradientNorm = 1f,                                     // recurrent nets: clip
    };
    using (Telemetry.Subscribe(new ConsoleLogger(TelemetryLevel.Training, epochInterval: 10)))
        trainer.Fit(new DataLoader(train, 32, shuffle: true, seed: 1), epochs: 60,
                    validation: new DataLoader(test, 256));
"""

EVAL = """
    // ---- one-step-ahead error in real units, against two simple baselines
    float[,] pred = trainer.Predict(test);
    float[] p = [.. Enumerable.Range(0, test.Count).Select(i => pred[i, 0])];
    scale.InverseTransform(p, 1);
    double mae = 0, naive = 0, seasonal = 0;
    for (int i = 0; i < test.Count; i++)
    {
        int day = split + i;
        mae += Math.Abs(p[i] - series[day]);
        naive += Math.Abs(series[day - 1] - series[day]);         // tomorrow = today
        seasonal += Math.Abs(series[day - 7] - series[day]);      // tomorrow = same weekday last week
    }
    Console.WriteLine($"MAE next day: GRU {mae / test.Count:F1}   'same as yesterday' {naive / test.Count:F1}" +
                      $"   'same as last week' {seasonal / test.Count:F1}");

    // ---- 14-day forecast: feed each prediction back in as the newest value
    var window = scaled[(split - T)..split].ToList();
    var forecast = new List<float>();
    model.Eval();
    for (int step = 0; step < 14; step++)
    {
        using var input = Tensor.From([.. window.TakeLast(T)], [1, T, 1]);
        using var next = model.Predict(input);
        float value = next.Item();
        window.Add(value);
        forecast.Add(value);
    }
    float[] f = [.. forecast];
    scale.InverseTransform(f, 1);
    Console.WriteLine("day   forecast   actual");
    for (int d = 0; d < 14; d += 2) Console.WriteLine($"{split + d,4} {f[d],9:F1} {series[split + d],8:F1}");
    Console.WriteLine($"14-day MAE {Enumerable.Range(0, 14).Average(d => Math.Abs(f[d] - series[split + d])):F1}");
"""


TRANSFORMER = """
    static Sequential BuildTransformer(int T, Random? r = null) => new()
    {
        new Linear(1, 32, random: r),                   // each value → 32 numbers
        new PositionalEncoding(T, 32),                  // its place in the window
        new TransformerEncoderLayer(32, heads: 4, ffDim: 64, dropout: 0f, random: r),
        new TransformerEncoderLayer(32, heads: 4, ffDim: 64, dropout: 0f, random: r),
        new LayerNorm(32),
        // keep the newest step (it has attended to the whole window)
        new Lambda(x => x.Narrow(1, T - 1, 1).Reshape(-1, 32), "LastStep"),
        new Linear(32, 1, random: r),                   // next value
    };
"""

COMPARISON = """
    All 1,000 training days: window 28, 972 training windows
      model         params       next-day MAE         14-day MAE  epochs  train s
      GRU            3,297      5.1 (4.8–5.7)      6.2 (5.3–7.0)      24      1.8
      Transformer   17,249      5.0 (4.8–5.2)     9.8 (4.9–14.2)      23      7.9
    Only the last 250 training days: window 28, 250 training windows
      GRU            3,297      5.3 (5.2–5.5)      5.9 (5.4–6.2)      46      1.3
      Transformer   17,249      5.1 (5.1–5.2)      5.0 (4.0–6.8)      35      3.6
    Longer window (16 weeks): window 112, 888 training windows
      GRU            3,297      5.1 (4.9–5.2)      6.3 (4.4–8.2)      30      7.2
      Transformer   17,249      4.7 (4.6–5.0)      5.5 (3.9–6.9)      30     48.7
"""

LATENCY = """
    window   28  GRU               602 µs      Transformer       787 µs
    window  112  GRU             1,193 µs      Transformer     1,994 µs
    window  448  GRU             3,306 µs      Transformer     8,836 µs
"""


def build():
    return page(
        chapter_open(
            "timeseries",
            "Forecasting predicts future values of a series from its past: sales, energy load, sensor readings, "
            "visitors. The network sees a window of recent values and predicts the next one; forecasting further "
            "ahead feeds predictions back in. This project forecasts a daily series with trend and weekly and yearly "
            "patterns using a GRU, and, most importantly, compares it with simple baselines, which every forecasting "
            "project must beat.",
            "Task type: <b>regression on windows</b>. Features <code>[N, T, 1]</code> (T past values), target the next value.",
            "Split by time, never randomly: the test period comes after the training period.",
            "Fit the scaler on the training period only.",
            "Result: next-day MAE 5.7, against 6.6 for \"same weekday last week\" and 17.1 for \"same as yesterday\".",
            "Multi-step forecasts feed each prediction back; errors grow with the horizon (14-day MAE 6.3).",
            "GRU or transformer (Section 29.4): the same accuracy within noise; the GRU trains 3–7× faster with 5× fewer parameters.",
        ),
        h2("29.1 Windows"),
        diagram("Figure 29.1 — Sliding windows over a series", windows_svg(),
                "Each training sample is T consecutive values and the value that follows; windows are built separately for the training and test periods."),
        snippet(PROGRAM, caption="Program.cs, part 1: series, windows, GRU, training"),
        output("""
            972 training windows, 200 test windows
            Sequential(2 layers)
              GRU(1 -> 32)  [3,264 params]
              Linear(32 -> 1)  [33 params]
            Total trainable parameters: 3,297
            Training on cpu (CPU (4 threads, 8-wide SIMD)) | 972 samples, 200 validation | batch 32, 31 steps/epoch | Adam lr=0.003 | 3,297 parameters | 4 CPU threads
            Epoch  1/60  loss 0.490665  val_loss 0.379358  422.2 ms  2,302 samples/s  *
            Epoch 10/60  loss 0.021693  val_loss 0.029009  59.4 ms  16,365 samples/s
            Finished 14 epochs in 1.77 s (early stop) | best epoch 6 loss 0.026781
            """),
        h2("29.2 Beating the baselines"),
        snippet(EVAL, caption="Program.cs, part 2: evaluation and a 14-day forecast"),
        output("""
            MAE next day: GRU 5.7   'same as yesterday' 17.1   'same as last week' 6.6
            day   forecast   actual
            1000     176.0    179.9
            1002     219.2    215.7
            1004     214.9    216.0
            1006     171.7    166.0
            1008     191.7    205.7
            1010     224.4    235.2
            1012     189.4    193.1
            14-day MAE 6.3
            """),
        para("\"Same weekday last week\" is a strong baseline for a series with a weekly rhythm: it already captures "
             "most of the pattern. The GRU improves on it by also using the trend and the latest level. The added "
             "noise has an average size of 4, so an MAE near 5–6 is close to the best possible. Always report a "
             "baseline: a network that cannot beat \"same as last week\" is not worth deploying."),
        honestbox("Why time must not be shuffled",
                  "<p>A random split puts days from the future into the training set and windows that overlap test windows "
                  "into both sets, so the test error measures memory, not forecasting. The same applies to the scaler: "
                  "statistics that include the future leak information (glossary <b>Data leakage</b>).</p>"),
        cpugpu("running the project",
               """
               dotnet run -c Release
               """,
               """
               dotnet run -c Release -- --cuda
               // a GRU over 28 steps with 32 units is small: expect the CPU to be as fast.
               // With many series (batch 512+) or hidden sizes of 128+, the GPU wins.
               """),
        h2("29.3 Extensions"),
        reftable(["Need", "How"], [
            ["Several input series (weather, price, promotions)", "Windows of shape <code>[T, F]</code>: <code>WithFeatureShape(T, F)</code> and <code>GRU(F, h)</code>"],
            ["Calendar effects (weekday, holiday)", "Add them as extra input columns per step (one-hot or embedded)"],
            ["Predict H days at once", "Targets of H values and <code>Linear(h, H)</code>: no feedback, errors do not compound"],
            ["Many related series (all stores)", "Train one model on windows from every series; add a series embedding (" + ch("recommender") + ")"],
            ["Uncertainty", "Train two outputs (value and spread) or use Monte-Carlo dropout (" + ch("norm") + ")"],
            ["Long histories", "A longer window, or a transformer (" + ch("attention") + "; compared in Section 29.4)"],
        ], caption="Table 29.1 — Forecasting extensions"),
        trap("forecasting with the training-period scaler forgotten",
             "<p>The model predicts scaled values; apply <code>InverseTransform</code> with the same scaler, and when feeding "
             "predictions back, keep them scaled (as the loop above does) and unscale only for output.</p>"),
        h2("29.4 GRU or transformer? Measured"),
        para("A GRU reads the window one step at a time and carries a state; a transformer lets every step attend to "
             "every other step at once (" + ch("recurrent") + ", " + ch("attention") + "). Which is better for "
             "forecasting? The comparison below keeps everything else fixed: the same series, windows, scaler, "
             "optimizer (Adam 3e-3), clipping, early stopping (patience 8, at most 60 epochs) and evaluation, and "
             "repeats each run with three seeds, because a single run of a small model can mislead."),
        snippet(TRANSFORMER, caption="The transformer forecaster: a drop-in replacement for Build()"),
        output(COMPARISON, caption="GRU vs transformer on the CPU (mean over seeds 1–3, with the range)"),
        para("<b>Accuracy is a tie.</b> The next-day errors differ by 0.1–0.4 visitors on a series whose noise alone "
             "averages 4, and the ranges overlap; both beat the baselines of Section 29.2 (6.6 and 17.1) in every setting. The transformer is slightly ahead with the longer window and, "
             "perhaps surprisingly, with only 250 training days; the GRU was not better on little data here. The "
             "14-day figures come from a single two-week stretch and swing widely with the seed (3.9 to 14.2 for "
             "the transformer), so they cannot separate the models."),
        para("<b>Cost is not a tie.</b> The GRU has 3,297 parameters against 17,249, trains 2.8 to 6.8 times faster, and "
             "answers faster, with a gap that grows with the window: a transformer's attention compares every step "
             "with every other (work growing with the square of the window), a GRU does a fixed amount of work per step."),
        output(LATENCY, caption="Latency of one prediction (batch 1, CPU, after warm-up)"),
        reftable(["Situation", "Choose", "Why"], [
            ["Short windows, small data, a CPU or small device", "GRU (or LSTM)", "Same accuracy, far cheaper to train and run"],
            ["Readings arrive one at a time (streaming sensors)", "GRU", "Work per step is fixed; a recurrent state can be carried from value to value, while attention re-reads the window"],
            ["Long windows, many series, a GPU", "Transformer", "Attention reaches any step directly and runs in parallel on the GPU"],
            ["Text, retrieval, generation", "Transformer", ch("gpt") + ", " + ch("reranker") + ", " + ch("summarizer")],
            ["Unsure", "Both, measured", "Run this comparison on your own data with several seeds and a baseline"],
        ], caption="Table 29.2 — Choosing the sequence model"),
        honestbox("One series, one conclusion",
                  "<p>These numbers come from one synthetic series with a clean weekly and yearly rhythm. On data with long, "
                  "irregular dependencies, or with many series and a GPU, the balance can move towards the transformer; "
                  "on tiny devices it moves further towards the GRU. The method transfers: fix everything else, vary the "
                  "model, repeat with several seeds, and compare against a baseline.</p>"),
        practice([
            (1, "Why is the test set built with <code>Windows(split, series.Length)</code> rather than from a random 20% of all windows?",
             "Forecasts must be evaluated on a period after the training data; random windows would overlap training windows and leak the future."),
            (1, "What does a next-day MAE of 5.7 mean for this series?",
             "Tomorrow's predicted value is on average 5.7 units (visitors) away from the actual value."),
            (2, "Replace the GRU with an LSTM. Compare MAE and training time.",
             "Swap <code>new GRU(1, 32, …)</code> for <code>new LSTM(1, 32, …)</code>; expect similar accuracy with about a third more "
             "parameters and slightly longer epochs (" + ch("recurrent") + ")."),
            (2, "Change the model to predict the next 7 days in one shot.",
             "Make each target the 7 values after the window (<code>float[count, 7]</code>), end with <code>Linear(32, 7)</code>, and "
             "stop windows 7 days before the end of each period."),
            (2, "The transformer's 14-day MAE ranged from 4.9 to 14.2 over three seeds. What would you do before claiming one model forecasts two weeks better?",
             "Evaluate many forecast origins (for example a 14-day forecast starting at every day of the test period) and "
             "more seeds, then compare the averages and their spread; one two-week stretch is a sample of one."),
            (3, "Add the weekday as a second input feature per step and compare against the univariate model.",
             "Build windows <code>[T, 2]</code> with the scaled value and weekday/6 at each step, use <code>WithFeatureShape(T, 2)</code> and "
             "<code>GRU(2, 32)</code>. With an explicit weekday the model needs less effort to learn the weekly rhythm."),
        ], PART),
        footer("Time series", "Forecasting", "Sliding window", "Baseline", "Horizon", "Recursive forecasting",
               "Data leakage", "GRU", "Transformer", "Attention"),
    )

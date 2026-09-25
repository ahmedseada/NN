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
            ["Long histories", "A causal transformer (" + ch("attention") + ", Practice 12.5) or a longer window"],
        ], caption="Table 29.1 — Forecasting extensions"),
        trap("forecasting with the training-period scaler forgotten",
             "<p>The model predicts scaled values; apply <code>InverseTransform</code> with the same scaler, and when feeding "
             "predictions back, keep them scaled (as the loop above does) and unscale only for output.</p>"),
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
            (3, "Add the weekday as a second input feature per step and compare against the univariate model.",
             "Build windows <code>[T, 2]</code> with the scaled value and weekday/6 at each step, use <code>WithFeatureShape(T, 2)</code> and "
             "<code>GRU(2, 32)</code>. With an explicit weekday the model needs less effort to learn the weekly rhythm."),
        ], PART),
        footer("Time series", "Forecasting", "Sliding window", "Baseline", "Horizon", "Recursive forecasting",
               "Data leakage", "GRU"),
    )

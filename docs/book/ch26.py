"""Chapter 26 — Binary Classification."""
from gen import *

PART = "V"

PROGRAM = """
    using NeuralSharp;
    using NeuralSharp.Data;
    using NeuralSharp.Diagnostics;
    using NeuralSharp.Layers;
    using NeuralSharp.Optimizers;
    using NeuralSharp.Training;

    Device.Default = args.Contains("--cuda") ? Device.Cuda() : Device.Cpu;

    // ---- data: 5,000 synthetic customers, about 1 in 6 churns (replace with Dataset.LoadCsv)
    string[] columns = ["tenure_months", "monthly_fee", "support_calls", "contract_years", "usage_hours"];
    var rng = new Random(11);
    var x = new float[5000, 5];
    var y = new float[5000, 1];
    for (int i = 0; i < 5000; i++)
    {
        float tenure = rng.Next(1, 73), fee = 20 + 100 * rng.NextSingle(), calls = rng.Next(0, 8);
        float contract = rng.Next(0, 3), usage = 5 + 60 * rng.NextSingle();
        double score = -1.9 - 0.04 * tenure + 0.02 * fee + 0.45 * calls - 0.9 * contract - 0.03 * usage
                     + 0.8 * (rng.NextDouble() - 0.5);
        x[i, 0] = tenure; x[i, 1] = fee; x[i, 2] = calls; x[i, 3] = contract; x[i, 4] = usage;
        y[i, 0] = rng.NextDouble() < 1 / (1 + Math.Exp(-score)) ? 1 : 0;
    }
    var data = Dataset.FromArrays(x, y, columns, ["churn"]);
    Console.WriteLine($"{data.Count} customers, churn rate {data.Targets.ToArray().Average():P1}");
    var (train, test) = data.Split(0.8, seed: 3);
    var scaler = StandardScaler.FitFeatures(train);
    train = train.Scale(scaler);                  // targets are 0/1: only features are scaled
    test = test.Scale(scaler);

    // ---- model: one raw score out, no Sigmoid layer
    static Sequential Build(Random? r = null) => new()
    {
        new Linear(5, 32, random: r), new ReLU(), new Dropout(0.1f, r),
        new Linear(32, 32, random: r), new ReLU(),
        new Linear(32, 1, random: r),
    };
    using var model = Build(new Random(1));
    using var optimizer = new AdamW(model.Parameters(), 3e-3f, weightDecay: 1e-4f);
    var trainer = new Trainer(model, optimizer, Losses.BinaryCrossEntropyWithLogits)
    {
        Metrics = { Metric.BinaryAccuracy(threshold: 0f) },     // score 0 <=> probability 0.5
        EarlyStoppingPatience = 10,
    };
    using (Telemetry.Subscribe(new ConsoleLogger(TelemetryLevel.Training, epochInterval: 10)))
        trainer.Fit(new DataLoader(train, 64, shuffle: true, seed: 1), epochs: 100,
                    validation: new DataLoader(test, 512));
"""

EVAL = """
    // ---- probabilities, then precision/recall at three thresholds
    float[,] logits = trainer.Predict(test);
    var p = new float[test.Count];
    var t = new float[test.Count];
    for (int i = 0; i < test.Count; i++)
    {
        p[i] = 1f / (1f + MathF.Exp(-logits[i, 0]));          // sigmoid
        t[i] = test.GetTargets(i)[0];
    }
    Console.WriteLine("threshold  accuracy  precision  recall    F1");
    foreach (float threshold in new[] { 0.3f, 0.5f, 0.7f })
    {
        int tp = 0, fp = 0, fn = 0, tn = 0;
        for (int i = 0; i < p.Length; i++)
        {
            bool predicted = p[i] >= threshold, actual = t[i] == 1;
            if (predicted && actual) tp++; else if (predicted) fp++; else if (actual) fn++; else tn++;
        }
        double precision = tp / (double)Math.Max(tp + fp, 1), recall = tp / (double)Math.Max(tp + fn, 1);
        double f1 = 2 * precision * recall / Math.Max(precision + recall, 1e-9);
        Console.WriteLine($"  {threshold,7:F1}  {(tp + tn) / (double)p.Length,8:P1}  {precision,9:P1}  {recall,6:P1}  {f1,5:F3}");
    }

    // ROC AUC: the chance that a random churner gets a higher score than a random non-churner
    var pos = p.Where((_, i) => t[i] == 1).ToArray();
    var neg = p.Where((_, i) => t[i] == 0).ToArray();
    double auc = pos.Sum(a => neg.Count(b => a > b) + 0.5 * neg.Count(b => a == b)) / ((double)pos.Length * neg.Length);
    Console.WriteLine($"ROC AUC {auc:F3}   (always 'no churn' would be {1 - t.Average():P1} accurate)");
"""

SERVE = """
    model.Save("churn.weights");
    scaler.Save("churn.scaler");

    // ---- later / elsewhere: score new customers
    using var served = Build();
    served.Load("churn.weights");
    served.Eval();
    var fx = StandardScaler.Load("churn.scaler");
    float[] customers = [3, 95, 5, 0, 10,   60, 40, 0, 2, 45];
    fx.Transform(customers, 5);
    using var input = Tensor.From(customers, [2, 5]);
    using var output = served.Predict(input);
    using var probability = output.Sigmoid();
    var risk = probability.ToArray();
    Console.WriteLine($"new customer A (3 months, $95, 5 calls, monthly): churn risk {risk[0]:P0}");
    Console.WriteLine($"new customer B (60 months, $40, 0 calls, 2-year):  churn risk {risk[1]:P0}");
"""


def build():
    return page(
        chapter_open(
            "binary",
            "Many business questions have a yes/no answer: will this customer leave, is this transaction fraud, will "
            "this machine fail, is this email spam. This project predicts customer churn from five account features. "
            "Beyond training, it shows what makes binary classification different: turning scores into "
            "probabilities, choosing a decision threshold, and judging a model with precision, recall and ROC AUC "
            "when one answer is much rarer than the other.",
            "Task type: <b>binary classification</b>. Last layer <code>Linear(h, 1)</code>, loss <code>BinaryCrossEntropyWithLogits</code>.",
            "Probability = <code>Sigmoid()</code> of the output; the decision threshold is a business choice, not always 0.5.",
            "With 17.5% churners, \"nobody churns\" is already about 83% accurate: report precision, recall and AUC.",
            "Result: ROC AUC 0.829; at threshold 0.3 the model catches 60% of churners.",
            "Save the weights and the feature scaler; serve probabilities.",
        ),
        h2("26.1 The program"),
        para("The whole project is one <code>Program.cs</code>. Synthetic data keeps it self-contained; for real data "
             "replace the generator with <code>Dataset.LoadCsv(\"customers.csv\", new CsvOptions { TargetColumns = "
             "[\"churn\"], IgnoreColumns = [\"customer_id\"] })</code>, where the churn column holds 0 or 1."),
        snippet(PROGRAM, caption="Program.cs, part 1: data, model, training"),
        output("""
            5000 customers, churn rate 17.5 %
            Training on cpu (CPU (4 threads, 8-wide SIMD)) | 4,000 samples, 1,000 validation | batch 64, 63 steps/epoch | AdamW lr=0.003 | 1,281 parameters | 4 CPU threads
            Epoch   1/100  loss 0.416125  accuracy 0.8233  val_loss 0.354400  val_accuracy 0.8560  97.6 ms  40,990 samples/s  *
            Epoch  10/100  loss 0.343585  accuracy 0.8468  val_loss 0.349250  val_accuracy 0.8520  20.8 ms  191,870 samples/s  *
            Epoch  20/100  loss 0.337062  accuracy 0.8545  val_loss 0.358105  val_accuracy 0.8540  24.3 ms  164,784 samples/s
            Finished 28 epochs in 0.80 s (early stop) | best epoch 18 loss 0.348009
            """, caption="Training output (model summary omitted)"),
        h2("26.2 Thresholds, precision and recall"),
        snippet(EVAL, caption="Program.cs, part 2: evaluation"),
        output("""
            threshold  accuracy  precision  recall    F1
                  0.3    81.1 %     46.2 %  59.9 %  0.522
                  0.5    85.2 %     63.6 %  32.6 %  0.431
                  0.7    84.7 %     88.0 %  12.8 %  0.223
            ROC AUC 0.829   (always 'no churn' would be 82.8 % accurate)
            """, caption="Evaluation output"),
        reftable(["Measure", "Question it answers", "Here, at 0.3"], [
            ["Precision", "Of the customers we flag, how many really churn?", "46%"],
            ["Recall", "Of the customers who churn, how many do we flag?", "60%"],
            ["F1", "One number balancing both (harmonic mean)", "0.522"],
            ["ROC AUC", "How well are churners ranked above non-churners, over all thresholds? (0.5 = random, 1 = perfect)", "0.829"],
        ], caption="Table 26.1 — Scores for imbalanced yes/no problems (glossary <b>Precision</b>, <b>Recall</b>, <b>ROC AUC</b>)"),
        para("Accuracy barely moves across thresholds and never clearly beats the 82.8% of always answering \"no\", "
             "yet the model is useful: at 0.3 it finds six in ten churners, at 0.7 it is right 88% of the time it "
             "raises an alarm. Which threshold is best depends on costs: if a retention offer is cheap and losing a "
             "customer expensive, favour recall (a low threshold); if every flag triggers a costly phone call, favour "
             "precision (a high threshold)."),
        h2("26.3 Serving probabilities"),
        snippet(SERVE, caption="Program.cs, part 3: save, load, score"),
        output("""
            new customer A (3 months, $95, 5 calls, monthly): churn risk 88 %
            new customer B (60 months, $40, 0 calls, 2-year):  churn risk 0 %
            """),
        cpugpu("running the project",
               """
               dotnet run -c Release
               """,
               """
               dotnet run -c Release -- --cuda
               // at 1,281 parameters the CPU is faster; the GPU pays off for wide models
               // or hundreds of thousands of rows (Chapter 20)
               """.replace("Chapter 20", ch("performance"))),
        h2("26.4 Variations"),
        reftable(["Situation", "Do this"], [
            ["Very rare positives (fraud: 0.5%)", "Oversample positives in the training set with <code>Subset</code> (repeat their indices), or weight the loss; judge by recall at a fixed precision"],
            ["Several independent yes/no labels (multi-label)", "<code>Linear(h, K)</code> with <code>BinaryCrossEntropyWithLogits</code>; each output has its own sigmoid and threshold"],
            ["Probabilities must be well calibrated", "Keep the logits loss; avoid heavy oversampling (it inflates probabilities) or rescale afterwards"],
            ["Categorical features (plan type, region)", "Embeddings (" + ch("recommender") + ")"],
            ["Sequences (click streams)", "An LSTM/GRU front end (" + ch("recurrent") + ", " + ch("sentiment") + ")"],
        ], caption="Table 26.2 — Adapting the churn model"),
        trap("tuning the threshold on the test set",
             "<p>Choosing the threshold that maximizes F1 on the test set and then reporting that F1 is optimistic. Pick the "
             "threshold on a validation split and report on a separate test split.</p>"),
        practice([
            (1, "Why does the model end with <code>Linear(32, 1)</code> and no <code>Sigmoid</code>?",
             "<code>BinaryCrossEntropyWithLogits</code> applies the sigmoid internally in a numerically stable way (" + ch("losses") + "). "
             "Apply <code>Sigmoid()</code> only when reading probabilities."),
            (1, "Why does <code>Metric.BinaryAccuracy</code> use threshold 0 here?",
             "The model outputs logits; a logit of 0 corresponds to a probability of 0.5."),
            (2, "Oversample churners so they make up about 40% of the training set, retrain, and compare recall at 0.5.",
             "Collect the indices of positive rows, repeat them three times, append to all indices, and train on "
             "<code>train.Subset(indices)</code>. Recall at 0.5 rises sharply, precision falls, and probabilities shift "
             "upwards; AUC changes little."),
            (2, "Write a function that returns the threshold with the best F1 on a validation set.",
             "Try thresholds 0.05, 0.10, …, 0.95, compute precision and recall at each as in part 2, and return the "
             "threshold with the largest F1."),
            (3, "Turn the project into a fraud detector for a CSV of transactions with a 0/1 <code>fraud</code> column.",
             "<code>LoadCsv</code> with <code>TargetColumns = [\"fraud\"]</code>; scale features; the same model and loss; "
             "oversample fraud rows; choose the threshold for the recall the business needs; report precision, recall and AUC."),
        ], PART),
        footer("Binary classification", "Logits", "Sigmoid", "Threshold", "Precision", "Recall", "F1 score",
               "ROC AUC", "Class imbalance", "Oversampling"),
    )

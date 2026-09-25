"""Chapter 17 — Data: Datasets, CSV, Scalers and Loaders."""
from gen import *

PART = "III"

CSV_LINES = """
    id,area,rooms,age,price
    1,120,3,15,310000
    2,85,2,30,205000
    3,60,1,40,150000
    4,200,5,5,480000
    5,95,3,22,240000
    6,150,4,10,370000
    7,70,2,35,170000
    8,110,3,18,285000
    9,130,4,12,330000
    10,55,1,45,135000
"""


CSV_HEAD = ("\n            const string csv = \"\"\"\n"
            + "".join("                " + line.strip() + "\n" for line in CSV_LINES.strip().splitlines())
            + "                \"\"\";\n")


def pipeline_svg():
    w, h = 470, 70
    steps = ["CSV / arrays", "Dataset", "Split", "Scale", "DataLoader", "Batch tensors"]
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">']
    bw = 68
    for i, s in enumerate(steps):
        x = 6 + i * 77
        p.append(f'<rect x="{x}" y="18" width="{bw}" height="30" rx="5" fill="{"#0f6b5c" if i == 5 else "#e6f2ef"}" stroke="#0f6b5c"/>')
        p.append(svg_text(x + bw / 2, 37, s, 7.6, "#ffffff" if i == 5 else "#0f6b5c"))
        if i < 5:
            p.append(f'<line x1="{x + bw}" y1="33" x2="{x + 77}" y2="33" stroke="#56606a"/>')
            p.append(f'<polygon points="{x + 77},33 {x + 71},30 {x + 71},36" fill="#56606a"/>')
    p.append(svg_text(235, 64, "host memory (float[]) until the loader creates tensors on the chosen device", 7.4, "#56606a"))
    p.append("</svg>")
    return "".join(p)


def build():
    return page(
        chapter_open(
            "data",
            "Models learn from numbers in tensors, but data arrives as CSV files, arrays, images and text. The "
            "<code>NeuralSharp.Data</code> namespace covers the path in between: an in-memory <code>Dataset</code>, a "
            "parallel CSV reader, train/test splitting, scalers that bring every column to a similar range, and a "
            "<code>DataLoader</code> that cuts shuffled mini-batches and places them on the device.",
            "<code>Dataset</code>: features <code>[Count, FeatureCount]</code> and targets <code>[Count, TargetCount]</code> in host memory; immutable.",
            "Create from CSV (<code>LoadCsv</code>/<code>ParseCsv</code>), arrays (<code>FromArrays</code>, <code>FromFlat</code>) or class labels (<code>FromClassLabels</code>).",
            "<code>Split(0.8, seed)</code> for train/test; fit scalers on the <b>training</b> part only, then <code>Scale</code> both parts.",
            "<code>DataLoader(dataset, batchSize, shuffle, dropLast, device, seed)</code> yields <code>Batch</code>es already on the device.",
            "<code>WithFeatureShape(1, 28, 28)</code> makes batches come out as images or sequences instead of flat rows.",
        ),
        diagram("Figure 17.1 — The data pipeline", pipeline_svg(),
                "Everything before the loader is plain .NET arrays; only batches become tensors."),
        h2("17.1 Datasets"),
        reftable(["Create with", "From"], [
            ["<code>Dataset.LoadCsv(path, CsvOptions)</code>", "A numeric CSV file (parsed in parallel)"],
            ["<code>Dataset.ParseCsv(text, CsvOptions)</code>", "CSV text in memory"],
            ["<code>Dataset.FromArrays(float[,] features, float[,] targets, names?)</code>", "Two rectangular arrays"],
            ["<code>Dataset.FromFlat(float[] f, float[] t, count, featureNames, targetNames)</code>", "Row-major arrays (used without copying)"],
            ["<code>Dataset.FromClassLabels(float[,] features, int[] labels, classes, classNames?)</code>", "Features + class ids → one-hot targets"],
        ], caption="Table 17.1 — Creating a Dataset (NeuralSharp.Data)"),
        reftable(["Member", "Purpose"], [
            ["<code>Count</code>, <code>FeatureCount</code>, <code>TargetCount</code>", "Sizes"],
            ["<code>FeatureNames</code>, <code>TargetNames</code>", "Column names (from the CSV header, or x0, x1, … / y0, …)"],
            ["<code>GetFeatures(i)</code>, <code>GetTargets(i)</code>", "One sample's values as spans"],
            ["<code>Features</code>, <code>Targets</code>", "All values, row-major spans"],
            ["<code>Split(fraction, seed)</code>", "Shuffled train/test split"],
            ["<code>Subset(indices)</code>", "Selected rows (for folds, balancing, custom splits)"],
            ["<code>Scale(featureScaler, targetScaler)</code>", "Scaled copy"],
            ["<code>ToOneHot(classes)</code>", "One class-id target column → one-hot targets"],
            ["<code>WithFeatureShape(params int[] shape)</code>", "Per-sample shape for batches, e.g. <code>[1, 28, 28]</code> or <code>[T]</code>"],
            ["<code>FeaturesToArray()</code>, <code>TargetsToArray()</code>", "Back to <code>float[,]</code>"],
        ], caption="Table 17.2 — The Dataset API"),
        h2("17.2 CSV files"),
        reftable(["CsvOptions property", "Default", "Meaning"], [
            ["<code>TargetColumns</code> (required)", "—", "Names (or 0-based indices as text) of the columns to predict"],
            ["<code>IgnoreColumns</code>", "none", "Columns to skip, such as ids"],
            ["<code>Delimiter</code>", "<code>','</code>", "Field separator (<code>';'</code>, <code>'\\t'</code>, …)"],
            ["<code>HasHeader</code>", "<code>true</code>", "First line holds names; without it, columns are named 0, 1, …"],
            ["<code>Culture</code>", "invariant", "Number format (e.g. <code>new CultureInfo(\"de-DE\")</code> for 1,5)"],
        ], caption="Table 17.3 — CsvOptions"),
        mex("from CSV text to scaled training batches",
            "Ten houses; predict the price from area, rooms and age. The same code reads a file with "
            "<code>LoadCsv(path, options)</code>.",
            CSV_HEAD + """
            var data = Dataset.ParseCsv(csv, new CsvOptions { TargetColumns = ["price"], IgnoreColumns = ["id"] });
            Console.WriteLine(data);

            var (train, test) = data.Split(0.8, seed: 42);
            Console.WriteLine($"train {train.Count}, test {test.Count}");

            var fx = StandardScaler.FitFeatures(train);            // statistics from TRAINING data only
            var fy = StandardScaler.FitTargets(train);
            Console.WriteLine($"feature means {string.Join(", ", fx.Mean.Select(v => v.ToString("F1")))}  " +
                              $"stds {string.Join(", ", fx.Std.Select(v => v.ToString("F1")))}");

            var trainScaled = train.Scale(fx, fy);
            var testScaled = test.Scale(fx, fy);
            Console.WriteLine($"first scaled row: {string.Join(", ", trainScaled.GetFeatures(0).ToArray().Select(v => v.ToString("F3")))}" +
                              $" -> {trainScaled.GetTargets(0)[0]:F3}");

            float[] predicted = [0.5f, -1.2f];                      // model outputs, scaled
            fy.InverseTransform(predicted, 1);
            Console.WriteLine($"unscaled predictions: {string.Join(", ", predicted.Select(v => v.ToString("N0")))}");
            """,
            out="""
            Dataset(10 samples, features [area, rooms, age] -> targets [price])
            train 8, test 2
            feature means 113.8, 3.0, 21.1  stds 43.4, 1.2, 11.9
            first scaled row: -1.009, -0.816, 1.166 -> -1.069
            unscaled predictions: 334,225, 156,235
            """),
        reftable(["Problem in the file", "Message"], [
            ["A value that is not a number", "<i>text line 2, column 'b': 'x' is not a number.</i>"],
            ["A misspelled target column", "<i>Column 'price' not found in text. Columns: a, b.</i>"],
            ["A row with too few or too many fields", "<i>… line N: expected 5 fields, found 4.</i>"],
        ], caption="Table 17.4 — CSV errors name the line and column"),
        honestbox("What the CSV reader does not do",
                  "<p>It reads numeric columns only: no text categories, no dates, no missing values, and quoted fields "
                  "may not contain the delimiter. Convert such columns first (a small preprocessing step with "
                  "<code>File.ReadLines</code> and <code>string.Split</code>, or a CSV library), then build the dataset "
                  "with <code>FromArrays</code>. " + ch("recommender") + " shows categorical columns turned into ids "
                  "for embeddings.</p>"),
        h2("17.3 Scaling"),
        para("Networks train best when every input column has a similar range, typically mean 0 and standard "
             "deviation 1 (glossary <b>Standardization</b>). Regression targets benefit too: the loss is then on a "
             "comparable scale whatever the unit, and learning rates transfer between problems."),
        reftable(["Scaler", "Fit with", "Transform"], [
            ["<code>StandardScaler</code>", "<code>FitFeatures(ds)</code>, <code>FitTargets(ds)</code>, <code>Fit(span, columns)</code>", "(x − mean) / std per column; constant columns get std 1"],
            ["<code>MinMaxScaler</code>", "<code>Fit(span, columns)</code>", "(x − min) / (max − min) into [0, 1]"],
            ["both", "—", "<code>Transform(span, columns)</code>, <code>InverseTransform(span, columns)</code>, in place"],
        ], caption="Table 17.5 — Scalers (IScaler)"),
        snippet("""
            fx.Save("features.scaler");                        // one "mean std" line per column
            fy.Save("price.scaler");

            // in the program that serves predictions:
            var fx = StandardScaler.Load("features.scaler");
            var fy = StandardScaler.Load("price.scaler");
            float[] row = [120, 3, 15];
            fx.Transform(row, 3);                              // scale the input like the training data
            float[,] output = model.Predict(new float[,] { { row[0], row[1], row[2] } });
            float[] price = [output[0, 0]];
            fy.InverseTransform(price, 1);                     // back to currency
            """, caption="Saving scalers with the model, and using them at inference"),
        trap("fitting the scaler on all the data",
             "<p>Fitting on train + test lets information about the test set leak into training, and the test score "
             "becomes optimistic. Split first, fit on the training part, then scale both parts with the same scaler.</p>"),
        trap("forgetting to save the scaler",
             "<p>A model trained on scaled inputs gives nonsense on raw inputs. Treat the scaler files as part of the "
             "model: save them next to the weights and load them together (" + ch("inference") + ").</p>"),
        h2("17.4 The DataLoader"),
        reftable(["Argument", "Default", "Meaning"], [
            ["<code>dataset</code>", "—", "The samples"],
            ["<code>batchSize</code>", "32", "Samples per batch"],
            ["<code>shuffle</code>", "false", "New random order every epoch (turn on for training)"],
            ["<code>dropLast</code>", "false", "Skip the final smaller batch"],
            ["<code>device</code>", "<code>Device.Default</code>", "Where batch tensors are created"],
            ["<code>seed</code>", "random", "Shuffle seed for reproducible runs"],
        ], caption="Table 17.6 — new DataLoader(dataset, batchSize, shuffle, dropLast, device, seed)"),
        mex("batches of the scaled house data", None,
            """
            var loader = new DataLoader(trainScaled, batchSize: 3, shuffle: true, seed: 1);
            Console.WriteLine($"{loader.BatchCount} batches per epoch");
            foreach (var batch in loader)
            {
                using (batch)                                   // frees the batch tensors
                    Console.WriteLine($"  batch {batch.Index}: features [{string.Join(", ", batch.Features.Shape.ToArray())}]" +
                                      $" targets [{string.Join(", ", batch.Targets.Shape.ToArray())}]");
            }
            Console.WriteLine($"dropLast: {new DataLoader(trainScaled, 3, dropLast: true).BatchCount} batches");
            """,
            out="""
            3 batches per epoch
              batch 0: features [3, 3] targets [3, 1]
              batch 1: features [3, 3] targets [3, 1]
              batch 2: features [2, 3] targets [2, 1]
            dropLast: 2 batches
            """),
        cpugpu("where batches live",
               """
               var loader = new DataLoader(train, batchSize: 64, shuffle: true, device: Device.Cpu);
               """,
               """
               var loader = new DataLoader(train, batchSize: 256, shuffle: true, device: Device.Cuda());
               // each batch is gathered into a host buffer and uploaded once; for large batches the
               // NEXT batch is gathered on a worker thread while the GPU trains on the current one
               """,
               "The loader's device must be the model's device. Larger batches use the GPU better; see " + ch("performance") + "."),
        h2("17.5 Classes, images and sequences"),
        mex("class labels, one-hot targets and image-shaped batches", None,
            """
            var classes = Dataset.FromClassLabels(
                new float[,] { { 0.1f, 0.2f }, { 0.9f, 0.8f }, { 0.5f, 0.1f } },
                labels: [0, 2, 1], classes: 3, classNames: ["low", "high", "mid"]);
            Console.WriteLine(classes);
            Console.WriteLine(string.Join(", ", classes.GetTargets(1).ToArray()));      // label 2 as one-hot

            var fromCsv = Dataset.ParseCsv("a,b,label\\n1,2,0\\n3,4,2\\n",
                                           new CsvOptions { TargetColumns = ["label"] }).ToOneHot(3);
            Console.WriteLine(fromCsv);

            var images = Dataset.FromArrays(new float[2, 784], new float[2, 1]).WithFeatureShape(1, 28, 28);
            foreach (var b in new DataLoader(images, 2))
                using (b) Console.WriteLine($"image batch [{string.Join(", ", b.Features.Shape.ToArray())}]");
            """,
            out="""
            Dataset(3 samples, features [x0, x1] -> targets [low, high, mid])
            0, 0, 1
            Dataset(2 samples, features [a, b] -> targets [class0, class1, class2])
            image batch [2, 1, 28, 28]
            """),
        reftable(["Data", "Features per sample", "Targets", "Loss"], [
            ["Table, regression", "<code>[F]</code>", "<code>[K]</code> numbers (scaled)", "MSE / MAE"],
            ["Table, classes (one-hot)", "<code>[F]</code>", "<code>[K]</code> one-hot (<code>FromClassLabels</code>, <code>ToOneHot</code>)", "<code>CrossEntropy</code>"],
            ["Table, classes (ids)", "<code>[F]</code>", "<code>[1]</code> id; reshape to <code>[N]</code> in the loss", "<code>SparseCrossEntropy</code>"],
            ["Images", "<code>WithFeatureShape(C, H, W)</code>", "one-hot or id", "cross-entropy"],
            ["Token sequences", "<code>WithFeatureShape(T)</code> (ids as floats)", "id or ids per position <code>[T]</code>", "<code>SparseCrossEntropy</code>"],
            ["Numeric sequences", "<code>WithFeatureShape(T, F)</code>", "next value(s)", "MSE"],
        ], caption="Table 17.7 — Shaping common data"),
        trap("id targets with SparseCrossEntropy through the Trainer",
             "<p>Loader targets always have shape <code>[N, TargetCount]</code>, so class ids arrive as "
             "<code>[N, 1]</code>. <code>SparseCrossEntropy</code> reshapes the ids to the logits' leading shape "
             "itself, so <code>[N, 1]</code> ids with <code>[N, K]</code> logits work directly, as do "
             "<code>[N, T]</code> ids with <code>[N, T, V]</code> logits.</p>"),
        practice([
            (1, "Load <code>sales.csv</code> (semicolon-separated, German number format), predicting <code>revenue</code> and ignoring <code>date</code>.",
             "<code>Dataset.LoadCsv(\"sales.csv\", new CsvOptions { TargetColumns = [\"revenue\"], IgnoreColumns = [\"date\"], "
             "Delimiter = ';', Culture = new CultureInfo(\"de-DE\") })</code>. The date column is ignored, so it need not be numeric."),
            (1, "How many batches does a loader over 1,000 samples with batch size 64 produce, with and without <code>dropLast</code>?",
             "16 without (15 full + one of 40), 15 with."),
            (2, "Make a 5-fold cross-validation split with <code>Subset</code>.",
             "Shuffle <code>Enumerable.Range(0, Count)</code> with a seeded <code>Random</code>, cut it into 5 parts, and for "
             "fold k use <code>Subset(part k)</code> as validation and <code>Subset(all other parts)</code> as training; fit "
             "the scaler on each fold's training part."),
            (2, "Turn 28×28 grey images stored as <code>byte[][]</code> with labels <code>int[]</code> into a dataset whose batches are <code>[N, 1, 28, 28]</code>.",
             "Fill a <code>float[N, 784]</code> with <code>pixel / 255f</code>, call <code>Dataset.FromClassLabels(features, labels, 10)</code> "
             "and then <code>.WithFeatureShape(1, 28, 28)</code>."),
            (3, "Write a sliding-window helper that turns a series <code>float[] s</code> into a dataset of windows of length T predicting the next value, with batches <code>[N, T, 1]</code>.",
             "For i from 0 to s.Length − T − 1, copy s[i..i+T] into row i of a <code>float[N, T]</code> and s[i+T] into the "
             "target; create with <code>FromArrays</code> and <code>WithFeatureShape(T, 1)</code>. " + ch("timeseries") +
             " uses exactly this."),
        ], PART),
        footer("Dataset", "CSV", "Train/test split", "Standardization", "StandardScaler", "MinMaxScaler",
               "Data leakage", "DataLoader", "Mini-batch", "Batch size", "Shuffle", "One-hot", "Cross-validation"),
    )

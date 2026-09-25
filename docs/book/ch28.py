"""Chapter 28 — Multi-Class Classification."""
from gen import *

PART = "V"


def build():
    return page(
        chapter_open(
            "multiclass",
            "When the answer is one of several categories (which product line, which species, which digit), the model "
            "outputs one score per class and the loss is cross-entropy. This project classifies points of three "
            "interleaved spirals, a problem no straight line can solve, and shows the full workflow including "
            "one-hot targets, label smoothing, a cosine schedule, a confusion matrix, class probabilities at "
            "inference and a decision map. The code is the repository's <code>NeuralSharp.Samples.Classification</code>.",
            "Task type: <b>multi-class classification</b>. Last layer <code>Linear(h, K)</code>; loss <code>CrossEntropy</code> (one-hot) or <code>SparseCrossEntropy</code> (ids).",
            "Targets: <code>Dataset.FromClassLabels(features, labels, K, names)</code> creates one-hot rows.",
            "Predictions: <code>ArgMax</code> for the class, <code>Softmax()</code> for probabilities.",
            "Result: 97.8% test accuracy on 3 spiral classes; the confusion matrix shows which classes are confused.",
            "The same code classifies any table of numbers into K classes.",
        ),
        h2("28.1 Data and model"),
        snippet("""
            const int Classes = 3, PerClass = 300;
            string[] classNames = ["red", "green", "blue"];

            var random = new Random(1);
            var features = new float[Classes * PerClass, 2];
            var labels = new int[Classes * PerClass];
            for (int c = 0; c < Classes; c++)
                for (int i = 0; i < PerClass; i++)
                {
                    int row = c * PerClass + i;
                    double radius = i / (double)PerClass;
                    double angle = c * 2 * Math.PI / Classes + radius * 5 + random.NextDouble() * 0.25;
                    features[row, 0] = (float)(radius * Math.Cos(angle));
                    features[row, 1] = (float)(radius * Math.Sin(angle));
                    labels[row] = c;
                }
            var data = Dataset.FromClassLabels(features, labels, Classes, classNames);   // one-hot targets
            var (train, test) = data.Split(0.8, seed: 2);

            var init = new Random(3);
            using var model = new Sequential
            {
                new Linear(2, 128, device: device, random: init), new BatchNorm(128, device: device), new ReLU(),
                new Linear(128, 64, device: device, random: init), new BatchNorm(64, device: device), new ReLU(),
                new Linear(64, Classes, device: device, random: init),                // raw scores
            };
            """, caption="Spiral data and a 9,219-parameter classifier"),
        h2("28.2 Training"),
        snippet("""
            int epochs = 200;
            using var optimizer = new AdamW(model.Parameters(), learningRate: 0.01f, weightDecay: 1e-4f);
            var trainer = new Trainer(model, optimizer,
                (logits, targets) => Losses.CrossEntropy(logits, targets, labelSmoothing: 0.05f))
            {
                Metrics = { Metric.Accuracy },
                Scheduler = new CosineAnnealing(optimizer, totalEpochs: epochs, warmupEpochs: 5),
            };
            trainer.Fit(new DataLoader(train, 64, shuffle: true, device: device, seed: 4), epochs,
                        validation: new DataLoader(test, 512, device: device));
            """, caption="Cross-entropy with label smoothing, AdamW and a cosine schedule"),
        output("""
            Training on cpu (CPU (4 threads, 8-wide SIMD)) | 720 samples, 180 validation | batch 64, 12 steps/epoch | AdamW lr=0.001667 | 9,219 parameters | 4 CPU threads
            Epoch   1/200  loss 0.886633  accuracy 0.5694  val_loss 1.077530  val_accuracy 0.4667  86.3 ms  8,344 samples/s  *
            Epoch  20/200  loss 0.317225  accuracy 0.9347  val_loss 0.247562  val_accuracy 0.9722  19.5 ms  36,992 samples/s  *
            Epoch  60/200  loss 0.244288  accuracy 0.9750  val_loss 0.224612  val_accuracy 0.9833  16.2 ms  44,349 samples/s
            Epoch 100/200  loss 0.239780  accuracy 0.9722  val_loss 0.216285  val_accuracy 0.9944  5.0 ms  144,401 samples/s
            Epoch 200/200  loss 0.229764  accuracy 0.9778  val_loss 0.218855  val_accuracy 0.9778  9.0 ms  80,380 samples/s
            Finished 200 epochs in 1.87 s | best epoch 182 loss 0.209481
            """, caption="Training output (every 20th epoch printed; some lines omitted here)"),
        para("The loss levels off near 0.2 rather than 0: with label smoothing the target itself is not a pure one-hot "
             "row, so a perfect model still has a positive loss (" + ch("losses") + "). The first epoch starts close to "
             "ln 3 ≈ 1.10, as expected for 3 classes."),
        h2("28.3 Where the errors are"),
        snippet("""
            var result = trainer.Evaluate(new DataLoader(test, 512, device: device));
            Console.WriteLine($"Test accuracy: {result.Metrics["accuracy"]:P1}  (cross-entropy {result.Loss:F4})");

            var scores = trainer.Predict(test);                        // [180, 3] raw scores
            var confusion = new int[Classes, Classes];
            for (int i = 0; i < test.Count; i++)
            {
                int actual = test.GetTargets(i).IndexOf(1f);           // position of the 1 in the one-hot row
                int predicted = Enumerable.Range(0, Classes).MaxBy(c => scores[i, c]);
                confusion[actual, predicted]++;
            }
            """, caption="Accuracy and a confusion matrix"),
        output("""
            Test accuracy: 97.8 %  (cross-entropy 0.2189)

            Confusion matrix (rows = actual, columns = predicted):
                        red  green   blue
              red        67      0      0
              green       1     48      0
              blue        3      0     61
            """),
        para("Of 180 test points 4 are wrong, and 3 of those are blue points taken for red, which is where the two "
             "spirals meet near the centre. A confusion matrix (glossary <b>Confusion matrix</b>) shows such patterns "
             "that a single accuracy figure hides."),
        h2("28.4 Probabilities at inference"),
        snippet("""
            model.Load(modelPath);
            using var points = Tensor.From(new float[,] { { 0.5f, 0.2f }, { -0.3f, -0.6f }, { 0f, 0.9f } }, device);
            using var logits = model.Predict(points);
            using var probabilities = logits.Softmax();                // rows sum to 1
            """, caption="--predict mode"),
        output("""
                 x      y  | class  |    red   green    blue
               0.50   0.20 | blue   |    1 %     3 %    96 %
              -0.30  -0.60 | red    |   79 %    19 %     2 %
               0.00   0.90 | blue   |    0 %     4 %    95 %
            """),
        output("""
            BBBBBBBBBBBBBBBBbbbbbbbbbbbbbbbbbbbbgggggGGGGGGGGGGGGGGGGGGG
            BBBBBBBBBBBBBbbbbbrrrrrrrrrrrrrbbbbBBBBBBBbbbggggGGGGGGGGGGG
            BBBBBBBbbbbrrrRRRRRRRRrrrrrrRRRRRRRrrbbBBBBBBbbgggGGGGGGGGGG
            BBBBBbbbrrrRRRRRRrrgggGGGGGgggrrRRrrbbBBBBBBbbgggGGGGGGGGGGG
            BBbbbrrrRRRRRRRRrrggGGGGGggbbBBBBBBBBBBBBBbbgggGGGGGGGGGGGGG
            bbbrrrrRRRRRRRRRRrrrggGGGGGGGggggggggggGGGGGGGGGGGGGGGGGGggg
            bbbrrrrrRRRRRRRRRRRRRRrrrrrrgggggggggggggggggggggggggggrrrrr
            bbrrrrrrrrRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR
            """, caption="Part of the decision map the sample prints (upper case: at least 90% confident)"),
        cpugpu("running the sample",
               """
               dotnet run -c Release --project samples/NeuralSharp.Samples.Classification -- --cpu
               dotnet run -c Release --project samples/NeuralSharp.Samples.Classification -- --cpu --predict --input "0.5,0.2;-0.3,-0.6"
               """,
               """
               dotnet run -c Release --project samples/NeuralSharp.Samples.Classification -- --cuda
               """),
        h2("28.5 From spirals to your data"),
        reftable(["Your data", "Change"], [
            ["CSV with a class-id column", "<code>Dataset.LoadCsv(…, new CsvOptions { TargetColumns = [\"species\"] }).ToOneHot(K)</code>, scale the features"],
            ["Many classes (100+)", "Wider last hidden layer; <code>SparseCrossEntropy</code> with id targets saves memory"],
            ["Unequal class sizes", "Report per-class accuracy or the confusion matrix; oversample small classes"],
            ["Images", "A CNN front end (" + ch("cnn") + ")"],
            ["Text", "Embeddings + LSTM/transformer (" + ch("sentiment") + ")"],
            ["\"None of the above\" matters", "Add an explicit \"other\" class, or reject predictions whose top probability is below a threshold"],
        ], caption="Table 28.1 — Adapting the classifier"),
        trap("reading scores as probabilities",
             "<p><code>Predict</code> returns raw scores; they can be negative and do not sum to 1. Apply "
             "<code>Softmax()</code> for probabilities; the class (<code>ArgMax</code>) is the same either way.</p>"),
        practice([
            (1, "What is the expected loss of an untrained 10-class classifier?",
             "About ln 10 ≈ 2.30 (slightly more with label smoothing)."),
            (1, "Train with ids instead of one-hot targets: what changes?",
             "Build the dataset with <code>FromArrays(features, ids)</code> (one column of class ids), use "
             "<code>Losses.SparseCrossEntropy</code> and <code>Metric.SparseAccuracy</code>."),
            (2, "Compute per-class recall from the confusion matrix above.",
             "red 67/67 = 100%, green 48/49 = 98%, blue 61/64 = 95%."),
            (2, "Reject uncertain predictions: output \"unsure\" when the top probability is below 0.6. How many test points would be rejected?",
             "Apply <code>Softmax</code> to <code>trainer.Predict(test)</code> scores (row by row on the host) and count rows "
             "whose maximum is below 0.6; those are near the class boundaries, where most errors are."),
            (3, "Classify the classic Iris flowers (4 measurements, 3 species) from a CSV.",
             "<code>LoadCsv</code> with the species id column as target, <code>ToOneHot(3)</code>, split 80/20, standardize, "
             "a 4 → 16 → 3 MLP, <code>CrossEntropy</code>, <code>Metric.Accuracy</code>, early stopping; expect over 90% test accuracy."),
        ], PART),
        footer("Multi-class classification", "Cross-entropy", "One-hot", "Softmax", "ArgMax", "Confusion matrix",
               "Label smoothing", "Decision boundary"),
    )

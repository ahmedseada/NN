"""Chapter 38 — Inference Modes and Model Files in Practice."""
from gen import *

PART = "VII"


def build():
    return page(
        chapter_open(
            "inference",
            "Training happens once; inference happens every time the model is used. This chapter collects what every "
            "project in Parts V and VI did to use a trained model: which files make up a model, how the samples switch "
            "between training and prediction with <code>--mode predict</code>, how to load on either device, and what "
            "inference really costs: a cold first call, then microseconds per row, especially in batches.",
            "A deployable model is a <b>package</b>: weights, plus whatever turns raw input into tensors and outputs into answers (scalers, vocabularies, class names, configuration).",
            "Build the architecture from one factory (or from a saved configuration), then <code>Load</code>, then <code>Eval()</code> once.",
            "<code>Predict</code> = evaluation mode + no gradients + freed intermediates.",
            "Measured: 13.6 ms cold first call, 20.6 µs per single-row call, 1.57 µs per row in batches of 1,000.",
            "Every sample project has an inference mode: <code>--predict</code> (or <code>--mode predict</code>) with <code>--model</code> and <code>--input</code>.",
        ),
        h2("38.1 What a model package contains"),
        reftable(["Project", "Files", "Why each is needed"], [
            ["House prices (" + ch("regression", None) + ")", "<code>house-price.weights</code>, <code>.features.txt</code>, <code>.price.txt</code>", "Weights; input scaling; output unscaling"],
            ["Churn (" + ch("binary", None) + ")", "<code>churn.weights</code>, <code>churn.scaler</code> (+ chosen threshold)", "The threshold is a business decision made after training"],
            ["Anomaly detector (" + ch("anomaly", None) + ")", "weights, scaler, threshold", "The detector is model + threshold"],
            ["Recommender (" + ch("recommender", None) + ")", "weights, user and item id maps", "Ids must map to the same embedding rows"],
            ["Sentiment (" + ch("sentiment", None) + ")", "weights, vocabulary", "Words must map to the same ids"],
            ["OCR (" + ch("ocr", None) + ")", "weights, alphabet", "Class index → character"],
            ["GPT (" + ch("gpt", None) + ")", "weights, config JSON (vocabulary, sizes, training record)", "The config rebuilds the architecture"],
        ], caption="Table 38.1 — Model packages from this book"),
        snippet("""
            // A small manifest saved next to the weights; loading reads it first.
            public sealed record ModelManifest(
                string Name, int Version, string Architecture,      // e.g. "mlp-9-64-32-1"
                string[] FeatureNames, string? TargetName,
                string[]? ClassNames, float? Threshold,
                DateTimeOffset TrainedAt, double ValidationScore)
            {
                public void Save(string path) => File.WriteAllText(path, JsonSerializer.Serialize(this));
                public static ModelManifest Load(string path) => JsonSerializer.Deserialize<ModelManifest>(File.ReadAllText(path))!;
            }

            // models/house-price/v3/ : model.weights, features.scaler, price.scaler, manifest.json
            """, caption="A pattern for versioned model folders"),
        trap("weights without their preprocessing",
             "<p>Weights alone are not a model. A missing scaler, vocabulary or class list produces confident nonsense rather "
             "than an error. Save everything the prediction path needs, in one folder, and load it together.</p>"),
        h2("38.2 Inference modes in the samples"),
        para("Every sample project trains, saves and demonstrates the model by default, and runs inference only with "
             "<code>--predict</code>. The shared <code>SampleOptions</code> class (samples/Shared) parses the options, "
             "so each <code>Program.cs</code> starts with the same few lines; reuse the pattern in your own tools."),
        reftable(["Option", "Meaning"], [
            ["<code>--mode train|predict</code>, <code>--predict</code>", "Train and save (default), or load and predict only"],
            ["<code>--model &lt;path&gt;</code>", "Model file to save or load (default <code>models/&lt;sample&gt;.weights</code> next to the executable)"],
            ["<code>--input &lt;value&gt;</code>", "Sample-specific input: features, sentences, shapes, a text line, a prompt"],
            ["<code>--data &lt;file.csv&gt;</code>", "Predict every row of a file (house prices)"],
            ["<code>--cpu</code>, <code>--cuda</code>, <code>--device cuda:N</code>", "Where inference runs"],
            ["<code>--threads</code>, <code>--gpu-memory</code>, <code>--cpu-memory</code>", "Resource limits (" + ch("devices") + ")"],
            ["<code>--log inference</code>, <code>--log-file run.jsonl</code>", "Latency telemetry (" + ch("telemetry") + ")"],
        ], caption="Table 38.2 — Sample command-line options"),
        snippet("""
            if (SampleOptions.Parse(args) is not { } options) return 0;      // prints help on --help or errors
            var device = options.Device;
            string modelPath = options.ModelPath("house-price.weights");

            if (options.PredictOnly)
            {
                if (!options.RequireModel(modelPath)) return 1;               // helpful message if not trained yet
                using var model = BuildModel(FeatureCount);
                model.Load(modelPath);
                // ... parse options.Input, predict, print ...
                return 0;
            }
            // ... otherwise train, evaluate, save ...
            """, caption="The skeleton every sample follows"),
        cpugpu("the same model file on either device",
               """
               using var model = BuildModel(device: Device.Cpu);
               model.Load("models/house-price.weights");
               model.Eval();
               """,
               """
               using var model = BuildModel(device: Device.Cuda());
               model.Load("models/house-price.weights");              // the file does not know about devices
               model.Eval();
               """),
        h2("38.3 What inference costs"),
        mex("cold start, single rows and batches (house-price model, CPU)", None,
            """
            model.Eval();
            var one = new float[1, 9];
            var many = new float[1000, 9];

            var cold = Stopwatch.StartNew();
            model.Predict(one);
            Console.WriteLine($"first call (cold): {cold.Elapsed.TotalMilliseconds:F2} ms");

            for (int i = 0; i < 200; i++) model.Predict(one);               // warm-up
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < 1000; i++) model.Predict(one);
            Console.WriteLine($"1 row per call:     {sw.Elapsed.TotalMilliseconds:F1} µs per call");  // 1000 calls: ms = µs each

            for (int i = 0; i < 20; i++) model.Predict(many);
            sw.Restart();
            for (int i = 0; i < 100; i++) model.Predict(many);
            double ms = sw.Elapsed.TotalMilliseconds / 100;
            Console.WriteLine($"1,000 rows per call: {ms:F3} ms per call = {ms:F2} µs per row");
            """,
            out="""
            first call (cold): 13.60 ms
            1 row per call:     20.6 µs per call
            1,000 rows per call: 1.574 ms per call = 1.57 µs per row
            """,
            after="The cold call pays for JIT compilation and first allocations; do one warm-up prediction at startup. "
                  "Batching makes each row about 13× cheaper, because per-call costs are shared. On the GPU the difference "
                  "is larger still: a single-row call is dominated by launches and the copy back."),
        reftable(["Situation", "Recommendation"], [
            ["Interactive, one request at a time", "CPU; warm up at startup; <code>Predict(float[,])</code>"],
            ["Many requests per second", "Batch requests arriving within a few milliseconds into one <code>Predict</code> (micro-batching)"],
            ["Nightly scoring of millions of rows", "Large batches (10,000+); the GPU for big models; <code>Trainer.Predict(dataset)</code> batches for you"],
            ["Latency budget below 1 ms", "Small model on the CPU; avoid per-request allocations of large arrays"],
        ], caption="Table 38.3 — Inference setups"),
        h2("38.4 Concurrency"),
        para("A loaded model can serve many threads at once if it is put in evaluation mode once after loading: "
             "<code>Predict</code> then never changes the model's state, each call works on its own tensors, and memory "
             "pools are thread-safe. The house-price Web API of " + ch("webapi") + " served 200 concurrent requests this "
             "way. Two cases still need a lock (for example a <code>SemaphoreSlim</code>): moving the model between devices "
             "while serving, and stateful generation with a KV cache, where one generation owns the caches (the GPT API "
             "serializes requests for this reason)."),
        trap("calling Eval() per request",
             "<p><code>Predict</code> switches to evaluation mode and restores the previous mode afterwards. If the model was "
             "left in training mode, two concurrent calls can interleave these switches so one runs with dropout active. "
             "Call <code>Eval()</code> once after loading and leave it.</p>"),
        practice([
            (1, "List the files needed to serve the churn model and explain each.",
             "Weights (the network), the feature scaler (to scale inputs like training data), and the chosen threshold "
             "(to turn probabilities into decisions)."),
            (1, "Why warm up a model at startup?",
             "The first call includes one-off JIT compilation and allocation (13.6 ms here, far more for GPU kernel "
             "compilation); a warm-up moves that cost out of the first user's request."),
            (2, "Write a <code>ModelPackage.Load(folder)</code> that reads a manifest, builds the right architecture and loads weights and scaler.",
             "Read <code>manifest.json</code>; switch on <code>Architecture</code> to call the matching factory; <code>Load</code> the "
             "weights; <code>StandardScaler.Load</code> the scaler files; <code>Eval()</code>; return an object exposing <code>Predict</code>."),
            (2, "Score a CSV of 2 million rows as fast as possible.",
             "Stream the file in chunks of e.g. 50,000 rows, scale each chunk, call <code>Predict</code> on the chunk (or several "
             "chunks in parallel on the CPU), write results as you go; on a GPU use chunks of 100,000+."),
            (3, "Implement micro-batching for a web API: collect requests for up to 5 ms or 256 rows, predict once, and complete each request.",
             "Use a <code>Channel</code> of (input, <code>TaskCompletionSource</code>) pairs; a background loop reads the first item, "
             "keeps reading until 5 ms have passed or 256 rows are collected, stacks the rows, calls <code>Predict</code>, and sets each "
             "request's result from its row of the output."),
        ], PART),
        footer("Inference mode", "Model package", "Manifest", "Warm-up", "Latency", "Micro-batching", "Evaluation mode"),
    )

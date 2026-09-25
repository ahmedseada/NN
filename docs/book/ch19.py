"""Chapter 19 — Telemetry: Logging and Tracking Everything."""
from gen import *

PART = "III"


def build():
    return page(
        chapter_open(
            "telemetry",
            "NeuralSharp reports what it is doing through one hub, <code>Telemetry</code>: training progress, every "
            "batch, gradient sizes, every layer and every tensor operation with timings, and every inference call. "
            "Nothing is produced unless something subscribes, so an unobserved program pays essentially nothing. "
            "This chapter covers the levels, the four built-in hooks (console, in-memory recorder, JSON Lines file, "
            "channel) and how to write your own for dashboards, databases or experiment trackers.",
            "<code>using var sub = Telemetry.Subscribe(hook);</code>; disposing the subscription unsubscribes.",
            "Levels (flags): <code>Training</code>, <code>Batches</code>, <code>Gradients</code>, <code>Layers</code>, <code>Operations</code>, <code>Inference</code>, <code>All</code>.",
            "Built-in hooks: <code>ConsoleLogger</code>, <code>MetricsRecorder</code> (with <code>SaveCsv</code>), <code>JsonLinesLogger</code>, <code>ChannelTelemetry</code>.",
            "A custom hook implements <code>ITelemetryHook</code>: <code>Levels</code> plus only the event methods it needs.",
            "Batch-, gradient- and layer-level telemetry add GPU synchronizations; keep them for debugging and profiling.",
        ),
        h2("19.1 Levels and events"),
        reftable(["Level", "Event (record struct)", "Published", "Cost on the GPU"], [
            ["<code>Training</code>", "<code>TrainingStarted</code>, <code>EpochCompleted</code>, <code>TrainingCompleted</code>", "by <code>Trainer.Fit</code>", "none extra (one read per epoch)"],
            ["<code>Batches</code>", "<code>BatchCompleted</code>", "after each optimizer step", "one synchronization per batch"],
            ["<code>Gradients</code>", "<code>GradientNorm</code> on <code>BatchCompleted</code>", "after each backward pass", "extra kernels + one synchronization"],
            ["<code>Layers</code>", "<code>LayerForward</code>", "after each module's forward pass", "timings are launch times unless <code>SynchronizeForTiming</code>"],
            ["<code>Operations</code>", "<code>OperationCompleted</code>", "after each tensor operation and its gradient", "very verbose"],
            ["<code>Inference</code>", "<code>InferenceCompleted</code>", "after each <code>Module.Predict</code>", "one synchronization per call"],
        ], caption="Table 19.1 — Telemetry levels (NeuralSharp.Diagnostics)"),
        reftable(["Event", "Main fields"], [
            ["<code>TrainingStarted</code>", "Model summary, Optimizer, Device, Epochs, TrainingSamples, ValidationSamples, BatchSize, BatchesPerEpoch, ParameterCount, LearningRate, CpuThreads"],
            ["<code>EpochCompleted</code>", "Epoch, Loss, Metrics, ValidationLoss, ValidationMetrics, LearningRate, Duration, SamplesPerSecond, Memory, IsBest"],
            ["<code>BatchCompleted</code>", "Epoch, Batch, Step, BatchSize, Loss, LearningRate, GradientNorm, DataTime, ComputeTime"],
            ["<code>TrainingCompleted</code>", "EpochsRun, Duration, FinalLoss, BestEpoch, BestLoss, StoppedEarly, Cancelled"],
            ["<code>LayerForward</code>", "Layer, LayerType, Depth, InputShape, OutputShape, Duration, Device, Training"],
            ["<code>OperationCompleted</code>", "Operation, Backward, Shape, Elements, Device, Duration"],
            ["<code>InferenceCompleted</code>", "Model, Samples, InputShape, OutputShape, Device, Latency, SamplesPerSecond"],
        ], caption="Table 19.2 — Event fields"),
        h2("19.2 Watching the model work"),
        mex("layers and inference for one prediction", None,
            """
            var r = new Random(1);
            using var model = new Sequential { new Linear(2, 8, random: r), new Tanh(), new Linear(8, 1, random: r), new Sigmoid() };
            model.Name = "xor";

            using (Telemetry.Subscribe(new ConsoleLogger(TelemetryLevel.Layers | TelemetryLevel.Inference)))
            using (var x = Tensor.From(new float[,] { { 1, 0 } }))
            using (var y = model.Predict(x)) { }
            """,
            out="""
                Linear(2 -> 8): [1, 2] -> [1, 8]  7318.3 µs
                Tanh: [1, 8] -> [1, 8]  5919.5 µs
                Linear(8 -> 1): [1, 8] -> [1, 1]  15.7 µs
                Sigmoid: [1, 1] -> [1, 1]  722.4 µs
              xor: [1, 2] -> [1, 1]  20342.4 µs
            Inference xor: 1 samples on cpu in 22.862 ms (44 samples/s)
            """,
            after="These times include one-off costs of the very first call (the .NET JIT compiling each kernel). Measure "
                  "after a warm-up call to see steady-state latency, which for this model is a few microseconds."),
        mex("every operation of one training step, forward and backward", None,
            """
            using (Telemetry.Subscribe(new ConsoleLogger(TelemetryLevel.Operations)))
            using (var scope = new TensorScope())
            {
                var x = Tensor.From(new float[,] { { 1, 0 } });
                var loss = Losses.MeanSquaredError(model.Forward(x), Tensor.From(new float[,] { { 1 } }));
                loss.Backward();
            }
            """,
            out="""
                 matmul [1, 8] on cpu  294.5 µs
                 add_bias [1, 8] on cpu  5.9 µs
                 tanh [1, 8] on cpu  2.8 µs
                 matmul [1, 1] on cpu  5.1 µs
                 add_bias [1, 1] on cpu  4.7 µs
                 sigmoid [1, 1] on cpu  3.1 µs
                 sub [1, 1] on cpu  694.0 µs
                 square [1, 1] on cpu  685.3 µs
                 sum [] on cpu  690.4 µs
                ∇sum [] on cpu  837.3 µs
                ∇square [1, 1] on cpu  1015.1 µs
                ∇sub [1, 1] on cpu  254.8 µs
                ∇sigmoid [1, 1] on cpu  711.0 µs
                ∇add_bias [1, 1] on cpu  275.5 µs
                ∇matmul [1, 1] on cpu  2110.7 µs
                ∇tanh [1, 8] on cpu  1552.5 µs
                ∇add_bias [1, 8] on cpu  23.4 µs
                ∇matmul [1, 8] on cpu  1.9 µs
            """,
            after="The backward pass visits the operations in exactly the reverse order (∇ marks gradients). "
                  "<code>Mean</code> appears as <code>sum</code> because it is a scaled sum. This view is the quickest way to "
                  "see what a custom layer really computes."),
        trap("timing GPU layers and operations",
             "<p>On the GPU, layer and operation durations measure how long it took to <i>queue</i> the work. Set "
             "<code>Telemetry.SynchronizeForTiming = true</code> while profiling to wait for each result and get real "
             "execution times; it slows everything down, so switch it off afterwards.</p>"),
        h2("19.3 Recording, CSV and JSON Lines"),
        mex("a custom hook, a recorder and a JSON Lines file during one Fit", None,
            """
            sealed class StartStopHook : ITelemetryHook
            {
                public TelemetryLevel Levels => TelemetryLevel.Training;
                public void OnTrainingStarted(in TrainingStarted e) =>
                    Console.WriteLine($"started: {e.ParameterCount} parameters on {e.Device}, {e.Epochs} epochs");
                public void OnTrainingCompleted(in TrainingCompleted e) =>
                    Console.WriteLine($"done in {e.Duration.TotalMilliseconds:F0} ms, final loss {e.FinalLoss:F6}");
            }

            var recorder = new MetricsRecorder();
            using var opt = new Adam(model.Parameters(), 0.05f);
            var trainer = new Trainer(model, opt, Losses.MeanSquaredError) { Metrics = { Metric.BinaryAccuracy() } };

            using (Telemetry.Subscribe(new StartStopHook()))
            using (Telemetry.Subscribe(recorder))
            using (var json = new JsonLinesLogger("train.jsonl", TelemetryLevel.Training | TelemetryLevel.Batches))
            using (Telemetry.Subscribe(json))
            {
                trainer.Fit(new DataLoader(xor, 4), epochs: 300);        // xor: the 4-row XOR Dataset
            }

            Console.WriteLine($"recorded {recorder.Epochs.Count} epochs; last loss {recorder.Epochs[^1].Loss:F6}, " +
                              $"accuracy {recorder.Epochs[^1].Metrics["accuracy"]:F2}");
            recorder.SaveCsv("epochs.csv");
            foreach (var line in File.ReadLines("epochs.csv").Take(3)) Console.WriteLine(line);
            """,
            out="""
            started: 33 parameters on cpu, 300 epochs
            done in 30 ms, final loss 0.000130
            recorded 300 epochs; last loss 0.000130, accuracy 1.00
            epoch,loss,accuracy,val_loss,learning_rate,duration_ms,samples_per_second,memory_in_use_bytes
            1,0.26254576444625854,0.5,,0.05,11.432,349.9,536
            2,0.25481265783309937,0.5,,0.05,0.277,14430.0,536
            """),
        para("The JSON Lines file has one object per event (602 lines here: 300 batches, 300 epochs, start and end). "
             "Each line is self-contained, so the file can be tailed live, loaded into pandas or a spreadsheet, or "
             "shipped to a log system. Two lines from the run:"),
        output("""
            {"time":"2026-09-25T15:24:20.1281123+00:00","event":"batch","epoch":1,"batch":1,"step":1,"batch_size":4,"loss":0.26254576444625854,"learning_rate":0.05,"data_ms":4.088,"compute_ms":4.9784}
            {"time":"2026-09-25T15:24:20.1310109+00:00","event":"epoch","epoch":1,"loss":0.26254576444625854,"metrics":{"accuracy":0.5},"learning_rate":0.05,"duration_ms":11.4316,"samples_per_second":349.90727457223835,"memory_in_use":536,"memory_cached":832,"best":true}
            """, caption="train.jsonl (excerpt)"),
        reftable(["Hook", "Constructor", "Use it for"], [
            ["<code>ConsoleLogger</code>", "<code>(levels = Training, epochInterval = 1, batchInterval = 1, output = Console.Out)</code>", "Human-readable progress; any <code>TextWriter</code>"],
            ["<code>MetricsRecorder</code>", "<code>(levels = Training)</code>", "Keeping epochs/batches in memory; <code>SaveCsv(path)</code>"],
            ["<code>JsonLinesLogger</code>", "<code>(path, levels = Training | Batches | Inference)</code>", "Machine-readable log file, written on a background task; dispose to flush"],
            ["<code>ChannelTelemetry</code>", "<code>(levels, capacity = 100,000)</code>", "Handing events to your own async consumer"],
        ], caption="Table 19.3 — Built-in hooks"),
        h2("19.4 Channels: never slow training down"),
        para("Hooks run synchronously on the training thread. Anything slow (network calls, databases, UI updates) "
             "belongs behind a <code>ChannelTelemetry</code>: the training thread only writes a record into a bounded "
             "channel (dropping the oldest when full) and your consumer reads at its own pace."),
        mex("an async consumer", None,
            """
            var channel = new ChannelTelemetry(TelemetryLevel.Training);
            var consumer = Task.Run(async () =>
            {
                int other = 0;
                await foreach (var record in channel.Reader.ReadAllAsync())
                {
                    if (record.Event is EpochCompleted e && e.Epoch % 100 == 0)
                        Console.WriteLine($"  consumer saw epoch {e.Epoch} loss {e.Loss:F6}");   // or: await dashboard.SendAsync(e)
                    else
                        other++;
                }
                return other;
            });

            using (Telemetry.Subscribe(channel))
                trainer.Fit(new DataLoader(xor, 4), epochs: 200);
            channel.Complete();                                   // ends the await foreach
            Console.WriteLine($"  other events: {await consumer}");
            """,
            out="""
              consumer saw epoch 100 loss 0.000084
              consumer saw epoch 200 loss 0.000059
              other events: 200
            """),
        cpugpu("production settings",
               """
               // CPU service: log inference latency to a file, nothing else
               using var log = new JsonLinesLogger("inference.jsonl", TelemetryLevel.Inference);
               using var sub = Telemetry.Subscribe(log);
               """,
               """
               // GPU training job: epochs only (no per-batch synchronization), to console and file
               using var console = Telemetry.Subscribe(new ConsoleLogger(TelemetryLevel.Training, epochInterval: 10));
               using var file = new JsonLinesLogger("run.jsonl", TelemetryLevel.Training);
               using var sub = Telemetry.Subscribe(file);
               """,
               "The GPT Web API (" + ch("webapi") + ") streams the same kind of events to a browser."),
        honestbox("How cheap is 'off'?",
                  "<p>Every publishing site first checks one static field with a bitwise AND. With no subscribers "
                  "nothing is allocated, no timestamps are taken and no strings are built: the whole cost is that one "
                  "check per operation, far below the cost of the operation itself.</p>"),
        practice([
            (1, "Print one line every 10 epochs and write every epoch to <code>run.csv</code>.",
             "Subscribe <code>new ConsoleLogger(TelemetryLevel.Training, epochInterval: 10)</code> and a "
             "<code>MetricsRecorder</code>; after <code>Fit</code> call <code>recorder.SaveCsv(\"run.csv\")</code>."),
            (1, "Which level shows the gradient norm, and what does it cost?",
             "<code>TelemetryLevel.Batches | TelemetryLevel.Gradients</code>; it adds kernels to compute the norm and a "
             "synchronization per batch."),
            (2, "Write a hook that saves the model whenever a new best validation loss is reached.",
             "Implement <code>ITelemetryHook</code> with <code>Levels =&gt; TelemetryLevel.Training</code> and "
             "<code>OnEpochCompleted(in EpochCompleted e) { if (e.IsBest) model.Save(path); }</code>."),
            (2, "Measure the real per-layer GPU time of a CNN forward pass.",
             "Set <code>Telemetry.SynchronizeForTiming = true</code>, subscribe a <code>ConsoleLogger(TelemetryLevel.Layers)</code>, "
             "run one warm-up <code>Predict</code> and then the measured one."),
            (3, "Stream live training curves to a web page.",
             "Put a <code>ChannelTelemetry</code> between the Trainer and an ASP.NET endpoint that returns "
             "<code>TypedResults.ServerSentEvents</code> of the <code>EpochCompleted</code> records; " + ch("webapi") +
             " shows the server-sent-events pattern."),
        ], PART),
        footer("Telemetry", "Telemetry level", "Hook", "Event", "JSON Lines", "Channel", "Synchronization",
               "Profiling", "Latency", "Throughput"),
    )

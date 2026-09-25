# NeuralSharp

A self-contained neural network library for **.NET 10**, written in pure C#. There is **no TensorFlow.dll,
no NuGet dependencies and no native libraries of its own**. It includes N-D tensors with automatic
differentiation; dense, convolutional, recurrent and transformer layers; regression and classification
losses; optimizers with learning-rate schedules; a training loop; data loading; resource limits; and
telemetry hooks.

| Backend | How it works | Requirements |
|---------|--------------|--------------|
| **CPU** | `Vector<T>` SIMD kernels (AVX2/AVX-512/NEON), a register-blocked matrix multiply, multi-threading for large tensors, pooled buffers | Any machine running .NET 10 |
| **CUDA** | P/Invoke straight into the NVIDIA **driver** API (`nvcuda.dll` / `libcuda.so.1`). The GPU kernels are PTX assembly generated in C# (`PtxKernels.cs`), and the driver JIT-compiles them for the installed GPU | An NVIDIA GPU and display driver. No CUDA Toolkit, cuBLAS, cuDNN or NVRTC |

`Device.Default` picks the first GPU when a driver is present and falls back to the CPU otherwise.
Set `NEURALSHARP_DISABLE_CUDA=1` to force the CPU.

## Layout

```
src/NeuralSharp/
  Tensor*.cs                        N-D tensors, operators, autograd, shape ops, number-type interop
  Device.cs, ComputeResources.cs    devices; thread and memory budgets
  Autograd.cs, TensorScope.cs       NoGrad(); deterministic disposal
  Losses.cs                         MSE, MAE, CrossEntropy, BinaryCrossEntropy(WithLogits)
  Layers/                           Linear, Conv2d, MaxPool2d, GlobalAveragePool2d, Flatten,
                                    BatchNorm, LayerNorm, Embedding, LSTM, GRU,
                                    MultiHeadAttention, TransformerEncoderLayer, PositionalEncoding,
                                    ReLU, Tanh, Sigmoid, GELU, Softmax, Dropout, Lambda, Sequential
  Optimizers/                       Sgd, Adam, AdamW, schedulers (step, exponential, cosine + warm-up)
  Data/                             Dataset (CSV, class labels, feature shapes), scalers, DataLoader
  Training/                         Trainer, Metric (MAE, RMSE, Accuracy), RegressionReport
  Diagnostics/                      Telemetry hub, events, ConsoleLogger, MetricsRecorder,
                                    ChannelTelemetry, JsonLinesLogger
  Backends/Cpu, Backends/Cuda       device implementations (CPU SIMD kernels, PTX kernels)
samples/
  NeuralSharp.Samples.Xor             the classic XOR problem
  NeuralSharp.Samples.HousePrices     regression: predict house prices from a CSV file
  NeuralSharp.Samples.Classification  multi-class: 3 spirals, softmax + cross-entropy, BatchNorm
  NeuralSharp.Samples.Images          CNN: classify drawn shapes (Conv2d, MaxPool2d, BatchNorm)
  NeuralSharp.Samples.Sequences       sentiment with negation: bag-of-words vs LSTM, GRU, Transformer
  NeuralSharp.Samples.Ocr             OCR: CNN character recognizer + line segmentation, reads PGM images
  NeuralSharp.Samples.Transformer     small GPT: character-level causal transformer that generates text
  NeuralSharp.Samples.GptApi          ASP.NET Core Web API + browser UI serving the GPT (Scalar docs, streaming)
  Shared/SampleOptions.cs             command-line options shared by the samples (train / predict modes)
  Shared/Gpt/                         GPT model, generation with metrics, training (console + Web API)
tests/NeuralSharp.Tests             self-contained test runner (runs on every available device)
```

## Samples

| Sample | Shows | Typical CPU result |
|--------|-------|--------------------|
| `Xor` | smallest possible network | 4/4 correct in 0.1 s |
| `HousePrices` | CSV loading, scaling, regression, early stopping | R² 0.975, 5.3% mean error, 1 s |
| `Classification` | softmax + cross-entropy, BatchNorm, AdamW, cosine schedule, confusion matrix | 97.8% accuracy, 2.4 s |
| `Images` | Conv2d, MaxPool2d, BatchNorm, Flatten on 16×16 images | 100% accuracy, about 11 s |
| `Sequences` | Embedding, LSTM, GRU, Transformer vs an order-blind baseline | LSTM/GRU/Transformer 99.5–100%, bag of words 70% |
| `Ocr` | CNN over 36 characters, projection-profile segmentation, PGM input | 100% per character, 99.9% of characters across whole lines |
| `Transformer` | decoder-only GPT: causal attention, sparse cross-entropy, sampling | 89% next-character accuracy, 100% real words generated |
| `GptApi` | serving a model: REST + server-sent events, Scalar, browser UI | about 600 characters/s on 4 CPU cores |

### Train and predict modes

Every console sample trains, saves its model under `models/` (next to the executable), and can then run
**inference only** from the saved model:

```bash
dotnet run -c Release --project samples/NeuralSharp.Samples.HousePrices                       # train + save
dotnet run -c Release --project samples/NeuralSharp.Samples.HousePrices -- --predict --input "2100,4,2,15,9.5,7,2,0,6500"
dotnet run -c Release --project samples/NeuralSharp.Samples.Sequences -- --predict --input "the movie was not good;not bad at all"
dotnet run -c Release --project samples/NeuralSharp.Samples.Ocr -- --predict --input "HELLO WORLD 2026"
dotnet run -c Release --project samples/NeuralSharp.Samples.Ocr -- --predict --image scan.pgm
dotnet run -c Release --project samples/NeuralSharp.Samples.Transformer -- --predict --input "the old wizard " --temperature 0.8
```

| Sample | `--input` in predict mode |
|--------|---------------------------|
| `Xor` | `"a,b;a,b"` bit pairs |
| `HousePrices` | 9 features per house, `;` between houses (or `--data file.csv` to price every row) |
| `Classification` | `"x,y;x,y"` points in [-1, 1] |
| `Images` | shape names to draw and classify, e.g. `"circle,cross"` |
| `Sequences` | sentences separated by `;`, judged by all four saved models side by side |
| `Ocr` | text to render and read (or `--image file.pgm`) |
| `Transformer` | prompt to continue (`--length`, `--temperature`, `--top-k`) |

Everything a model needs at inference time is saved next to its weights. That includes the scalers for
house prices, the BatchNorm running statistics, and the GPT's vocabulary and architecture (a `.json`
file). `--model <path>` chooses another location.

### GPT Web API

```bash
dotnet run -c Release --project samples/NeuralSharp.Samples.GptApi
# http://localhost:5080         inference UI
# http://localhost:5080/scalar  API reference (Scalar)
```

| Endpoint | Purpose |
|----------|---------|
| `GET /api/status` | Loading / Training (with live progress) / Ready / Failed |
| `GET /api/model` | parameters, blocks, heads, width, context, vocabulary, validation results, layer summary |
| `GET /api/devices` | CPU and CUDA GPUs with memory usage and the active one |
| `POST /api/generate` | one or more samples, each character's probability, entropy, top-5 alternatives and latency, plus aggregate metrics (including decoding mode) |
| `POST /api/generate/stream` | the same as server-sent events: `token` events (with their sample index) every `chunkSize` characters, then `metrics` |

The request body is `{ "prompt", "length", "temperature", "topK", "seed", "device": "cpu" | "cuda",
"samples", "useCache", "useGraph", "chunkSize" }`. The UI has matching controls: a samples slider
(shown as tabs), and KV cache and CUDA graph switches for comparing modes.
The device can change per request, and the model moves between CPU and GPU as needed.

The service loads `Gpt:ModelPath` from `appsettings.json`. Point it at a model saved by the
`Transformer` sample (or pass `--Gpt:ModelPath path`). When no model exists, the service trains one in
the background and reports progress through NeuralSharp telemetry.

The browser UI (`wwwroot/index.html`, dark and light themes, responsive) is laid out in two columns:
- **Left:** the streamed output, with each character coloured by the model's confidence (hover a
  character for its alternatives, click it for the token inspector); metric cards for throughput,
  total time, first-token latency, time per token, confidence, perplexity, entropy and device
  memory; and live confidence and latency charts.
- **Right:** prompt, length, temperature, top-k, seed and streaming controls; the CPU/GPU switch; and
  the model's parameters and layer summary.

All samples take the same options:

```bash
dotnet run -c Release --project samples/NeuralSharp.Samples.HousePrices               # GPU if available, else CPU
dotnet run -c Release --project samples/NeuralSharp.Samples.HousePrices -- --cpu      # force CPU
dotnet run -c Release --project samples/NeuralSharp.Samples.HousePrices -- --cuda     # force GPU
dotnet run -c Release --project samples/NeuralSharp.Samples.Xor -- --help
```

| Option | Meaning |
|--------|---------|
| `--device auto\|cpu\|cuda\|cuda:N`, `--cpu`, `--cuda` | where to run |
| `--threads N` | CPU threads to use (default: all cores) |
| `--gpu-memory MiB`, `--cpu-memory MiB` | cap tensor memory (default: unlimited) |
| `--epochs N`, `--batch-size N` | training settings |
| `--log training,batches,gradients,layers,operations,inference,all` | what the console logger prints |
| `--log-file run.jsonl` | also write telemetry as JSON Lines |
| `--data path.csv` | dataset file (house prices) |

The house-price sample loads `data/houses.csv`: 2,500 synthetic but realistic listings with 9 features
(area, bedrooms, bathrooms, age, distance to the city centre, quality, garage, pool, lot size).
It splits them 80/20, standardizes features and prices, and trains a 9→64→32→1 network with
dropout, Adam and early stopping. It then reports test error in dollars, prices three new listings,
and saves the weights, scalers and a training-history CSV. A typical CPU run reaches **R² ≈ 0.97 and
about 5% mean error** in about a second, close to the ±6% noise built into the data.

## Using the library

### Training with the `Trainer`

```csharp
using NeuralSharp;
using NeuralSharp.Data;
using NeuralSharp.Layers;
using NeuralSharp.Optimizers;
using NeuralSharp.Training;

var data = Dataset.LoadCsv("houses.csv", new CsvOptions { TargetColumns = ["price"], IgnoreColumns = ["id"] });
var (train, test) = data.Split(0.8, seed: 1);

var x = StandardScaler.FitFeatures(train);
var y = StandardScaler.FitTargets(train);
var trainLoader = new DataLoader(train.Scale(x, y), batchSize: 64, shuffle: true);
var testLoader  = new DataLoader(test.Scale(x, y), batchSize: 512);

using var model = new Sequential
{
    new Linear(data.FeatureCount, 64), new ReLU(), new Dropout(0.05f),
    new Linear(64, 32), new ReLU(),
    new Linear(32, 1),
};

var trainer = new Trainer(model, new Adam(model.Parameters(), 2e-3f), Losses.MeanSquaredError)
{
    Metrics = { Metric.MeanAbsoluteError },
    EarlyStoppingPatience = 30,             // restores the best weights when it stops
};
TrainingHistory history = trainer.Fit(trainLoader, epochs: 400, validation: testLoader);
float[,] predictions = trainer.Predict(test.Scale(x, y));
```

### Low-level: tensors and a hand-written loop

```csharp
var a = Tensor.From(new float[,] { { 1, 2 }, { 3, 4 } }, requiresGrad: true);
var loss = (a.MatMul(a).Relu() * 2f + 1f).Mean();
loss.Backward();                      // a.Grad holds d(loss)/da

for (int epoch = 0; epoch < 1000; epoch++)
{
    using var scope = new TensorScope();   // releases this iteration's tensors
    var l = Losses.MeanSquaredError(model.Forward(inputs), targets);
    optimizer.ZeroGrad();
    l.Backward();
    optimizer.Step();
}

using var prediction = model.Predict(inputs);   // evaluation mode, no gradients
```

### Classification

Classifiers output raw scores (logits). `CrossEntropy` applies log-softmax itself, which is more
stable than a separate softmax layer. Add `Softmax` only when you want probabilities at inference time.

```csharp
var data = Dataset.FromClassLabels(features, labels, classes: 3);      // one-hot targets
// or: Dataset.LoadCsv(path, new CsvOptions { TargetColumns = ["label"] }).ToOneHot(3)

var trainer = new Trainer(model, new AdamW(model.Parameters(), 1e-3f), (p, t) => Losses.CrossEntropy(p, t))
{
    Metrics = { Metric.Accuracy },
    Scheduler = new CosineAnnealing(optimizer, totalEpochs: 100, warmupEpochs: 5),
};
using var probabilities = model.Predict(x).Softmax();
using var classes = probabilities.ArgMax();
```

For two classes with one output column, use `Losses.BinaryCrossEntropyWithLogits` and
`Metric.BinaryAccuracy(threshold: 0)`.

### Images (CNN)

```csharp
var images = Dataset.FromClassLabels(pixels, labels, 10).WithFeatureShape(1, 28, 28);   // batches are [N, 1, 28, 28]
var cnn = new Sequential
{
    new Conv2d(1, 32, kernelSize: 3, padding: 1), new BatchNorm(32), new ReLU(), new MaxPool2d(2),
    new Conv2d(32, 64, kernelSize: 3, padding: 1), new BatchNorm(64), new ReLU(), new MaxPool2d(2),
    new Flatten(), new Linear(64 * 7 * 7, 128), new ReLU(), new Dropout(0.3f), new Linear(128, 10),
};
```

`Conv2d` unfolds image patches (im2col) and runs one large matrix product on the same optimized GEMM
as `Linear`, on both CPU and GPU.

### Sequences (RNN and transformer)

Token ids go in as floats, shape [batch, time]:

```csharp
var lstm = new Sequential
{
    new Embedding(vocabulary, 64), new LSTM(64, 128), new Linear(128, classes),   // LSTM returns the last state
};

var transformer = new Sequential
{
    new Embedding(vocabulary, 64), new PositionalEncoding(maxLength, 64),
    new TransformerEncoderLayer(64, heads: 4), new TransformerEncoderLayer(64, heads: 4),
    new LayerNorm(64), new Lambda(x => x.Mean(1), "MeanOverTime"), new Linear(64, classes),
};
```

Use `LSTM(..., returnSequences: true)` or `GRU` for per-step outputs. `MultiHeadAttention(dim, heads, causal: true)`
masks future positions for autoregressive models. Setting `Trainer.MaxGradientNorm` clips gradients,
which recurrent networks usually need.

### Tensor operations

N-D tensors support batched `MatMul` (with transpose flags), `Permute`, `Transpose`, `Narrow`,
`Tensor.Concat`, `Tensor.Stack`, `Flatten`, `Sum(dim)`, `Mean(dim)`, `Softmax`, `LogSoftmax`,
`ArgMax`, `Exp`, `Log` and `Gelu`. Adding a tensor whose shape matches the trailing dimensions
broadcasts it, as with a bias [F] over [N, F] or a mask [T, T] over [B, T, T]. All of these are
differentiated automatically.

### Other number types

Every kernel computes in float32, the standard for neural networks and the only type with fast
SIMD and GPU paths everywhere. Data of any numeric type converts at the edges:

```csharp
using var t = Tensor.From<double>(doubles, [rows, cols]);   // also int, long, byte, Half, decimal, ...
using var m = Tensor.From(new int[,] { { 1, 2 }, { 3, 4 } });
int[] ints = t.ToArray<int>();                               // saturating, truncating conversion
```

### Fast autoregressive generation

Generating text one token at a time is dominated by overhead rather than math, so NeuralSharp provides
three tools that work together (`CharGpt.Generate` in `samples/Shared/Gpt` shows them in use):

* **KV cache.** Pass a `DecodingContext` to `Sequential.ForwardCached(ids, context)`. The prompt is
  processed once (prefill), then every step processes only the newest token, while each attention
  layer's keys and values accumulate in a `KeyValueCache`. `MultiHeadAttention`,
  `TransformerEncoderLayer` and `PositionalEncoding` implement `ICachedModule`.
* **On-device sampling.** `TokenSampler` draws the next token (temperature, top-k, top-p, min-p,
  repeat/presence/frequency penalties over a device-side history of recent tokens) on the device
  that holds the logits and writes each token's probability, entropy and top-5 alternatives to a
  device buffer. The next step therefore needs no host round trip; `Read` fetches the statistics in
  chunks. Its randomness is counter-based (seed, step, row), so results are reproducible and match
  between CPU and GPU.
* **Compute graphs.** `DecodingContext.CaptureStep(step)` records a whole decoding step, and
  `ReplayStep` runs it again with one launch; on CUDA this is a CUDA Graph. The position counter,
  causal mask and cache offsets are computed on the device, so one recording is valid at every
  position. On the CPU, or if the driver refuses, replay simply re-runs the step.
* **Batching and fused kernels.** Many samples decode together as one batch. During inference,
  scale+mask+softmax, LayerNorm and bias+GELU each run as a single kernel; training uses the
  differentiable path.

Measured on the sample GPT (341K parameters) on a 4-core CPU container
(`dotnet run --project samples/NeuralSharp.Samples.Transformer -- --predict --benchmark true`):

| Mode | chars/s | Speedup |
|------|---------|---------|
| Full recompute, 1 sample | 283 | 1× |
| KV cache, 1 sample | 4,119 | 14.5× |
| KV cache, 8 samples | 6,595 | 23× |
| KV cache, 32 samples | 8,957 | 32× |

On a GPU, CUDA graphs remove the per-step launch cost, and batching keeps the GPU busy.

### Text generation, chat and tools (`NeuralSharp.Generation`)

A higher-level layer on top of the cache, sampler and graphs, with the options and conventions of common
local LLM servers:

| Type | Purpose |
|------|---------|
| `ITokenizer`, `CharTokenizer` | text ↔ token ids |
| `GenerationOptions` | `Temperature`, `TopK`, `TopP`, `MinP`, `RepeatPenalty`, `RepeatLastN`, `PresencePenalty`, `FrequencyPenalty`, `Seed`, `NumCtx`, `NumPredict`, `Stop`, plus `UseCache`, `UseGraph`, `ChunkSize` |
| `TextGenerator` | streams a continuation: prompt truncated to `NumCtx`, sliding context window, stop sequences (never partially emitted), done reason `stop` / `length`, prompt and generation timings |
| `ChatMessage`, `ToolDefinition`, `ToolCall` | conversations with `system`, `user`, `assistant` and `tool` roles, and function tools |
| `ChatTemplate`, `ChatMLTemplate` | renders a conversation and its tools as the prompt (Qwen-style ChatML: `<think>`, `<tool_call>`, `<tool_response>`); `think: false` closes an empty reasoning block |
| `ChatOutputParser` | splits streamed output into reasoning, answer and tool calls (JSON), holding back partial tags |
| `ChatGenerator` | chat = template + generator + parser; streams `ChatChunk`s and ends with the full assistant message and statistics |
| `ModelHost<T>`, `KeepAlive` | keeps models loaded and unloads each one when its keep-alive (`"30m"`, `"1h30m"`, `300`, `0`, `-1`) expires |

```csharp
var chat = new ChatGenerator(new TextGenerator(model, new CharTokenizer(vocabulary), contextLength: 256));
var request = new ChatRequest(
    [new ChatMessage("system", "You are a helpful assistant."), new ChatMessage("user", "What is the latest Ollama version?")],
    Tools: [new ToolDefinition("web_fetch", "Fetch a page.", JsonNode.Parse("""{"type":"object","properties":{"url":{"type":"string"}}}"""))],
    Think: true,
    Options: new GenerationOptions { Temperature = 1f, TopK = 20, TopP = 0.95f, NumCtx = 4096, NumPredict = 2048 });
foreach (var chunk in chat.Stream(request))
    Console.Write(chunk.Delta.Thinking + chunk.Delta.Content);   // chunk.Delta.ToolCalls: completed tool calls
```

### Ollama-compatible chat API

The GPT Web API also serves `POST /api/chat` with the Ollama request body: `model` (any name selects the
served model), `messages`, `stream` (NDJSON, default true), `think` (true/false or low/medium/high),
`keep_alive`, `options` (the keys above in snake_case; other keys are accepted and ignored) and `tools`.
Replies have Ollama's shape: `message.content`, `message.thinking`, `message.tool_calls`, `done`,
`done_reason` and nanosecond `total_duration`, `load_duration`, `prompt_eval_count`,
`prompt_eval_duration`, `eval_count`, `eval_duration`. `GET /api/tags`, `GET /api/ps` (loaded models with
`expires_at`) and `GET /api/version` are there too.

```bash
curl http://localhost:5080/api/chat -d '{"model":"any","stream":true,"think":true,"keep_alive":"30m",
  "options":{"temperature":1,"top_k":20,"top_p":0.95,"num_ctx":4096,"num_predict":2048},
  "messages":[{"role":"user","content":"What is the latest Ollama version?"}],
  "tools":[{"type":"function","function":{"name":"web_fetch","parameters":{"type":"object","properties":{"url":{"type":"string"}}}}}]}'
```

The endpoint serves `Gpt:ChatModelPath` if that file exists, otherwise `Gpt:ModelPath`. A model trained on
plain text only continues text. To see reasoning and tool calls, train the small chat model on synthetic
ChatML transcripts (reasoning, `web_fetch` calls, answers citing tool results):

```bash
dotnet run -c Release --project samples/NeuralSharp.Samples.Transformer -- --chat true   # saves models/chat.weights and runs a two-turn demo
```

### Optimizers and schedules

`Sgd(momentum, weightDecay)`, `Adam(weightDecay)` (L2) and `AdamW` (decoupled weight decay) are
available, together with the `StepDecay`, `ExponentialDecay`, `CosineAnnealing(warmupEpochs)` and
`LambdaSchedule` schedulers. `optimizer.ClipGradientNorm(max)` and `optimizer.GradientNorm()` work
in hand-written loops too.

## Telemetry: logging and tracking

Every step of training and inference is published to a hub, and you choose what to listen to.
Hooks are **zero-cost when nothing is subscribed**. Each publishing site checks one static field
with a bitwise AND, and no event is built or allocated unless some hook asked for that level.

```csharp
using NeuralSharp.Diagnostics;

using var a = Telemetry.Subscribe(new ConsoleLogger(TelemetryLevel.Training, epochInterval: 10));
using var b = Telemetry.Subscribe(new MetricsRecorder());                        // in memory, SaveCsv()
await using var file = new JsonLinesLogger("run.jsonl", TelemetryLevel.All);     // background writer
using var c = Telemetry.Subscribe(file);
```

| Level | Event | Data |
|-------|-------|------|
| `Training` | `TrainingStarted`, `EpochCompleted`, `TrainingCompleted` | model summary, device, sizes; per-epoch loss, metrics, validation loss/metrics, learning rate, duration, samples/s, memory, best flag; early-stop info |
| `Batches` | `BatchCompleted` | epoch, batch, global step, batch loss, learning rate, data-wait time, compute time |
| `Gradients` | (adds to `BatchCompleted`) | global gradient L2 norm |
| `Layers` | `LayerForward` | layer name/type, nesting depth, input and output shapes, time, train/eval mode |
| `Operations` | `OperationCompleted` | every tensor op, forward and backward (∇), shape, device, time |
| `Inference` | `InferenceCompleted` | model, samples, shapes, device, latency, samples/s |

**Custom hooks.** Implement `ITelemetryHook`, set `Levels`, and override only the methods you need.
The other methods default to no-ops. Events are `readonly record struct`s passed by `in`, so they
are not copied or boxed.

```csharp
sealed class SlackAlert : ITelemetryHook
{
    public TelemetryLevel Levels => TelemetryLevel.Training;
    public void OnEpochCompleted(in EpochCompleted e) { if (double.IsNaN(e.Loss)) Alert("loss diverged"); }
}
```

**Why synchronous hooks plus an optional `Channel`.** A direct interface call is the fastest possible
dispatch, with no queueing, allocation or thread hop. Hooks that do slow work, such as files,
databases or HTTP, should use `ChannelTelemetry`. It copies each event into a bounded `Channel`
that never blocks training (the oldest events are dropped when it is full), and your consumer reads
it with `await foreach`. `JsonLinesLogger` is built this way.

GPU work is asynchronous, so layer and operation timings measure launch time by default. Set
`Telemetry.SynchronizeForTiming = true` when profiling to get true execution times.

## Data loading

* `Dataset.LoadCsv(path, options)` reads numeric CSV files. Lines are parsed in parallel, and errors
  name the line and column. `Dataset.FromArrays` and `Dataset.FromFlat` build a dataset from memory.
* `Split`, `Subset` and `Scale` return new, immutable datasets.
* `StandardScaler` and `MinMaxScaler` are fitted on training data. They provide `Transform` and
  `InverseTransform` (to turn predictions back into real units), plus `Save`/`Load` for deployment.
* `DataLoader` batches and shuffles data and moves each batch to the device. For large batches, the
  next batch is gathered on a worker thread while the current one trains, using double-buffered
  prefetch.

## Resources and parallelism

```csharp
ComputeResources.MaxCpuThreads  = 4;                   // default: every core
ComputeResources.GpuMemoryLimit = 2L << 30;            // bytes per GPU; default: unlimited
ComputeResources.CpuMemoryLimit = 1L << 30;            // default: unlimited
MemoryUsage usage = ComputeResources.GetMemoryUsage(Device.Default);   // in use / cached / limit
ComputeResources.ReleaseCachedMemory();
```

By default nothing is reserved up front, and memory is allocated as the network needs it. Freed blocks
are cached and reused, so a training loop stops allocating after its first iteration. Going over a
limit first releases the cache, then throws `ResourceLimitExceededException` with a clear message.

Parallelism is used where it pays off:

* SIMD in every CPU kernel.
* Multi-threaded kernels above 65,536 elements, and a multi-threaded matrix multiply for large
  products. On 4 AVX2 cores, a 1024×1024 multiply reaches about 86 GFLOP/s.
* Parallel CSV parsing.
* Background batch prefetch.
* On the GPU, thousands of threads per operation. Loss and metric sums stay on the device, so each
  epoch needs only one synchronization.

Everything on the CPU respects `MaxCpuThreads`, and small tensors stay single-threaded because
scheduling would cost more than it saves.

## NPU support

NPUs (Intel AI Boost, AMD Ryzen AI / XDNA, Qualcomm Hexagon) are **not supported**, and they cannot be
supported under this library's no-dependency rule:

* Unlike NVIDIA GPUs, which accept PTX through the driver, NPUs expose no general-purpose instruction
  set. They only run networks compiled by the vendor's own toolchain: OpenVINO for Intel, Ryzen AI /
  Vitis AI for AMD, QNN for Qualcomm, or DirectML / Windows ML on Windows.
* Those toolchains are the "prebuilt libraries" the project excludes.
* Current NPUs are mostly inference engines (INT8/FP16) and do not support training.

The backend design (`Backends/Backend.cs`) leaves room for an optional add-on package, for example
`NeuralSharp.OpenVino`, that runs inference on an NPU while the core stays dependency-free.

## Tests

```bash
dotnet run -c Release --project tests/NeuralSharp.Tests
```

There are 62 tests. They cover reference comparisons for every kernel (matrix products, softmax,
convolution and pooling against direct implementations) and finite-difference gradient checks for
every op and layer, including their weights. They also cover end-to-end learning (regression, spiral
classification, a CNN, LSTM and transformer sequence models), optimizers and schedules, CSV parsing,
data loading, telemetry, memory limits and thread budgets, the sampler's filters and penalties, and the
generation layer (stop sequences, context sliding, cache/graph/recompute agreement, template rendering,
streaming parser, keep-alive expiry). The suite runs on every available device.

## Status

* **Verified on real hardware.** 51 tests pass on both the CPU and an NVIDIA GeForce RTX 5050
  Laptop GPU (Blackwell), 102 of 102. The 5 newer decoding tests (KV cache, batched decoding, graph
  replay, sampler, fused kernels) and the 6 newest ones (sampler filters and penalties, generation layer)
  pass on the CPU and still need a run on a GPU. That covers every GPU kernel: matrix products, softmax,
  normalization, embeddings, convolution, pooling, recurrent and attention layers, and end-to-end
  training of classifiers, a CNN, an LSTM and a transformer.
* The 59 GPU kernels also assemble without errors for sm_50, sm_75, sm_89 and sm_120 (checked with
  `ptxas` 12.9), covering Maxwell through Blackwell.
* Everything computes in float32. Tensors hold up to 2³¹ elements, and embedding ids must be below
  2²⁴ (the largest integer a float stores exactly).
* Small models such as the samples are dominated by kernel-launch overhead on the GPU. Recurrent
  models are hit hardest, since they launch kernels for every time step. The GPU pays off with wide
  layers, large batches and convolutions: the CNN test trains faster on the RTX 5050 than on a
  16-thread CPU.

"""Chapter 36 — Serving Models over HTTP."""
from gen import *

PART = "VII"

PRICE_API = """
    using NeuralSharp;
    using NeuralSharp.Data;
    using NeuralSharp.Layers;

    var builder = WebApplication.CreateBuilder(args);
    builder.Services.AddSingleton<PricePredictor>();          // loaded once, shared by all requests
    var app = builder.Build();

    app.MapPost("/predict", (House[] houses, PricePredictor predictor) =>
        houses.Length is 0 or > 1000
            ? Results.BadRequest("send 1-1000 houses")
            : Results.Ok(predictor.Predict(houses)));
    app.MapGet("/health", (PricePredictor p) => Results.Ok(new { device = p.Device.ToString(), parameters = p.ParameterCount }));
    app.Run();

    public sealed record House(float Area, float Bedrooms, float Bathrooms, float Age, float DistanceKm,
                               float Quality, float Garage, float Pool, float Lot);

    public sealed class PricePredictor : IDisposable
    {
        private const int Features = 9;
        private readonly Sequential _model;
        private readonly StandardScaler _features, _price;

        public PricePredictor(IConfiguration config)
        {
            string dir = config["Model:Directory"] ?? "models";
            Device = config["Model:Device"] == "cuda" && NeuralSharp.Device.IsCudaAvailable
                ? NeuralSharp.Device.Cuda() : NeuralSharp.Device.Cpu;
            _model = new Sequential
            {
                new Linear(Features, 64, device: Device), new ReLU(), new Dropout(0.05f),
                new Linear(64, 32, device: Device), new ReLU(),
                new Linear(32, 1, device: Device),
            };
            _model.Load(Path.Combine(dir, "house-price.weights"));
            _model.Eval();                                      // once: requests never switch modes
            _features = StandardScaler.Load(Path.Combine(dir, "house-price.features.txt"));
            _price = StandardScaler.Load(Path.Combine(dir, "house-price.price.txt"));
        }

        public Device Device { get; }
        public long ParameterCount => _model.ParameterCount;

        public float[] Predict(IReadOnlyList<House> houses)
        {
            var x = new float[houses.Count * Features];
            for (int i = 0; i < houses.Count; i++)
            {
                var h = houses[i];
                float[] row = [h.Area, h.Bedrooms, h.Bathrooms, h.Age, h.DistanceKm, h.Quality, h.Garage, h.Pool, h.Lot];
                row.CopyTo(x, i * Features);
            }
            _features.Transform(x, Features);
            using var input = Tensor.From(x, [houses.Count, Features], Device);
            using var output = _model.Predict(input);
            var prices = output.ToArray();
            _price.InverseTransform(prices, 1);
            return prices;
        }

        public void Dispose() => _model.Dispose();
    }
"""


def build():
    return page(
        chapter_open(
            "webapi",
            "Most trained models end up behind an HTTP endpoint that other programs call. This chapter builds two "
            "services with ASP.NET Core minimal APIs: a small, complete prediction API for the house-price model, and "
            "a tour of the repository's GPT Web API (<code>NeuralSharp.Samples.GptApi</code>), which adds an OpenAPI "
            "description with a Scalar reference page, a browser UI, streaming with server-sent events, background "
            "training, device switching and live metrics.",
            "Register the model as a <b>singleton</b> service: load once at startup, <code>Eval()</code> once, share across requests.",
            "A stateless predictor serves concurrent requests without locks (200 concurrent requests, all succeeded).",
            "Stateful work (text generation with a KV cache) is serialized with a <code>SemaphoreSlim</code>.",
            "Stream long outputs with <code>TypedResults.ServerSentEvents</code>.",
            "Expose health, model info and device endpoints for operations.",
        ),
        h2("36.1 A prediction API in one file"),
        deriv("Creating it", [
            "<code>dotnet new web -n PriceApi</code> and <code>dotnet add reference …/NeuralSharp.csproj</code>.",
            "Replace <code>Program.cs</code> with the code below.",
            "Copy the three model files of " + ch("regression") + " into <code>models/</code> (or pass <code>--Model:Directory</code>).",
            "<code>dotnet run -c Release --urls http://localhost:5090</code> (add <code>--Model:Device cuda</code> for the GPU).",
        ]),
        snippet(PRICE_API, caption="Program.cs of a complete house-price API"),
        output("""
            $ curl localhost:5090/health
            {"device":"cpu","parameters":2753}

            $ curl -X POST localhost:5090/predict -H 'Content-Type: application/json' -d '[
                {"area":2100,"bedrooms":4,"bathrooms":2,"age":15,"distanceKm":9.5,"quality":7,"garage":2,"pool":0,"lot":6500},
                {"area":1200,"bedrooms":3,"bathrooms":1,"age":30,"distanceKm":12,"quality":5,"garage":1,"pool":0,"lot":4000}]'
            [390611.53,192872.94]

            $ seq 200 | xargs -P 32 -I{} curl -s -o /dev/null -w "%{http_code}\\n" -X POST localhost:5090/predict ... | sort | uniq -c
                200 200

            $ curl -X POST localhost:5090/predict -H 'Content-Type: application/json' -d '[]'
            "send 1-1000 houses"
            """, caption="Calling the API (the concurrent test sends 200 requests, 32 at a time)"),
        para("The predictions match the console sample's (<code>$390,612</code> and <code>$192,873</code>, " + ch("regression") + "). "
             "Because the model is in evaluation mode from the start and each request creates its own tensors, "
             "concurrent requests need no lock (" + ch("inference") + ", Section 35.4)."),
        cpugpu("configuring the device",
               """
               // appsettings.json
               { "Model": { "Directory": "models", "Device": "cpu" } }
               """,
               """
               // appsettings.Production.json on a GPU server
               { "Model": { "Directory": "/srv/models/house-price/v3", "Device": "cuda" } }
               // or on the command line: --Model:Device cuda
               """),
        h2("36.2 The GPT Web API"),
        reftable(["Endpoint", "Returns"], [
            ["<code>GET /api/status</code>", "Loading, Training (with live progress), Ready or Failed"],
            ["<code>GET /api/model</code>", "Parameters, layers, heads, width, context, vocabulary, training record, layer summary"],
            ["<code>GET /api/devices</code>", "CPU and every GPU, with memory usage and which one holds the model"],
            ["<code>POST /api/generate</code>", "Generated samples, every character with probability, entropy and top-5 alternatives, and metrics"],
            ["<code>POST /api/generate/stream</code>", "The same as server-sent events: <code>token</code> events, then a <code>metrics</code> event"],
            ["<code>GET /</code>, <code>GET /scalar</code>", "The browser UI (output and metrics left, inputs and model parameters right) and the API reference"],
        ], caption="Table 36.1 — NeuralSharp.Samples.GptApi"),
        snippet("""
            var builder = WebApplication.CreateBuilder(args);
            builder.Services.AddOpenApi();
            builder.Services.AddSingleton<GptService>();
            builder.Services.AddHostedService(services => services.GetRequiredService<GptService>());   // loads or trains in the background

            var app = builder.Build();
            app.UseDefaultFiles();
            app.UseStaticFiles();                                     // wwwroot/index.html
            app.MapOpenApi();
            app.MapScalarApiReference();                              // /scalar

            var api = app.MapGroup("/api");
            api.MapGet("/status", (GptService gpt) => gpt.Status);
            api.MapPost("/generate", async (GenerateRequest request, GptService gpt, CancellationToken ct) =>
                TypedResults.Ok(await gpt.GenerateAsync(request, ct)));
            api.MapPost("/generate/stream", (GenerateRequest request, GptService gpt, CancellationToken ct) =>
                TypedResults.ServerSentEvents(gpt.StreamAsync(request, ct)));
            app.Run();
            """, caption="Program.cs of the GPT API (abridged: error handling and OpenAPI descriptions removed)"),
        output("""
            $ curl localhost:5080/api/status
            {"status":"Ready","message":"Ready on cpu","training":null}

            $ curl localhost:5080/api/devices
            [{"id":"cpu","name":"CPU (4 threads, 8-wide SIMD)","type":"Cpu","available":true,"active":true,
              "memory":{"inUse":1389812,"cached":15204792,"limit":null,"reserved":16594604}},
             {"id":"cuda","name":"No NVIDIA GPU detected","type":"Cuda","available":false,"active":false,"memory":null}]

            $ curl localhost:5080/api/model
            {"name":"char-gpt","parameters":341309,"layers":3,"heads":4,"dim":96,"context":64,"vocabularySize":29,
             "trainedEpochs":2,"validationLoss":0.2504,"validationAccuracy":0.8938,"corpus":"built-in grammar", ...}
            """, caption="Status, devices and model (long values shortened)"),
        output("""
            $ curl -X POST localhost:5080/api/generate -H 'Content-Type: application/json' \\
                   -d '{"prompt":"the little robot ","length":60,"seed":6}'
            "text": "built a tiny garden and then carried the heavy box.\\na quiet ",
            first token: {"token":"b","probability":0.394,"entropy":2.55,
                          "alternatives":[b 0.394, f 0.205, r 0.093, s 0.086, w 0.067]}
            "metrics": {"mode":"KV cache","generatedTokens":60,"totalMs":28.98,"firstTokenMs":6.81,
                        "tokensPerSecond":2070.8,"averageProbability":0.882,"perplexity":1.226,
                        "contextResets":1,"graphNote":"graphs are GPU-only; CPU runs the step directly", ...}

            $ curl -N -X POST localhost:5080/api/generate/stream -d '{"prompt":"a curious cat ","length":20,"seed":3,"chunkSize":8}' ...
            event: token
            data: {"sample":0,"index":0,"token":"f","probability":0.2556171,"entropy":2.7526093,...}

            event: token
            data: {"sample":0,"index":1,"token":"o","probability":0.99999976,"entropy":8.111985E-06,...}
            """, caption="Generating and streaming (JSON condensed for print)"),
        para("The first character is genuinely uncertain (entropy 2.55 bits: several verbs could follow \"the little "
             "robot\"), while the next ones are nearly certain once the word has started. The API also accepts "
             "<code>\"device\": \"cuda\"</code> to move the model to the GPU between requests, <code>samples</code> for "
             "batched continuations, and <code>useCache</code>/<code>useGraph</code> to compare decoding modes (" + ch("generation") + ")."),
        honestbox("Why the GPT service uses a lock",
                  "<p>Generation owns stateful resources for its duration: the KV caches, the sampler, and on the GPU a "
                  "recorded graph. The service therefore lets one generation run at a time "
                  "(<code>SemaphoreSlim(1, 1)</code>) and queues the others; with a small model each takes milliseconds. "
                  "For higher throughput, batch several prompts' samples into one generation, or run several model "
                  "copies.</p>"),
        h2("36.3 Production checklist"),
        reftable(["Concern", "Practice"], [
            ["Startup", "Load, <code>Eval()</code>, warm up with one prediction, then report ready (health endpoint)"],
            ["Input validation", "Check sizes and ranges; reject NaN; cap batch sizes"],
            ["Throughput", "Accept batches; micro-batch single requests (" + ch("inference") + ")"],
            ["Resources", "<code>ComputeResources.MaxCpuThreads</code> below the core count, and memory limits, so the web server keeps capacity"],
            ["Monitoring", "<code>JsonLinesLogger</code> at <code>TelemetryLevel.Inference</code> for latency; log input distributions to catch drift"],
            ["Model updates", "Versioned folders; load the new version into a new singleton and swap references"],
            ["GPU servers", "One process per GPU, or <code>Device.Cuda(n)</code> per model copy"],
        ], caption="Table 36.2 — Serving in production"),
        practice([
            (1, "Why is the predictor registered as a singleton rather than per request?",
             "Loading weights and scalers is expensive (file reads, allocation, warm-up); a singleton loads once and shares the model."),
            (1, "Which endpoint would a load balancer's health check call?",
             "<code>/health</code> (or <code>/api/status</code> for the GPT API), returning success only when the model is loaded."),
            (2, "Add an endpoint that returns the churn probability for a customer.",
             "Load the churn weights and scaler in a singleton like <code>PricePredictor</code>; in the handler scale the features, "
             "<code>Predict</code>, apply <code>Sigmoid()</code>, and return <code>{ probability, churn = probability &gt;= threshold }</code>."),
            (2, "Log each request's latency to a JSON Lines file without writing timing code.",
             "Subscribe a <code>JsonLinesLogger(\"inference.jsonl\", TelemetryLevel.Inference)</code> at startup; every "
             "<code>Predict</code> call then writes an <code>inference</code> event with latency and throughput (" + ch("telemetry") + ")."),
            (3, "Serve two versions of the house-price model side by side for an A/B test.",
             "Register two predictors (keyed services, or a dictionary keyed by version), route a share of requests to each "
             "(e.g. by a hash of the client id), return the version in the response, and compare their errors on later "
             "observed prices."),
        ], PART),
        footer("Web API", "Minimal API", "Singleton", "Server-sent events", "OpenAPI", "Health check", "Micro-batching"),
    )

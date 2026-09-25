"""Chapter 38 — NeuralSharp in Every .NET Project Type."""
from gen import *

PART = "VII"

LIBRARY = """
    using NeuralSharp;
    using NeuralSharp.Data;
    using NeuralSharp.Layers;

    namespace Pricing;

    public sealed record House(float Area, float Bedrooms, float Bathrooms, float Age, float DistanceKm,
                               float Quality, float Garage, float Pool, float Lot);

    /// The house-price model as a reusable component, referenced by every application below.
    public sealed class HousePriceModel : IDisposable
    {
        public const int Features = 9;
        private readonly Sequential _model;
        private readonly StandardScaler _features, _price;

        private HousePriceModel(Sequential model, StandardScaler features, StandardScaler price, Device device)
        {
            _model = model; _features = features; _price = price; Device = device;
        }

        public Device Device { get; }

        /// The architecture, identical in training and inference.
        public static Sequential CreateNetwork(Device? device = null, Random? random = null) => new()
        {
            new Linear(Features, 64, device: device, random: random), new ReLU(), new Dropout(0.05f, random),
            new Linear(64, 32, device: device, random: random), new ReLU(),
            new Linear(32, 1, device: device, random: random),
        };

        public static HousePriceModel Load(string directory, Device? device = null)
        {
            device ??= Device.IsCudaAvailable ? Device.Cuda() : Device.Cpu;
            var model = CreateNetwork(device);
            model.Load(Path.Combine(directory, "house-price.weights"));
            model.Eval();
            return new HousePriceModel(model,
                StandardScaler.Load(Path.Combine(directory, "house-price.features.txt")),
                StandardScaler.Load(Path.Combine(directory, "house-price.price.txt")), device);
        }

        public float Predict(House house) => Predict([house])[0];

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

WORKER = """
    using Pricing;

    namespace PriceWorker;

    /// Watches a folder for CSV files of houses (9 columns, no header) and writes a priced copy of each.
    public sealed class Worker(ILogger<Worker> logger, IConfiguration config) : BackgroundService
    {
        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            string inbox = config["Inbox"] ?? "inbox", outbox = config["Outbox"] ?? "outbox";
            Directory.CreateDirectory(inbox);
            Directory.CreateDirectory(outbox);
            using var model = HousePriceModel.Load(config["ModelDirectory"] ?? "models");
            logger.LogInformation("Model loaded on {Device}", model.Device);

            using var timer = new PeriodicTimer(TimeSpan.FromSeconds(5));
            do
            {
                foreach (var file in Directory.GetFiles(inbox, "*.csv"))
                {
                    var houses = File.ReadLines(file)
                        .Select(line => line.Split(',').Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray())
                        .Select(v => new House(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]))
                        .ToList();
                    var prices = model.Predict(houses);                           // one batch per file
                    await File.WriteAllLinesAsync(Path.Combine(outbox, Path.GetFileName(file)),
                        File.ReadLines(file).Zip(prices, (line, p) => $"{line},{p:F0}"), stoppingToken);
                    File.Delete(file);
                    logger.LogInformation("Priced {Count} houses from {File}", houses.Count, Path.GetFileName(file));
                }
            }
            while (await timer.WaitForNextTickAsync(stoppingToken));
        }
    }
"""

BLAZOR = """
    @page "/price"
    @rendermode InteractiveServer
    @inject Pricing.HousePriceModel Model

    <h3>House price</h3>
    <label>Area (sq ft) <input type="number" @bind="area" /></label>
    <label>Bedrooms <input type="number" @bind="bedrooms" /></label>
    <label>Age (years) <input type="number" @bind="age" /></label>
    <label>Quality (1-10) <input type="number" @bind="quality" /></label>
    <button @onclick="Estimate">Estimate</button>
    @if (price is not null)
    {
        <p>Estimated price: <b>@price.Value.ToString("C0")</b></p>
    }

    @code {
        float area = 2100, bedrooms = 4, age = 15, quality = 7;
        float? price;

        void Estimate() =>
            price = Model.Predict(new Pricing.House(area, bedrooms, 2, age, 9.5f, quality, 2, 0, 6500));
    }

    // Program.cs, after CreateBuilder:
    builder.Services.AddSingleton(_ => Pricing.HousePriceModel.Load(builder.Configuration["ModelDirectory"] ?? "models"));
"""

TESTS = """
    using NeuralSharp;
    using Pricing;

    public sealed class HousePriceModelTests : IDisposable
    {
        private static readonly string ModelDirectory = Environment.GetEnvironmentVariable("HOUSE_MODEL_DIR") ?? "models";
        private readonly HousePriceModel _model = HousePriceModel.Load(ModelDirectory, Device.Cpu);

        [Fact]
        public void Family_house_is_priced_in_a_plausible_range()
        {
            float price = _model.Predict(new House(2100, 4, 2, 15, 9.5f, 7, 2, 0, 6500));
            Assert.InRange(price, 300_000f, 500_000f);
        }

        [Fact]
        public void Bigger_house_costs_more_all_else_equal()
        {
            float small = _model.Predict(new House(1200, 3, 2, 15, 9.5f, 7, 2, 0, 6500));
            float large = _model.Predict(new House(2400, 3, 2, 15, 9.5f, 7, 2, 0, 6500));
            Assert.True(large > small);
        }

        [Fact]
        public void Batch_and_single_predictions_agree()
        {
            var houses = new[] { new House(950, 2, 1, 45, 30, 4, 0, 0, 2900), new House(4200, 5, 4, 2, 2, 10, 3, 1, 12000) };
            float[] batch = _model.Predict(houses);
            Assert.Equal(batch[0], _model.Predict(houses[0]), 1e-3f);
            Assert.Equal(batch[1], _model.Predict(houses[1]), 1e-3f);
        }

        [Fact]
        public void Gpu_matches_cpu_when_available()
        {
            if (!Device.IsCudaAvailable) return;                     // passes trivially without a GPU
            using var gpu = HousePriceModel.Load(ModelDirectory, Device.Cuda());
            var house = new House(2100, 4, 2, 15, 9.5f, 7, 2, 0, 6500);
            Assert.Equal(_model.Predict(house), gpu.Predict(house), 1f);   // within $1
        }

        public void Dispose() => _model.Dispose();
    }
"""

WINFORMS = """
    using Pricing;

    public partial class Form1 : Form
    {
        private readonly HousePriceModel _model = HousePriceModel.Load("models");
        private readonly NumericUpDown _area = new() { Maximum = 10_000, Value = 2100, Dock = DockStyle.Top };
        private readonly NumericUpDown _quality = new() { Minimum = 1, Maximum = 10, Value = 7, Dock = DockStyle.Top };
        private readonly Label _result = new() { Dock = DockStyle.Top, Height = 40 };

        public Form1()
        {
            InitializeComponent();
            Text = $"House price ({_model.Device})";
            var button = new Button { Text = "Estimate", Dock = DockStyle.Top };
            button.Click += async (_, _) =>
            {
                var house = new House((float)_area.Value, 4, 2, 15, 9.5f, (float)_quality.Value, 2, 0, 6500);
                float price = await Task.Run(() => _model.Predict(house));        // keep the UI thread free
                _result.Text = $"Estimated price: {price:C0}";
            };
            Controls.AddRange([_result, button, _quality, _area]);
            FormClosed += (_, _) => _model.Dispose();
        }
    }
"""

WPF = """
    // MainWindow.xaml: two TextBoxes (Area, Quality), a Button (Click="Estimate_Click") and a TextBlock (Result)
    public partial class MainWindow : Window
    {
        private readonly HousePriceModel _model = HousePriceModel.Load("models");

        public MainWindow()
        {
            InitializeComponent();
            Closed += (_, _) => _model.Dispose();
        }

        private async void Estimate_Click(object sender, RoutedEventArgs e)
        {
            var house = new House(float.Parse(Area.Text), 4, 2, 15, 9.5f, float.Parse(Quality.Text), 2, 0, 6500);
            float price = await Task.Run(() => _model.Predict(house));
            Result.Text = $"Estimated price: {price:C0}";
        }
    }
"""


def build():
    return page(
        chapter_open(
            "projecttypes",
            "NeuralSharp is an ordinary .NET library: any project that can reference a class library can train or "
            "run models. This chapter packages the house-price model once as a class library and uses it from a "
            "worker service, a web API, a Blazor app, a unit-test project, a Windows Forms app and a WPF app, all "
            "built from the standard <code>dotnet new</code> templates and compiled for this book. It closes with "
            "notes on containers, cloud functions, mobile and the browser.",
            "Put the model code (factory, loading, preprocessing, prediction) in a <b>class library</b>; every application references it.",
            "Services (web, worker, Blazor): load once as a singleton or in <code>ExecuteAsync</code>; <code>Eval()</code> once.",
            "Desktop apps: predict on a background thread (<code>Task.Run</code>) to keep the UI responsive.",
            "Tests: plausibility, monotonicity, batch/single agreement and CPU/GPU parity.",
            "GPU use needs only the NVIDIA driver on the machine (or in the container); everything else is managed code.",
        ),
        h2("38.1 The pattern: one library, many front ends"),
        reftable(["Project", "Template", "Built here", "Ran here"], [
            ["Class library", "<code>dotnet new classlib</code>", "yes", "(used by all)"],
            ["Console app", "<code>dotnet new console</code>", "yes (every sample)", "yes"],
            ["Web API", "<code>dotnet new web</code>", "yes", "yes: 200 concurrent requests (" + ch("webapi") + ")"],
            ["Worker service", "<code>dotnet new worker</code>", "yes", "yes: priced a CSV from an inbox folder"],
            ["Blazor web app", "<code>dotnet new blazor --interactivity Server</code>", "yes", "yes: page served (HTTP 200)"],
            ["xUnit tests", "<code>dotnet new xunit</code>", "yes", "yes: 4 passed"],
            ["Windows Forms", "<code>dotnet new winforms</code>", "yes (EnableWindowsTargeting)", "no (Windows only)"],
            ["WPF", "<code>dotnet new wpf</code>", "yes (EnableWindowsTargeting)", "no (Windows only)"],
        ], caption="Table 38.1 — Project types covered, and what was verified for this book (Linux container)"),
        deriv("Setting up the solution", [
            "<code>dotnet new classlib -n Pricing</code>; <code>dotnet add Pricing reference path/to/NeuralSharp.csproj</code>.",
            "Create each application with its template and <code>dotnet add &lt;App&gt; reference Pricing/Pricing.csproj</code>.",
            "Copy the model folder (weights + scalers) next to each application, or configure its path "
            "(<code>ModelDirectory</code> in <code>appsettings.json</code> or an environment variable).",
            "Optionally <code>dotnet new sln</code> and <code>dotnet sln add</code> every project.",
        ]),
        h2("38.2 The class library"),
        snippet(LIBRARY, caption="Pricing/HousePriceModel.cs"),
        para("This is the house-price model of " + ch("regression") + " behind a small API: <code>CreateNetwork</code> "
             "for training code, <code>Load</code> for applications (GPU when available, CPU otherwise, unless a device "
             "is passed), and <code>Predict</code> for one house or a batch. It is safe to share between threads "
             "(" + ch("inference") + ")."),
        cpugpu("choosing the device for any application",
               """
               using var model = HousePriceModel.Load("models", Device.Cpu);        // always the CPU
               """,
               """
               using var model = HousePriceModel.Load("models", Device.Cuda());     // a specific GPU
               using var auto = HousePriceModel.Load("models");                     // GPU if present, else CPU
               """,
               "Expose the choice as a setting (" + ch("devices") + ", Section 2.3) so operators can switch without a rebuild."),
        h2("38.3 Worker service"),
        para("A worker service is a long-running background process (a Windows service, a Linux systemd unit, a "
             "container). Here it watches a folder and prices every CSV file dropped into it."),
        snippet(WORKER, caption="PriceWorker/Worker.cs (with <code>using System.Globalization;</code>)"),
        output("""
            $ cat inbox/batch1.csv
            2100,4,2,15,9.5,7,2,0,6500
            1200,3,1,30,12,5,1,0,4000
            950,2,1,45,30,4,0,0,2900
            $ dotnet PriceWorker.dll --ModelDirectory models --Inbox inbox --Outbox outbox
            $ cat outbox/batch1.csv
            2100,4,2,15,9.5,7,2,0,6500,390612
            1200,3,1,30,12,5,1,0,4000,192873
            950,2,1,45,30,4,0,0,2900,87454
            """, caption="The worker at work"),
        h2("38.4 Web API and Blazor"),
        para("The web API is the one of " + ch("webapi") + ", with <code>PricePredictor</code> replaced by the library "
             "class. A Blazor Server app runs its components on the server, so it can inject the model directly:"),
        snippet(BLAZOR, caption="PriceWeb/Components/Pages/Price.razor and the registration in Program.cs"),
        honestbox("Blazor WebAssembly and the browser",
                  "<p>In Blazor WebAssembly the code runs inside the browser: there is no CUDA driver, and threads are limited, "
                  "so only the CPU backend could run, single-threaded and slowly. This was not tested for this book. For "
                  "browser front ends, keep the model on the server (Blazor Server, or a Web API called from the page), as "
                  "the GPT sample's <code>index.html</code> does.</p>"),
        h2("38.5 Unit tests"),
        snippet(TESTS, caption="Pricing.Tests/HousePriceModelTests.cs"),
        output("""
            $ HOUSE_MODEL_DIR=/path/to/models dotnet test Pricing.Tests -c Release
            Passed!  - Failed:     0, Passed:     4, Skipped:     0, Total:     4, Duration: 57 ms - Pricing.Tests.dll (net10.0)
            """),
        para("Tests of a model check behaviour, not exact numbers: plausible ranges, directions that must hold (a bigger "
             "house is not cheaper), consistency between code paths, and parity between devices. " + ch("testing") +
             " covers testing the library itself."),
        h2("38.6 Desktop: Windows Forms and WPF"),
        snippet(WINFORMS, caption="PriceDesk/Form1.cs (Windows Forms)"),
        snippet(WPF, caption="PriceWpf/MainWindow.xaml.cs (WPF)"),
        para("Both compile on any OS with <code>&lt;EnableWindowsTargeting&gt;true&lt;/EnableWindowsTargeting&gt;</code> in the "
             "project file and run on Windows. Loading the model in a field initializer is fine for a small model; for large "
             "ones, load asynchronously after the window appears. A desktop GPU is used automatically when present."),
        honestbox(".NET MAUI (Android, iOS, Mac, Windows)",
                  "<p>NeuralSharp is plain .NET, so the CPU backend should work in MAUI apps; the CUDA backend applies only on "
                  "Windows desktops with NVIDIA GPUs. MAUI was not built or tested for this book. On phones, prefer small models "
                  "and ship the weights as app resources.</p>"),
        h2("38.7 Containers and the cloud"),
        snippet("""
            # Dockerfile for the web API (CPU)
            FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
            WORKDIR /src
            COPY . .
            RUN dotnet publish PriceApi -c Release -o /app

            FROM mcr.microsoft.com/dotnet/aspnet:10.0
            WORKDIR /app
            COPY --from=build /app .
            COPY models ./models
            ENTRYPOINT ["dotnet", "PriceApi.dll"]

            # For the GPU: run on a host with the NVIDIA driver and the NVIDIA Container Toolkit:
            #   docker run --gpus all -p 8080:8080 price-api
            # The container needs no CUDA libraries: the driver's libcuda.so.1 is mounted by the toolkit.
            """, caption="A container image (not built for this book: no Docker in the book's environment)"),
        reftable(["Host", "Notes"], [
            ["Linux or Windows VM", "Any; install the NVIDIA driver on GPU VMs"],
            ["Kubernetes", "GPU nodes with the NVIDIA device plugin; request <code>nvidia.com/gpu: 1</code>"],
            ["Azure Functions / AWS Lambda", "CPU only; keep models small and load them once per instance (static field) to limit cold starts"],
            ["Azure App Service / AWS App Runner", "CPU; as the web API above"],
        ], caption="Table 38.2 — Where the applications can run"),
        honestbox("What \"verified\" means in this chapter",
                  "<p>The class library, worker, web API, Blazor app and tests were built and run in this book's Linux container "
                  "on the CPU. The Windows Forms and WPF projects were compiled but, being Windows-only, not run. The Docker, "
                  "Kubernetes, serverless and MAUI notes describe standard .NET deployment and were not exercised.</p>"),
        practice([
            (1, "Which project should contain the model's architecture, and why?",
             "The class library: training and every application must build the identical network for the saved weights to "
             "fit, so the factory lives in one place."),
            (1, "Why does the WinForms handler call <code>Predict</code> inside <code>Task.Run</code>?",
             "So the UI thread stays free to repaint and respond; a large model or a cold first call could otherwise freeze the window."),
            (2, "Add a CSV upload to the Blazor page that prices every row.",
             "Use <code>InputFile</code>, read the stream line by line into <code>House</code> records, call "
             "<code>Model.Predict(houses)</code> once, and show the results in a table."),
            (2, "Make the worker use the GPU when the machine has one.",
             "Nothing to change: <code>HousePriceModel.Load</code> picks <code>Device.Cuda()</code> when available; set "
             "<code>NEURALSHARP_DISABLE_CUDA=1</code> to force the CPU."),
            (3, "Turn the worker into a training service that retrains the model every night on new data.",
             "Add a second loop (or a scheduled job) that loads the accumulated CSV data, trains with <code>CreateNetwork</code> "
             "and the Trainer, evaluates on recent data, writes a new versioned model folder only if it beats the current "
             "model, and signals the scoring loop to reload."),
        ], PART),
        footer("Class library", "Worker service", "Blazor", "Unit test", "Windows Forms", "WPF", "Container", "Dependency injection"),
    )

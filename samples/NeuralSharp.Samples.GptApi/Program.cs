// NeuralSharp GPT Web API: serves the character-level transformer from the Transformer sample.
//
//   dotnet run -c Release --project samples/NeuralSharp.Samples.GptApi
//   then open http://localhost:5080          the inference UI (index.html)
//             http://localhost:5080/scalar   interactive API reference (Scalar)
//
// Model location: Gpt:ModelPath in appsettings.json (or --Gpt:ModelPath <path>). Point it at a model saved by
// the Transformer sample, or leave the default: when no model exists, one is trained in the background
// and /api/status reports the progress.

using System.Text.Json.Serialization;
using Microsoft.AspNetCore.Http.HttpResults;
using NeuralSharp.Samples.Gpt;
using NeuralSharp.Samples.GptApi;
using Scalar.AspNetCore;

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddOpenApi(options => options.AddDocumentTransformer((document, _, _) =>
{
    document.Info.Title = "NeuralSharp GPT API";
    document.Info.Description = "Text generation with a character-level transformer trained and served by NeuralSharp (pure C#, CPU or CUDA).";
    return Task.CompletedTask;
}));
builder.Services.ConfigureHttpJsonOptions(options => options.SerializerOptions.Converters.Add(new JsonStringEnumConverter()));
builder.Services.AddSingleton<GptService>();
builder.Services.AddHostedService(services => services.GetRequiredService<GptService>());

var app = builder.Build();
app.UseDefaultFiles();
app.UseStaticFiles();
app.MapOpenApi();
app.MapScalarApiReference(options => options
    .WithTitle("NeuralSharp GPT API")
    .WithTheme(ScalarTheme.DeepSpace));

var api = app.MapGroup("/api").WithTags("GPT");

api.MapGet("/status", (GptService gpt) => gpt.Status)
    .WithName("GetStatus")
    .WithSummary("Service state")
    .WithDescription("Loading, Training (with live progress), Ready or Failed.");

api.MapGet("/model", Results<Ok<ModelInfo>, ProblemHttpResult> (GptService gpt) =>
        gpt.Status.Status == ModelStatus.Ready ? TypedResults.Ok(gpt.Describe()) : NotReady(gpt))
    .WithName("GetModel")
    .WithSummary("Model architecture and training record")
    .WithDescription("Parameter count, layers, heads, width, context length, vocabulary, validation results, weights file and a layer summary.");

api.MapGet("/devices", (GptService gpt) => gpt.Devices())
    .WithName("GetDevices")
    .WithSummary("Available compute devices")
    .WithDescription("The CPU and every CUDA GPU, with memory usage and which one the model is on.");

api.MapPost("/generate", async Task<Results<Ok<GenerationResult>, ProblemHttpResult>> (GenerateRequest request, GptService gpt, CancellationToken cancellationToken) =>
    {
        if (gpt.Status.Status != ModelStatus.Ready)
        {
            return NotReady(gpt);
        }

        try
        {
            return TypedResults.Ok(await gpt.GenerateAsync(request, cancellationToken));
        }
        catch (ArgumentException ex)
        {
            return TypedResults.Problem(ex.Message, statusCode: StatusCodes.Status400BadRequest);
        }
    })
    .WithName("Generate")
    .WithSummary("Generate text")
    .WithDescription("Continues the prompt and returns the text, every generated character with its probability, entropy and top alternatives, and aggregate metrics (latency, throughput, confidence, perplexity, memory). Set device to switch between cpu and cuda.");

api.MapPost("/generate/stream", Results<ServerSentEventsResult<object>, ProblemHttpResult> (GenerateRequest request, GptService gpt, CancellationToken cancellationToken) =>
        gpt.Status.Status == ModelStatus.Ready ? TypedResults.ServerSentEvents(gpt.StreamAsync(request, cancellationToken)) : NotReady(gpt))
    .WithName("GenerateStream")
    .WithSummary("Generate text as a live stream (server-sent events)")
    .WithDescription("Same as /api/generate, but emits a \"token\" event for each character as soon as it is produced, then a \"metrics\" event.");

app.Run();

static ProblemHttpResult NotReady(GptService gpt) =>
    TypedResults.Problem(gpt.Status.Message, statusCode: StatusCodes.Status503ServiceUnavailable, title: $"Model is {gpt.Status.Status}");

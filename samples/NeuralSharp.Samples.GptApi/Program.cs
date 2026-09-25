// NeuralSharp GPT Web API: serves the character-level transformer from the Transformer sample.
//
//   dotnet run -c Release --project samples/NeuralSharp.Samples.GptApi
//   then open http://localhost:5080          the inference UI (index.html)
//             http://localhost:5080/scalar   interactive API reference (Scalar)
//
// Model location: Gpt:ModelPath in appsettings.json (or --Gpt:ModelPath <path>). Point it at a model saved by
// the Transformer sample, or leave the default: when no model exists, one is trained in the background
// and /api/status reports the progress.

using System.Text.Json;
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
builder.Services.AddSingleton<ChatService>();
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
    .WithDescription("Continues the prompt and returns one or more samples, every generated character with its probability, entropy and top alternatives, and aggregate metrics (latency, throughput, confidence, perplexity, memory, decoding mode). Set device to switch between cpu and cuda; samples to generate a batch; useCache/useGraph to compare decoding modes.");

api.MapPost("/generate/stream", Results<ServerSentEventsResult<object>, ProblemHttpResult> (GenerateRequest request, GptService gpt, CancellationToken cancellationToken) =>
        gpt.Status.Status == ModelStatus.Ready ? TypedResults.ServerSentEvents(gpt.StreamAsync(request, cancellationToken)) : NotReady(gpt))
    .WithName("GenerateStream")
    .WithSummary("Generate text as a live stream (server-sent events)")
    .WithDescription("Same as /api/generate, but emits \"token\" events (each with its sample index) every chunkSize characters as they are produced, then a \"metrics\" event.");

// ---------------------------------------------------------------- Ollama-compatible endpoints
var ollama = app.MapGroup("/api").WithTags("Ollama-compatible");
var ndjson = new JsonSerializerOptions(JsonSerializerDefaults.Web);

ollama.MapPost("/chat", (OllamaChatRequest request, ChatService chat, CancellationToken cancellationToken) =>
    {
        try
        {
            ChatService.Translate(request);                                // validate before anything is streamed
        }
        catch (Exception ex) when (ex is ArgumentException or InvalidOperationException or FormatException)
        {
            return Results.Json(new { error = ex.Message }, statusCode: StatusCodes.Status400BadRequest);
        }

        if (!chat.IsAvailable)
        {
            return Results.Json(new { error = $"model '{request.Model}' is not available yet: no trained model file" }, statusCode: StatusCodes.Status404NotFound);
        }

        if (request.Stream == false)
        {
            return Results.Stream(async body =>
            {
                OllamaChatResponse? last = null;
                await foreach (var item in chat.ChatAsync(request, stream: false, cancellationToken))
                {
                    last = item;
                }

                await JsonSerializer.SerializeAsync(body, last, ndjson, cancellationToken);
            }, "application/json");
        }

        return Results.Stream(async body =>
        {
            await foreach (var item in chat.ChatAsync(request, stream: true, cancellationToken))
            {
                await JsonSerializer.SerializeAsync(body, item, ndjson, cancellationToken);
                await body.WriteAsync("\n"u8.ToArray(), cancellationToken);
                await body.FlushAsync(cancellationToken);
            }
        }, "application/x-ndjson");
    })
    .WithName("OllamaChat")
    .WithSummary("Chat (Ollama-compatible)")
    .WithDescription("Accepts the Ollama /api/chat body: model (any name selects the served model), messages (system, user, assistant, tool), " +
                     "stream (NDJSON lines, default true), think (true/false or low/medium/high), keep_alive (e.g. \"30m\", 0, -1), options " +
                     "(temperature, top_k, top_p, min_p, repeat_penalty, repeat_last_n, presence_penalty, frequency_penalty, seed, num_ctx, " +
                     "num_predict, stop; others ignored) and tools (function definitions; calls come back in message.tool_calls).");

ollama.MapGet("/tags", (ChatService chat) => Results.Ok(new { models = chat.Tags() }))
    .WithName("OllamaTags").WithSummary("Available models (Ollama-compatible)");
ollama.MapGet("/ps", (ChatService chat) => Results.Ok(new { models = chat.Running() }))
    .WithName("OllamaPs").WithSummary("Loaded models and when they expire (Ollama-compatible)");
ollama.MapGet("/version", () => Results.Ok(new { version = "0.1.0-neuralsharp" }))
    .WithName("OllamaVersion").WithSummary("Server version (Ollama-compatible)");

app.Run();

static ProblemHttpResult NotReady(GptService gpt) =>
    TypedResults.Problem(gpt.Status.Message, statusCode: StatusCodes.Status503ServiceUnavailable, title: $"Model is {gpt.Status.Status}");

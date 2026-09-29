using System.Diagnostics;
using System.Text.Encodings.Web;
using System.Text.Json;
using Microsoft.AspNetCore.Http.Json;
using Microsoft.Extensions.Diagnostics.HealthChecks;
using Microsoft.Extensions.Options;
using Qasd;
using Qasd.Api;
using Scalar.AspNetCore;

// The intent service: serves the classifier from 'qasd train' and, when there is one, the chat model tuned on the intents
// with idrak-tune (Idrak's fine-tuning tool). Test page at /, API reference (Scalar) at /scalar.
var builder = WebApplication.CreateBuilder(args);
builder.Services.Configure<IntentModelOptions>(builder.Configuration.GetSection("IntentModel"));
builder.Services.AddSingleton<ModelHost>();
builder.Services.AddSingleton<TunedHost>();
builder.Services.AddOpenApi(options => options.AddDocumentTransformer((document, _, _) =>
{
    document.Info.Title = "Qasd: intent classifier";
    document.Info.Description = "Classifies a user's message into an intent (retrieve, function_call, direct_reply, identity, … as trained), with "
                                + "the fast classifier or the tuned language model, as one JSON response or streamed as Server-Sent Events (\"stream\": true).";
    return Task.CompletedTask;
}));
builder.Services.AddProblemDetails();
builder.Services.AddHealthChecks().AddCheck<ModelHealthCheck>("model");
builder.Services.ConfigureHttpJsonOptions(o => o.SerializerOptions.Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping);   // Arabic as written

var app = builder.Build();
app.UseExceptionHandler();
app.Services.GetRequiredService<ModelHost>();                                   // load the models now: fail at startup, not on the first request
app.Services.GetRequiredService<TunedHost>();
app.MapOpenApi();
app.MapScalarApiReference(options => options.WithTitle("Qasd: intent classifier"));
app.UseDefaultFiles();                                                           // / → wwwroot/index.html, the test page
app.UseStaticFiles();
app.MapHealthChecks("/health");

var api = app.MapGroup("/v1").WithTags("Intents");

api.MapPost("/classify", (ClassifyRequest request, HttpContext http, ModelHost host, TunedHost tuned, IOptions<IntentModelOptions> options,
        IOptions<JsonOptions> json) => Classify([request.Text], request.Model, request.Stream, single: true, http, host, tuned, options.Value, json.Value))
    .WithName("Classify")
    .WithSummary("Classify one message")
    .WithDescription("The most likely intent, its confidence, whether it reaches MinConfidence, and every intent's probability. "
                     + "\"model\": \"classifier\" or \"tuned\". \"stream\": true answers with Server-Sent Events: \"token\" (the tuned model's answer as it "
                     + "is generated), \"result\" and \"done\".")
    .Produces<ClassifyResponse>(StatusCodes.Status200OK, "application/json", "text/event-stream")
    .ProducesValidationProblem();

api.MapPost("/classify/batch", (ClassifyBatchRequest request, HttpContext http, ModelHost host, TunedHost tuned, IOptions<IntentModelOptions> options,
        IOptions<JsonOptions> json) => Classify(request.Texts, request.Model, request.Stream, single: false, http, host, tuned, options.Value, json.Value))
    .WithName("ClassifyBatch")
    .WithSummary("Classify several messages")
    .WithDescription("Results in the order of the request. \"stream\": true sends a \"result\" event per message as it is ready (with the tuned model, "
                     + "its \"token\" events first) and a final \"done\" event.")
    .Produces<ClassifyBatchResponse>(StatusCodes.Status200OK, "application/json", "text/event-stream")
    .ProducesValidationProblem();

api.MapGet("/models", (ModelHost host, TunedHost tuned, IOptions<IntentModelOptions> options) =>
    {
        var info = host.Info;
        string defaultModel = options.Value.DefaultModel;
        return TypedResults.Ok(new ModelsResponse(
        [
            new ModelDescription("classifier", true, defaultModel != "tuned", "Hashed n-gram features and a small network: about 0.1 ms per message on a CPU.",
                info.Labels, info.Device, info.Path, false),
            new ModelDescription("tuned", tuned.Classifier is not null, defaultModel == "tuned",
                tuned.Classifier is { } t ? $"{t.BaseModel} tuned with LoRA to answer with the intent; streams its answer." : "A pretrained chat model tuned with LoRA (idrak-tune train, see the README).",
                tuned.Classifier?.Labels ?? [], tuned.Classifier?.Device.Name, tuned.Status, true),
        ], options.Value.MinConfidence));
    })
    .WithName("GetModels")
    .WithSummary("The models the service can use");

api.MapGet("/model", (ModelHost host) => TypedResults.Ok(host.Info))
    .WithName("GetModel")
    .WithSummary("The classifier model being served");

api.MapPost("/model/reload", (HttpRequest http, ModelHost host, IOptions<IntentModelOptions> options, ILogger<ModelHost> logger) =>
    {
        if (string.IsNullOrEmpty(options.Value.AdminKey) || http.Headers["X-Admin-Key"] != options.Value.AdminKey)
        {
            return Results.Unauthorized();
        }

        try
        {
            var info = host.Reload();
            logger.LogInformation("Reloaded the intent model {Path}", info.Path);
            return Results.Ok(info);
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or InvalidOperationException)
        {
            return Results.Problem($"The model file could not be loaded; the previous model keeps serving. {ex.Message}", statusCode: StatusCodes.Status500InternalServerError);
        }
    })
    .WithName("ReloadModel")
    .WithSummary("Load the classifier's model file again")
    .WithDescription("After retraining: reads IntentModel:Path again and serves it, without dropping requests in flight. Needs the X-Admin-Key header (IntentModel:AdminKey).")
    .Produces<ModelInfo>()
    .Produces(StatusCodes.Status401Unauthorized)
    .ProducesProblem(StatusCodes.Status500InternalServerError);

app.Run();

// Validates the request, then answers with one JSON response or, with stream, Server-Sent Events.
static async Task<IResult> Classify(IReadOnlyList<string>? texts, string? requestedModel, bool stream, bool single, HttpContext http, ModelHost host,
    TunedHost tuned, IntentModelOptions options, JsonOptions json)
{
    string model = (requestedModel ?? options.DefaultModel).Trim().ToLowerInvariant();
    var errors = new Dictionary<string, string[]>();
    if (model is not ("classifier" or "tuned"))
    {
        errors["model"] = ["Use \"classifier\" or \"tuned\"."];
    }
    else if (model == "tuned" && tuned.Classifier is null)
    {
        errors["model"] = [$"The tuned model is {tuned.Status}."];
    }

    int maxBatch = model == "tuned" ? options.MaxTunedBatch : options.MaxBatch;
    if (texts is null || texts.Count == 0)
    {
        errors[single ? "text" : "texts"] = [single ? "Send a message." : "Send at least one message."];
    }
    else if (texts.Count > maxBatch)
    {
        errors["texts"] = [$"At most {maxBatch} messages per request to the {model} model."];
    }
    else if (texts.Any(string.IsNullOrWhiteSpace))
    {
        errors["text"] = ["Messages must not be empty."];
    }
    else if (texts.Any(t => t.Length > options.MaxTextLength))
    {
        errors["text"] = [$"Messages must be at most {options.MaxTextLength} characters."];
    }

    if (errors.Count > 0)
    {
        return TypedResults.ValidationProblem(errors);
    }

    if (stream)
    {
        await StreamEvents(texts!, model, http, host, tuned, options, json.SerializerOptions);
        return Results.Empty;
    }

    var clock = Stopwatch.StartNew();
    IReadOnlyList<TextPrediction> predictions;
    if (model == "tuned")
    {
        predictions = tuned.Classifier!.Predict(texts!);
    }
    else
    {
        using var lease = host.Lease();
        predictions = lease.Classifier.Predict(texts!);
    }

    double elapsed = clock.Elapsed.TotalMilliseconds;
    var results = texts!.Zip(predictions, (text, p) => Response(text, p, model, options)).ToList();
    return single ? TypedResults.Ok(results[0]) : TypedResults.Ok(new ClassifyBatchResponse(results, elapsed, model));
}

// text/event-stream: "token" (tuned: the generated answer), "result" per message, "done"; "error" if something fails midway.
static async Task StreamEvents(IReadOnlyList<string> texts, string model, HttpContext http, ModelHost host, TunedHost tuned, IntentModelOptions options,
    JsonSerializerOptions json)
{
    var response = http.Response;
    response.ContentType = "text/event-stream; charset=utf-8";
    response.Headers.CacheControl = "no-cache";
    response.Headers["X-Accel-Buffering"] = "no";                              // no buffering in nginx and similar proxies
    var cancel = http.RequestAborted;
    async Task Send<T>(string name, T data)
    {
        await response.WriteAsync($"event: {name}\ndata: {JsonSerializer.Serialize(data, json)}\n\n", cancel);
        await response.Body.FlushAsync(cancel);
    }

    var clock = Stopwatch.StartNew();
    double firstToken = -1;
    try
    {
        if (model == "tuned")
        {
            // The answers of every message generated together (one pass through the model per token), then the results.
            await foreach (var item in tuned.Classifier!.StreamAsync(texts, cancel))
            {
                firstToken = firstToken < 0 ? clock.Elapsed.TotalMilliseconds : firstToken;
                if (item.Token is { } token)
                {
                    await Send("token", new TokenEvent(item.Index, token));
                }
                else
                {
                    await Send("result", new ResultEvent(item.Index, Response(texts[item.Index], item.Prediction!, model, options)));
                }
            }
        }
        else
        {
            for (int i = 0; i < texts.Count; i++)
            {
                TextPrediction prediction;
                using (var lease = host.Lease())
                {
                    prediction = lease.Classifier.Predict(texts[i]);
                }

                firstToken = firstToken < 0 ? clock.Elapsed.TotalMilliseconds : firstToken;
                await Send("result", new ResultEvent(i, Response(texts[i], prediction, model, options)));
            }
        }

        await Send("done", new DoneEvent(texts.Count, clock.Elapsed.TotalMilliseconds, firstToken, model));
    }
    catch (OperationCanceledException) when (cancel.IsCancellationRequested)
    {
        // The client went away.
    }
    catch (Exception ex) when (ex is InvalidOperationException or IOException)
    {
        await Send("error", new { message = ex.Message });
    }
}

static ClassifyResponse Response(string text, TextPrediction prediction, string model, IntentModelOptions options) =>
    new(text, prediction.Label, prediction.Confidence, prediction.Confidence >= options.MinConfidence,
        [.. prediction.Probabilities.Select(p => new IntentScore(p.Label, p.Probability))], model);

/// <summary>Healthy while the classifier is loaded and answers (the tuned model is optional).</summary>
internal sealed class ModelHealthCheck(ModelHost host, TunedHost tuned) : IHealthCheck
{
    public Task<HealthCheckResult> CheckHealthAsync(HealthCheckContext context, CancellationToken cancellationToken = default)
    {
        try
        {
            var info = host.Info;
            return Task.FromResult(HealthCheckResult.Healthy($"classifier: {info.Labels.Count} intents on {info.Device}; tuned: "
                                                             + (tuned.Classifier is { } t ? $"{t.BaseModel} on {t.Device.Name}" : "not configured")));
        }
        catch (Exception ex)
        {
            return Task.FromResult(HealthCheckResult.Unhealthy("The classifier is not available.", ex));
        }
    }
}

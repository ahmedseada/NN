using System.Text.Encodings.Web;
using IntentClassifier;
using IntentClassifier.Api;
using Microsoft.AspNetCore.Http.HttpResults;
using Microsoft.Extensions.Diagnostics.HealthChecks;
using Microsoft.Extensions.Options;
using Scalar.AspNetCore;

// The intent service: serves a model trained with 'intent-classifier train'. API reference (Scalar) at /scalar.
var builder = WebApplication.CreateBuilder(args);
builder.Services.Configure<IntentModelOptions>(builder.Configuration.GetSection("IntentModel"));
builder.Services.AddSingleton<ModelHost>();
builder.Services.AddOpenApi(options => options.AddDocumentTransformer((document, _, _) =>
{
    document.Info.Title = "Intent classifier";
    document.Info.Description = "Classifies a user's message into an intent (retrieve, function_call, direct_reply, identity, … as trained).";
    return Task.CompletedTask;
}));
builder.Services.AddProblemDetails();
builder.Services.AddHealthChecks().AddCheck<ModelHealthCheck>("model");
builder.Services.ConfigureHttpJsonOptions(o => o.SerializerOptions.Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping);   // Arabic as written

var app = builder.Build();
app.UseExceptionHandler();
app.Services.GetRequiredService<ModelHost>();                                   // load the model now: fail at startup, not on the first request
app.MapOpenApi();
app.MapScalarApiReference(options => options.WithTitle("Intent classifier"));
app.MapGet("/", () => Results.Redirect("/scalar")).ExcludeFromDescription();
app.MapHealthChecks("/health");

var api = app.MapGroup("/v1").WithTags("Intents");

api.MapPost("/classify", Results<Ok<ClassifyResponse>, ValidationProblem> (ClassifyRequest request, ModelHost host, IOptions<IntentModelOptions> options) =>
    {
        if (Check([request.Text], options.Value) is { } problem)
        {
            return problem;
        }

        using var lease = host.Lease();
        return TypedResults.Ok(Response(request.Text, lease.Classifier.Predict(request.Text), options.Value));
    })
    .WithName("Classify")
    .WithSummary("Classify one message")
    .WithDescription("The most likely intent, its confidence, whether it reaches MinConfidence, and every intent's probability.");

api.MapPost("/classify/batch", Results<Ok<ClassifyBatchResponse>, ValidationProblem> (ClassifyBatchRequest request, ModelHost host, IOptions<IntentModelOptions> options) =>
    {
        if (Check(request.Texts, options.Value) is { } problem)
        {
            return problem;
        }

        using var lease = host.Lease();
        var predictions = lease.Classifier.Predict(request.Texts);
        return TypedResults.Ok(new ClassifyBatchResponse([.. request.Texts.Zip(predictions, (text, p) => Response(text, p, options.Value))]));
    })
    .WithName("ClassifyBatch")
    .WithSummary("Classify several messages")
    .WithDescription("One pass through the model for all of them; results in the order of the request.");

api.MapGet("/model", (ModelHost host) => TypedResults.Ok(host.Info))
    .WithName("GetModel")
    .WithSummary("The model being served");

api.MapPost("/model/reload", Results<Ok<ModelInfo>, UnauthorizedHttpResult, ProblemHttpResult> (HttpRequest http, ModelHost host, IOptions<IntentModelOptions> options, ILogger<ModelHost> logger) =>
    {
        if (string.IsNullOrEmpty(options.Value.AdminKey) || http.Headers["X-Admin-Key"] != options.Value.AdminKey)
        {
            return TypedResults.Unauthorized();
        }

        try
        {
            var info = host.Reload();
            logger.LogInformation("Reloaded the intent model {Path}", info.Path);
            return TypedResults.Ok(info);
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or InvalidOperationException)
        {
            return TypedResults.Problem($"The model file could not be loaded; the previous model keeps serving. {ex.Message}", statusCode: StatusCodes.Status500InternalServerError);
        }
    })
    .WithName("ReloadModel")
    .WithSummary("Load the model file again")
    .WithDescription("After retraining: reads IntentModel:Path again and serves it, without dropping requests in flight. Needs the X-Admin-Key header (IntentModel:AdminKey).");

app.Run();

static ValidationProblem? Check(IReadOnlyList<string>? texts, IntentModelOptions options)
{
    var errors = new Dictionary<string, string[]>();
    if (texts is null || texts.Count == 0)
    {
        errors["texts"] = ["Send at least one message."];
    }
    else if (texts.Count > options.MaxBatch)
    {
        errors["texts"] = [$"At most {options.MaxBatch} messages per request."];
    }
    else if (texts.Any(string.IsNullOrWhiteSpace))
    {
        errors["text"] = ["Messages must not be empty."];
    }
    else if (texts.Any(t => t.Length > options.MaxTextLength))
    {
        errors["text"] = [$"Messages must be at most {options.MaxTextLength} characters."];
    }

    return errors.Count > 0 ? TypedResults.ValidationProblem(errors) : null;
}

static ClassifyResponse Response(string text, TextPrediction prediction, IntentModelOptions options) =>
    new(text, prediction.Label, prediction.Confidence, prediction.Confidence >= options.MinConfidence,
        [.. prediction.Probabilities.Select(p => new IntentScore(p.Label, p.Probability))]);

/// <summary>Healthy while a model is loaded and answers.</summary>
internal sealed class ModelHealthCheck(ModelHost host) : IHealthCheck
{
    public Task<HealthCheckResult> CheckHealthAsync(HealthCheckContext context, CancellationToken cancellationToken = default)
    {
        try
        {
            var info = host.Info;
            return Task.FromResult(HealthCheckResult.Healthy($"{info.Labels.Count} intents on {info.Device}"));
        }
        catch (Exception ex)
        {
            return Task.FromResult(HealthCheckResult.Unhealthy("The model is not available.", ex));
        }
    }
}

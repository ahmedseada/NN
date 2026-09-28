namespace Qasd.Api;

/// <summary>One message to classify.</summary>
/// <param name="Text">The user's message.</param>
/// <param name="Model">"classifier" (the fast n-gram model) or "tuned" (the tuned language model); default IntentModel:DefaultModel.</param>
/// <param name="Stream">
/// false (default): one JSON response. true: Server-Sent Events as the work progresses: "token" events with the tuned
/// model's answer as it is generated, a "result" event per message and a final "done" event.
/// </param>
public sealed record ClassifyRequest(string Text, string? Model = null, bool Stream = false);

/// <summary>Several messages to classify in one call.</summary>
/// <param name="Texts">The messages.</param>
/// <param name="Model">"classifier" or "tuned"; default IntentModel:DefaultModel.</param>
/// <param name="Stream">false: one JSON response with every result; true: Server-Sent Events, a "result" event per message as it is ready.</param>
public sealed record ClassifyBatchRequest(IReadOnlyList<string> Texts, string? Model = null, bool Stream = false);

/// <summary>An intent and its probability.</summary>
/// <param name="Label">The intent.</param>
/// <param name="Probability">From 0 to 1.</param>
public sealed record IntentScore(string Label, float Probability);

/// <summary>The classification of one message.</summary>
/// <param name="Text">The message.</param>
/// <param name="Label">The most likely intent.</param>
/// <param name="Confidence">Its probability, from 0 to 1.</param>
/// <param name="Accepted">Whether the confidence reaches the service's MinConfidence (below it, send the message to a fallback).</param>
/// <param name="Probabilities">Every intent, most likely first.</param>
/// <param name="Model">The model that classified it.</param>
public sealed record ClassifyResponse(string Text, string Label, float Confidence, bool Accepted, IReadOnlyList<IntentScore> Probabilities, string Model);

/// <summary>The classifications of a batch, in the order of the request.</summary>
/// <param name="Results">One per message.</param>
/// <param name="ElapsedMilliseconds">Time the service spent classifying them (without the network).</param>
/// <param name="Model">The model that classified them.</param>
public sealed record ClassifyBatchResponse(IReadOnlyList<ClassifyResponse> Results, double ElapsedMilliseconds, string Model);

/// <summary>Stream event "token": a piece of the tuned model's answer to message Index, as it is generated.</summary>
public sealed record TokenEvent(int Index, string Token);

/// <summary>Stream event "result": the classification of message Index.</summary>
public sealed record ResultEvent(int Index, ClassifyResponse Result);

/// <summary>Stream event "done": all messages classified.</summary>
/// <param name="Count">Messages classified.</param>
/// <param name="ElapsedMilliseconds">Time the service spent on them.</param>
/// <param name="FirstTokenMilliseconds">Time to the first token (tuned model), else to the first result.</param>
/// <param name="Model">The model used.</param>
public sealed record DoneEvent(int Count, double ElapsedMilliseconds, double FirstTokenMilliseconds, string Model);

/// <summary>The classifier model being served.</summary>
/// <param name="Path">The model file.</param>
/// <param name="Labels">The intents it knows.</param>
/// <param name="Device">Where it runs.</param>
/// <param name="LoadedAt">When it was loaded (UTC).</param>
/// <param name="MinConfidence">Predictions below this confidence are returned with Accepted = false.</param>
public sealed record ModelInfo(string Path, IReadOnlyList<string> Labels, string Device, DateTimeOffset LoadedAt, float MinConfidence);

/// <summary>A model the service can use.</summary>
/// <param name="Name">"classifier" or "tuned": the value of the requests' Model field.</param>
/// <param name="Available">Whether it is loaded.</param>
/// <param name="Default">Whether requests without a Model use it.</param>
/// <param name="Description">What it is.</param>
/// <param name="Labels">Its intents (empty when not available).</param>
/// <param name="Device">Where it runs.</param>
/// <param name="Source">Its file or folder, or why it is not available.</param>
/// <param name="Streams">Whether it streams generated tokens (the tuned model); both stream results.</param>
public sealed record ModelDescription(string Name, bool Available, bool Default, string Description, IReadOnlyList<string> Labels, string? Device,
    string? Source, bool Streams);

/// <summary>The models and the service settings.</summary>
/// <param name="Models">The classifier and the tuned model.</param>
/// <param name="MinConfidence">Predictions below this confidence are returned with Accepted = false.</param>
public sealed record ModelsResponse(IReadOnlyList<ModelDescription> Models, float MinConfidence);

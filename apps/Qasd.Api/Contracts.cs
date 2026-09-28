namespace Qasd.Api;

/// <summary>One message to classify.</summary>
/// <param name="Text">The user's message.</param>
public sealed record ClassifyRequest(string Text);

/// <summary>Several messages to classify in one call (faster than one call each).</summary>
/// <param name="Texts">The messages.</param>
public sealed record ClassifyBatchRequest(IReadOnlyList<string> Texts);

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
public sealed record ClassifyResponse(string Text, string Label, float Confidence, bool Accepted, IReadOnlyList<IntentScore> Probabilities);

/// <summary>The classifications of a batch, in the order of the request.</summary>
/// <param name="Results">One per message.</param>
/// <param name="ElapsedMilliseconds">Time the service spent classifying them (without the network).</param>
public sealed record ClassifyBatchResponse(IReadOnlyList<ClassifyResponse> Results, double ElapsedMilliseconds);

/// <summary>The model being served.</summary>
/// <param name="Path">The model file.</param>
/// <param name="Labels">The intents it knows.</param>
/// <param name="Device">Where it runs.</param>
/// <param name="LoadedAt">When it was loaded (UTC).</param>
/// <param name="MinConfidence">Predictions below this confidence are returned with Accepted = false.</param>
public sealed record ModelInfo(string Path, IReadOnlyList<string> Labels, string Device, DateTimeOffset LoadedAt, float MinConfidence);

using Idrak.Datasets;

namespace Qasd;

/// <summary>
/// Qasd's rule for when two messages are the same text: lower case; Arabic without diacritics and tatweel, with one form
/// of alef, yaa and taa marbuta; Arabic-Indic digits as ASCII; punctuation as spaces; one space between words. The same
/// rule the features use, handed to the library (<see cref="Dataset.Normalize(ITextNormalizer, string, string)"/>) for
/// the deduplication and the split key.
/// </summary>
public sealed class ArabicTextNormalizer : ITextNormalizer
{
    /// <summary>The one instance (the rule has no settings).</summary>
    public static ArabicTextNormalizer Instance { get; } = new();

    /// <inheritdoc />
    public string Normalize(string text) => TextFeatures.Normalize(text);
}

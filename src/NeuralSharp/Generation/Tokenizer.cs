namespace NeuralSharp.Generation;

/// <summary>Turns text into token ids and back.</summary>
public interface ITokenizer
{
    /// <summary>Number of distinct token ids (the model's output width).</summary>
    int VocabularySize { get; }

    /// <summary>Token ids for <paramref name="text"/>.</summary>
    IReadOnlyList<int> Encode(string text);

    /// <summary>The text of a sequence of token ids.</summary>
    string Decode(IEnumerable<int> ids);
}

/// <summary>
/// One token per character of a fixed alphabet (the vocabulary string). Characters outside the alphabet are
/// dropped when encoding, or replaced by <see cref="UnknownCharacter"/> when it is part of the alphabet.
/// </summary>
public sealed class CharTokenizer : ITokenizer
{
    private readonly Dictionary<char, int> _index;

    /// <summary>Creates the tokenizer for the characters of <paramref name="vocabulary"/> (id = position).</summary>
    public CharTokenizer(string vocabulary, char? unknownCharacter = null)
    {
        Vocabulary = vocabulary;
        _index = vocabulary.Select((c, i) => (c, i)).ToDictionary(p => p.c, p => p.i);
        UnknownCharacter = unknownCharacter is { } u && _index.ContainsKey(u) ? u : null;
    }

    /// <summary>The alphabet; character i has id i.</summary>
    public string Vocabulary { get; }

    /// <summary>Stand-in for characters outside the alphabet, or null to drop them.</summary>
    public char? UnknownCharacter { get; }

    /// <inheritdoc />
    public int VocabularySize => Vocabulary.Length;

    /// <summary>Whether every character of <paramref name="text"/> is in the alphabet.</summary>
    public bool Covers(string text) => text.All(_index.ContainsKey);

    /// <inheritdoc />
    public IReadOnlyList<int> Encode(string text)
    {
        var ids = new List<int>(text.Length);
        foreach (char c in text)
        {
            if (_index.TryGetValue(c, out int id))
            {
                ids.Add(id);
            }
            else if (UnknownCharacter is { } u)
            {
                ids.Add(_index[u]);
            }
        }

        return ids;
    }

    /// <inheritdoc />
    public string Decode(IEnumerable<int> ids) => string.Concat(ids.Select(i => (uint)i < (uint)Vocabulary.Length ? Vocabulary[i] : '�'));
}

/// <summary>
/// One token per word, number or punctuation mark of a fixed vocabulary. Text is split into special tokens in angle
/// brackets (such as <c>&lt;sum&gt;</c>), runs of letters and digits, and single punctuation marks; words outside the
/// vocabulary become <see cref="UnknownToken"/>. Decoding puts a space before every token except closing punctuation,
/// so decoding a sequence piece by piece gives the same text as decoding it at once (streaming-safe).
/// </summary>
public sealed partial class WordTokenizer : ITokenizer
{
    private readonly Dictionary<string, int> _index;

    /// <summary>Creates the tokenizer for <paramref name="vocabulary"/> (id = position); the unknown token is added if missing.</summary>
    public WordTokenizer(IEnumerable<string> vocabulary, string unknownToken = "<unk>", bool lowercase = true)
    {
        var words = vocabulary.Distinct().ToList();
        if (!words.Contains(unknownToken))
        {
            words.Add(unknownToken);
        }

        Vocabulary = words;
        _index = words.Select((w, i) => (w, i)).ToDictionary(p => p.w, p => p.i);
        UnknownToken = unknownToken;
        Lowercase = lowercase;
    }

    /// <summary>Builds a vocabulary from sample texts: <paramref name="specials"/> first, then words by frequency (at least <paramref name="minCount"/> uses).</summary>
    public static WordTokenizer FromTexts(IEnumerable<string> texts, IEnumerable<string> specials, int minCount = 1, bool lowercase = true)
    {
        var counts = new Dictionary<string, int>();
        foreach (var text in texts)
        {
            foreach (var word in Split(text, lowercase))
            {
                counts[word] = counts.GetValueOrDefault(word) + 1;
            }
        }

        var first = specials.ToList();
        return new WordTokenizer([.. first, .. counts.Where(p => p.Value >= minCount && !first.Contains(p.Key))
            .OrderByDescending(p => p.Value).ThenBy(p => p.Key, StringComparer.Ordinal).Select(p => p.Key)], lowercase: lowercase);
    }

    /// <summary>The words; word i has id i.</summary>
    public IReadOnlyList<string> Vocabulary { get; }

    /// <summary>The stand-in for words outside the vocabulary.</summary>
    public string UnknownToken { get; }

    /// <summary>Whether text is lower-cased before lookup.</summary>
    public bool Lowercase { get; }

    /// <inheritdoc />
    public int VocabularySize => Vocabulary.Count;

    /// <summary>The id of a word (or of the unknown token).</summary>
    public int this[string word] => _index.TryGetValue(word, out int id) ? id : _index[UnknownToken];

    /// <summary>Splits text into the tokenizer's units without looking them up.</summary>
    public static IEnumerable<string> Split(string text, bool lowercase = true) =>
        Pattern().Matches(lowercase ? text.ToLowerInvariant() : text).Select(m => m.Value);

    /// <inheritdoc />
    public IReadOnlyList<int> Encode(string text) => [.. Split(text, Lowercase).Select(w => this[w])];

    /// <inheritdoc />
    public string Decode(IEnumerable<int> ids)
    {
        var text = new System.Text.StringBuilder();
        foreach (int id in ids)
        {
            string word = (uint)id < (uint)Vocabulary.Count ? Vocabulary[id] : UnknownToken;
            if (!(word.Length == 1 && ".,!?;:%)'".Contains(word[0])))
            {
                text.Append(' ');
            }

            text.Append(word);
        }

        return text.ToString();
    }

    [System.Text.RegularExpressions.GeneratedRegex(@"<[\w|/]+>|\w+|[^\w\s]")]
    private static partial System.Text.RegularExpressions.Regex Pattern();
}

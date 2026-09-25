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

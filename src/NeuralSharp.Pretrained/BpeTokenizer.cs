using System.Text;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using NeuralSharp.Generation;

namespace NeuralSharp.Pretrained;

/// <summary>
/// A byte-pair-encoding tokenizer read from a Hugging Face <c>tokenizer.json</c>: byte-level BPE (GPT-2, Llama 3, Qwen,
/// …) and SentencePiece-style BPE with the "▁" word marker and byte fallback (Llama 2, Mistral, Gemma, …). Special
/// (added) tokens are matched exactly before anything else. <see cref="Encode"/> adds no special tokens itself (chat
/// templates write them), like <c>add_special_tokens=False</c>. Pieces of the file it does not know are reported, not
/// guessed.
/// </summary>
public sealed class BpeTokenizer : ITokenizer
{
    private static readonly string Gpt2Pattern = @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
    private static readonly char[] ByteToChar = BuildByteMap();
    private static readonly Dictionary<char, byte> CharToByte = ByteToChar.Select((c, b) => (c, b)).ToDictionary(p => p.c, p => (byte)p.b);

    private readonly Dictionary<string, int> _vocabulary;
    private static readonly UTF8Encoding StrictUtf8 = new(false, throwOnInvalidBytes: true);
    private string[] _tokens;
    private readonly Dictionary<(string, string), int> _ranks;
    private readonly List<(string Content, int Id)> _added;
    private readonly HashSet<int> _special;
    private readonly HashSet<int> _addedIds;
    private readonly Regex? _addedPattern;
    private readonly List<Func<List<string>, bool, List<string>>> _preTokenizers = [];   // (pieces, text starts the input)
    private readonly List<Func<string, string>> _normalizers = [];
    private readonly List<Func<List<string>, List<string>>> _decoders = [];
    private readonly Dictionary<string, int[]> _cache = [];
    private readonly bool _byteLevel;
    private readonly bool _byteFallback;
    private readonly bool _ignoreMerges;
    private readonly int? _unknown;

    private BpeTokenizer(JsonObject json)
    {
        var model = json["model"]?.AsObject() ?? throw new InvalidDataException("tokenizer.json has no model.");
        if ((string?)model["type"] != "BPE")
        {
            throw new NotSupportedException($"Tokenizer model '{model["type"]}' is not supported (BPE).");
        }

        _vocabulary = model["vocab"]!.AsObject().ToDictionary(p => p.Key, p => (int)p.Value!);
        _ranks = [];
        int rank = 0;
        foreach (var merge in model["merges"]!.AsArray())
        {
            var (left, right) = merge is JsonArray pair ? ((string)pair[0]!, (string)pair[1]!) : Split((string)merge!);
            _ranks.TryAdd((left, right), rank++);
        }

        _byteFallback = (bool?)model["byte_fallback"] ?? false;
        _ignoreMerges = (bool?)model["ignore_merges"] ?? false;
        _unknown = model["unk_token"] is JsonValue unk && _vocabulary.TryGetValue((string)unk!, out int u) ? u : null;
        _added = [.. (json["added_tokens"]?.AsArray() ?? []).Select(t => ((string)t!["content"]!, (int)t["id"]!))];
        _special = [.. (json["added_tokens"]?.AsArray() ?? []).Where(t => (bool?)t!["special"] ?? false).Select(t => (int)t!["id"]!)];
        _addedIds = [.. _added.Select(a => a.Id)];
        foreach (var (content, id) in _added)
        {
            _vocabulary[content] = id;
        }

        int size = _vocabulary.Values.Max() + 1;
        _tokens = new string[size];
        foreach (var (token, id) in _vocabulary)
        {
            _tokens[id] = token;
        }

        if (_added.Count > 0)
        {
            _addedPattern = new Regex(string.Join('|', _added.Select(a => a.Content).OrderByDescending(c => c.Length).Select(Regex.Escape)));
        }

        AddNormalizer(json["normalizer"]);
        _byteLevel = AddPreTokenizer(json["pre_tokenizer"]);
        AddDecoder(json["decoder"]);
    }

    /// <summary>Reads tokenizer.json from a file or a model folder.</summary>
    public static BpeTokenizer Load(string path)
    {
        if (Directory.Exists(path))
        {
            path = Path.Combine(path, "tokenizer.json");
        }

        return new BpeTokenizer(JsonNode.Parse(File.ReadAllText(path))!.AsObject());
    }

    /// <summary>Reads a tokenizer from the JSON of a tokenizer.json.</summary>
    public static BpeTokenizer FromJson(JsonObject json) => new(json);

    /// <inheritdoc />
    public int VocabularySize => _tokens.Length;

    /// <summary>
    /// Extends the vocabulary to <paramref name="size"/> ids (models often have more embedding rows than the tokenizer has
    /// tokens, for alignment); the extra ids decode to nothing.
    /// </summary>
    public void PadVocabulary(int size)
    {
        if (size > _tokens.Length)
        {
            Array.Resize(ref _tokens, size);
        }
    }

    /// <summary>The id of <paramref name="token"/>, or null.</summary>
    public int? IdOf(string token) => _vocabulary.TryGetValue(token, out int id) ? id : null;

    /// <summary>The token with <paramref name="id"/>.</summary>
    public string TokenOf(int id) => _tokens[id] ?? "";

    /// <summary>Whether <paramref name="id"/> is a special token.</summary>
    public bool IsSpecial(int id) => _special.Contains(id);

    /// <inheritdoc />
    public IReadOnlyList<int> Encode(string text)
    {
        var ids = new List<int>();
        int last = 0;
        if (_addedPattern is not null)
        {
            foreach (Match match in _addedPattern.Matches(text))
            {
                EncodeText(text[last..match.Index], ids, last == 0);
                ids.Add(_vocabulary[match.Value]);
                last = match.Index + match.Length;
            }
        }

        EncodeText(text[last..], ids, last == 0);
        return ids;
    }

    /// <inheritdoc />
    public string Decode(IEnumerable<int> ids)
    {
        var tokens = ids.Select(id => (uint)id < (uint)_tokens.Length ? _tokens[id] ?? "" : "").ToList();
        if (_byteLevel)
        {
            // Byte-level tokens are bytes written as characters; added tokens are plain text.
            var bytes = new List<byte>();
            foreach (var (token, id) in tokens.Zip(ids))
            {
                if (_addedIds.Contains(id))
                {
                    bytes.AddRange(Encoding.UTF8.GetBytes(token));
                    continue;
                }

                foreach (char c in token)
                {
                    if (CharToByte.TryGetValue(c, out byte b))
                    {
                        bytes.Add(b);
                    }
                    else
                    {
                        bytes.AddRange(Encoding.UTF8.GetBytes(c.ToString()));
                    }
                }
            }

            return Encoding.UTF8.GetString([.. bytes]);
        }

        foreach (var decoder in _decoders)
        {
            tokens = decoder(tokens);
        }

        return string.Concat(tokens);
    }

    // atStart: the text begins the input (Metaspace's "first" scheme only prepends there, not after an added token).
    private void EncodeText(string text, List<int> ids, bool atStart)
    {
        if (text.Length == 0)
        {
            return;
        }

        foreach (var normalize in _normalizers)
        {
            text = normalize(text);
        }

        var pieces = new List<string> { text };
        foreach (var preTokenize in _preTokenizers)
        {
            pieces = preTokenize(pieces, atStart);
        }

        foreach (var piece in pieces.Where(p => p.Length > 0))
        {
            if (!_cache.TryGetValue(piece, out var cached))
            {
                cached = [.. Bpe(piece)];
                if (_cache.Count < 100_000)
                {
                    _cache[piece] = cached;
                }
            }

            ids.AddRange(cached);
        }
    }

    // Byte-pair merges over one piece, best (lowest-rank) pair first, leftmost among equals.
    private IEnumerable<int> Bpe(string piece)
    {
        if (_ignoreMerges && _vocabulary.TryGetValue(piece, out int whole))
        {
            return [whole];
        }

        var symbols = new List<string>();
        for (int i = 0; i < piece.Length; i += char.IsSurrogatePair(piece, i) ? 2 : 1)
        {
            symbols.Add(char.IsSurrogatePair(piece, i) ? piece.Substring(i, 2) : piece[i].ToString());
        }

        while (symbols.Count > 1)
        {
            int best = -1, bestRank = int.MaxValue;
            for (int i = 0; i < symbols.Count - 1; i++)
            {
                if (_ranks.TryGetValue((symbols[i], symbols[i + 1]), out int rank) && rank < bestRank)
                {
                    (best, bestRank) = (i, rank);
                }
            }

            if (best < 0)
            {
                break;
            }

            symbols[best] += symbols[best + 1];
            symbols.RemoveAt(best + 1);
        }

        var result = new List<int>();
        foreach (var symbol in symbols)
        {
            if (_vocabulary.TryGetValue(symbol, out int id))
            {
                result.Add(id);
            }
            else if (_byteFallback)
            {
                foreach (byte b in Encoding.UTF8.GetBytes(symbol))
                {
                    result.Add(_vocabulary.TryGetValue($"<0x{b:X2}>", out int byteId) ? byteId
                        : throw new InvalidDataException($"Byte fallback token <0x{b:X2}> is missing from the vocabulary."));
                }
            }
            else
            {
                result.Add(_unknown ?? throw new InvalidDataException($"'{symbol}' is not in the vocabulary and the tokenizer has no unknown token."));
            }
        }

        return result;
    }

    private void AddNormalizer(JsonNode? node)
    {
        switch (node)
        {
            case null:
                return;
            case JsonObject n when (string?)n["type"] == "Sequence":
                foreach (var inner in n["normalizers"]!.AsArray())
                {
                    AddNormalizer(inner);
                }

                return;
            case JsonObject n:
                string type = (string)n["type"]!;
                _normalizers.Add(type switch
                {
                    "NFC" => s => s.Normalize(NormalizationForm.FormC),
                    "NFKC" => s => s.Normalize(NormalizationForm.FormKC),
                    "NFD" => s => s.Normalize(NormalizationForm.FormD),
                    "NFKD" => s => s.Normalize(NormalizationForm.FormKD),
                    "Prepend" => s => (string)n["prepend"]! + s,
                    "Replace" => Replacer(n),
                    "Lowercase" => s => s.ToLowerInvariant(),
                    _ => throw new NotSupportedException($"Tokenizer normalizer '{type}' is not supported."),
                });
                return;
        }
    }

    private static Func<string, string> Replacer(JsonObject n)
    {
        string content = (string)n["content"]!;
        if (n["pattern"]?["String"] is { } literal)
        {
            string from = (string)literal!;
            return s => s.Replace(from, content, StringComparison.Ordinal);
        }

        var regex = new Regex((string)n["pattern"]!["Regex"]!);
        return s => regex.Replace(s, content);
    }

    // Returns true when the tokenizer is byte-level (its pieces are bytes written as characters).
    private bool AddPreTokenizer(JsonNode? node)
    {
        if (node is not JsonObject p)
        {
            return false;
        }

        string type = (string)p["type"]!;
        switch (type)
        {
            case "Sequence":
                bool byteLevel = false;
                foreach (var inner in p["pretokenizers"]!.AsArray())
                {
                    byteLevel |= AddPreTokenizer(inner);
                }

                return byteLevel;
            case "Split":
                var pattern = p["pattern"]!["Regex"] is { } r ? new Regex((string)r!) : new Regex(Regex.Escape((string)p["pattern"]!["String"]!));
                string behavior = (string?)p["behavior"] ?? "Isolated";
                bool invert = (bool?)p["invert"] ?? false;
                _preTokenizers.Add((pieces, atStart) => [.. pieces.SelectMany(piece => SplitPiece(piece, pattern, behavior, invert))]);
                return false;
            case "ByteLevel":
                bool prefix = (bool?)p["add_prefix_space"] ?? false, useRegex = (bool?)p["use_regex"] ?? true;
                var gpt2 = new Regex(Gpt2Pattern);
                _preTokenizers.Add((pieces, atStart) => [.. pieces.SelectMany(piece =>
                {
                    if (prefix && !piece.StartsWith(' '))
                    {
                        piece = " " + piece;
                    }

                    var parts = useRegex ? gpt2.Matches(piece).Select(m => m.Value) : [piece];
                    return parts.Select(part => new string([.. Encoding.UTF8.GetBytes(part).Select(b => ByteToChar[b])]));
                })]);
                return true;
            case "Metaspace":
                string replacement = (string?)p["replacement"] ?? "▁";
                string scheme = (string?)p["prepend_scheme"] ?? (((bool?)p["add_prefix_space"] ?? true) ? "always" : "never");
                bool split = (bool?)p["split"] ?? true;
                _preTokenizers.Add((pieces, atStart) =>
                {
                    var result = new List<string>();
                    for (int i = 0; i < pieces.Count; i++)
                    {
                        string s = pieces[i].Replace(" ", replacement, StringComparison.Ordinal);
                        if ((scheme == "always" || scheme == "first" && i == 0 && atStart) && !s.StartsWith(replacement, StringComparison.Ordinal))
                        {
                            s = replacement + s;
                        }

                        if (!split)
                        {
                            result.Add(s);
                            continue;
                        }

                        int start = 0;
                        for (int j = 1; j <= s.Length; j++)
                        {
                            if (j == s.Length || s.AsSpan(j).StartsWith(replacement, StringComparison.Ordinal))
                            {
                                result.Add(s[start..j]);
                                start = j;
                            }
                        }
                    }

                    return result;
                });
                return false;
            case "Digits":
                bool individual = (bool?)p["individual_digits"] ?? false;
                var digits = new Regex(individual ? @"\p{Nd}" : @"\p{Nd}+");
                _preTokenizers.Add((pieces, atStart) => [.. pieces.SelectMany(piece => SplitPiece(piece, digits, "Isolated", false))]);
                return false;
            case "Whitespace":
                var words = new Regex(@"\w+|[^\w\s]+");
                _preTokenizers.Add((pieces, atStart) => [.. pieces.SelectMany(piece => words.Matches(piece).Select(m => m.Value))]);
                return false;
            default:
                throw new NotSupportedException($"Tokenizer pre-tokenizer '{type}' is not supported.");
        }
    }

    private static IEnumerable<string> SplitPiece(string piece, Regex pattern, string behavior, bool invert)
    {
        var parts = new List<(string Text, bool Matched)>();
        int last = 0;
        foreach (Match m in pattern.Matches(piece))
        {
            if (m.Length == 0)
            {
                continue;
            }

            if (m.Index > last)
            {
                parts.Add((piece[last..m.Index], invert));
            }

            parts.Add((m.Value, !invert));
            last = m.Index + m.Length;
        }

        if (last < piece.Length)
        {
            parts.Add((piece[last..], invert));
        }

        switch (behavior)
        {
            case "Isolated":
                return parts.Select(p => p.Text);
            case "Removed":
                return parts.Where(p => !p.Matched).Select(p => p.Text);
            case "MergedWithPrevious":
                var previous = new List<string>();
                foreach (var (text, matched) in parts)
                {
                    if (matched && previous.Count > 0)
                    {
                        previous[^1] += text;
                    }
                    else
                    {
                        previous.Add(text);
                    }
                }

                return previous;
            case "MergedWithNext":
                var next = new List<string>();
                string pending = "";
                foreach (var (text, matched) in parts)
                {
                    if (matched)
                    {
                        pending += text;
                    }
                    else
                    {
                        next.Add(pending + text);
                        pending = "";
                    }
                }

                if (pending.Length > 0)
                {
                    next.Add(pending);
                }

                return next;
            default:
                throw new NotSupportedException($"Split behavior '{behavior}' is not supported.");
        }
    }

    private void AddDecoder(JsonNode? node)
    {
        if (node is not JsonObject d)
        {
            return;
        }

        string type = (string)d["type"]!;
        switch (type)
        {
            case "Sequence":
                foreach (var inner in d["decoders"]!.AsArray())
                {
                    AddDecoder(inner);
                }

                break;
            case "ByteLevel":
                break;                                                           // handled in Decode
            case "Replace":
                var replace = Replacer(d);
                _decoders.Add(tokens => [.. tokens.Select(replace)]);
                break;
            case "ByteFallback":
                _decoders.Add(tokens =>
                {
                    var result = new List<string>();
                    var bytes = new List<byte>();
                    void Flush()
                    {
                        if (bytes.Count > 0)
                        {
                            // As the tokenizers library: bytes that are not valid UTF-8 become one U+FFFD each.
                            try
                            {
                                result.Add(StrictUtf8.GetString([.. bytes]));
                            }
                            catch (DecoderFallbackException)
                            {
                                result.Add(new string('\uFFFD', bytes.Count));
                            }

                            bytes.Clear();
                        }
                    }

                    foreach (var token in tokens)
                    {
                        if (token.Length == 6 && token.StartsWith("<0x", StringComparison.Ordinal) && token[5] == '>')
                        {
                            bytes.Add(Convert.ToByte(token[3..5], 16));
                        }
                        else
                        {
                            Flush();
                            result.Add(token);
                        }
                    }

                    Flush();
                    return result;
                });
                break;
            case "Fuse":
                _decoders.Add(tokens => [string.Concat(tokens)]);
                break;
            case "Strip":
                string content = (string)d["content"]!;
                int start = (int?)d["start"] ?? 0, stop = (int?)d["stop"] ?? 0;
                _decoders.Add(tokens =>
                {
                    if (tokens.Count == 0)
                    {
                        return tokens;
                    }

                    string first = tokens[0];
                    for (int i = 0; i < start && first.StartsWith(content, StringComparison.Ordinal); i++)
                    {
                        first = first[content.Length..];
                    }

                    tokens[0] = first;
                    string last = tokens[^1];
                    for (int i = 0; i < stop && last.EndsWith(content, StringComparison.Ordinal); i++)
                    {
                        last = last[..^content.Length];
                    }

                    tokens[^1] = last;
                    return tokens;
                });
                break;
            case "Metaspace":
                string replacement = (string?)d["replacement"] ?? "▁";
                _decoders.Add(tokens =>
                {
                    var text = string.Concat(tokens).Replace(replacement, " ", StringComparison.Ordinal);
                    return [text.StartsWith(' ') ? text[1..] : text];
                });
                break;
            default:
                throw new NotSupportedException($"Tokenizer decoder '{type}' is not supported.");
        }
    }

    private static (string, string) Split(string merge)
    {
        int space = merge.IndexOf(' ', 1);
        return (merge[..space], merge[(space + 1)..]);
    }

    // GPT-2's byte → printable character table: printable bytes map to themselves, the rest to 256 + n.
    private static char[] BuildByteMap()
    {
        var map = new char[256];
        var printable = Enumerable.Range('!', '~' - '!' + 1).Concat(Enumerable.Range('¡', '¬' - '¡' + 1)).Concat(Enumerable.Range('®', 'ÿ' - '®' + 1)).ToHashSet();
        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            map[b] = printable.Contains(b) ? (char)b : (char)(256 + n++);
        }

        return map;
    }
}

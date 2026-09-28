using System.Text.Json.Nodes;
using NeuralSharp.Datasets;

namespace NeuralSharp.Pretrained;

/// <summary>
/// Where a model comes from: a local folder, or a Hugging Face model id such as "Qwen/Qwen3-0.6B". An id is looked up in
/// Hugging Face's own cache (models fetched with transformers or huggingface-cli), then in NeuralSharp's download cache,
/// and downloaded otherwise: only the files the library reads (config, tokenizer, chat template, generation config and
/// the safetensors weights), into downloads/huggingface/models/&lt;owner&gt;/&lt;name&gt;/&lt;commit&gt;/. Gated and private models
/// need a token (HF_TOKEN or huggingface-cli login).
/// </summary>
public static class ModelSource
{
    private static readonly string[] Wanted = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "added_tokens.json", "chat_template.jinja", "chat_template.json", "model.safetensors.index.json"];

    /// <summary>Whether <paramref name="model"/> reads as a Hugging Face id ("owner/name") rather than a folder.</summary>
    public static bool IsModelId(string model) =>
        !Directory.Exists(model) && !Path.IsPathRooted(model) && model.Count(c => c == '/') == 1 && !model.StartsWith('.') && !model.Contains('\\');

    /// <summary>
    /// The local folder of <paramref name="model"/>: the folder itself, or a Hugging Face id found in a cache or
    /// downloaded (see the class summary). With <paramref name="download"/> false, only the caches are searched.
    /// </summary>
    public static string Resolve(string model, string revision = "main", string? token = null, Downloader? downloader = null, bool download = true)
    {
        if (Directory.Exists(model))
        {
            return model;
        }

        if (!IsModelId(model))
        {
            throw new DirectoryNotFoundException($"'{model}' is neither a folder nor a Hugging Face model id (owner/name).");
        }

        if (revision == "main" && HuggingFaceCache(model) is { } cached)
        {
            (downloader ?? Downloader.Shared).Log?.Invoke($"{model}: found in the Hugging Face cache ({cached})");
            return cached;
        }

        if (!download)
        {
            return NewestDownloaded(model, downloader) ?? throw new DirectoryNotFoundException($"{model} is not in a local cache.");
        }

        return DownloadAsync(model, revision, token, downloader).GetAwaiter().GetResult();
    }

    /// <summary>Downloads the files the library reads of a Hugging Face model (once) and returns their folder.</summary>
    public static async Task<string> DownloadAsync(string repo, string revision = "main", string? token = null, Downloader? downloader = null,
        CancellationToken cancellationToken = default)
    {
        var d = downloader ?? Downloader.Shared;
        IReadOnlyList<RepoFile> files;
        string commit;
        try
        {
            commit = await HuggingFace.ResolveRevisionAsync(repo, "models", revision, token, d, cancellationToken).ConfigureAwait(false);
            files = await HuggingFace.ListFilesAsync(repo, "models", commit, token, d, cancellationToken).ConfigureAwait(false);
        }
        catch (HttpRequestException ex) when (ex.StatusCode is null && NewestDownloaded(repo, d) is { } offline)
        {
            d.Log?.Invoke($"{repo}: Hugging Face cannot be reached ({ex.Message.Split('\n')[0]}); using the copy downloaded before: {offline}");
            return offline;
        }

        var chosen = Choose(repo, files);
        d.Log?.Invoke($"{repo}: {chosen.Count} of {files.Count} files needed ({Downloader.Size(chosen.Sum(f => f.Size))})");
        string? folder = null;
        foreach (var file in chosen)
        {
            string local = await HuggingFace.DownloadFileAsync(repo, file.Path, "models", commit, token, d, cancellationToken).ConfigureAwait(false);
            folder ??= Path.GetDirectoryName(local);
        }

        return folder!;
    }

    // The files a model needs: its configuration, tokenizer and chat template, and the safetensors weights (the shards the
    // index names, or the single file). PyTorch .bin files, ONNX exports and subfolders (original/, onnx/ …) are skipped.
    private static List<RepoFile> Choose(string repo, IReadOnlyList<RepoFile> files)
    {
        var root = files.Where(f => !f.Path.Contains('/')).ToList();
        if (!root.Any(f => f.Path == "config.json"))
        {
            throw new FileNotFoundException(root.Any(f => f.Path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
                ? $"{repo} holds GGUF files, not a transformers model (config.json and safetensors)."
                : $"{repo} has no config.json at its top level; it is not a transformers model.");
        }

        var chosen = root.Where(f => Wanted.Contains(f.Path)).ToList();
        var safetensors = root.Where(f => f.Path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase)).ToList();
        if (safetensors.Count == 0)
        {
            throw new FileNotFoundException(root.Any(f => f.Path.EndsWith(".bin", StringComparison.OrdinalIgnoreCase))
                ? $"{repo} has only PyTorch .bin weights; NeuralSharp reads safetensors (look for a safetensors version of the model)."
                : $"{repo} has no safetensors weights.");
        }

        // A sharded model lists its shards in the index; otherwise prefer the transformers file over a "consolidated" copy.
        bool sharded = root.Any(f => f.Path == "model.safetensors.index.json");
        chosen.AddRange(sharded
            ? safetensors.Where(f => !f.Path.StartsWith("consolidated", StringComparison.Ordinal))
            : safetensors.Any(f => f.Path == "model.safetensors") ? safetensors.Where(f => f.Path == "model.safetensors") : safetensors);
        return chosen;
    }

    // A snapshot in Hugging Face's cache (HF_HOME or ~/.cache/huggingface/hub) with a config and safetensors weights.
    private static string? HuggingFaceCache(string repo)
    {
        string home = Environment.GetEnvironmentVariable("HF_HUB_CACHE")
                      ?? Path.Combine(Environment.GetEnvironmentVariable("HF_HOME")
                                      ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "huggingface"), "hub");
        string snapshots = Path.Combine(home, "models--" + repo.Replace("/", "--", StringComparison.Ordinal), "snapshots");
        if (!Directory.Exists(snapshots))
        {
            return null;
        }

        return Directory.GetDirectories(snapshots)
            .Where(d => File.Exists(Path.Combine(d, "config.json")) && Directory.EnumerateFiles(d, "*.safetensors").Any())
            .OrderByDescending(Directory.GetLastWriteTimeUtc)
            .FirstOrDefault();
    }

    // The most recent download of a model in NeuralSharp's cache.
    private static string? NewestDownloaded(string repo, Downloader? downloader)
    {
        string folder = Path.Combine([(downloader ?? Downloader.Shared).CacheFolder, "huggingface", "models", .. repo.Split('/')]);
        return Directory.Exists(folder)
            ? Directory.GetDirectories(folder).Where(d => File.Exists(Path.Combine(d, "config.json"))).OrderByDescending(Directory.GetLastWriteTimeUtc).FirstOrDefault()
            : null;
    }
}

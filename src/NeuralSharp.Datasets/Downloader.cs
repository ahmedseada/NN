using System.Net;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;

namespace NeuralSharp.Datasets;

/// <summary>Progress of a download.</summary>
/// <param name="Url">What is downloaded.</param>
/// <param name="Received">Bytes so far.</param>
/// <param name="Total">Size, when the server tells it.</param>
public sealed record DownloadProgress(string Url, long Received, long? Total);

/// <summary>
/// Downloads files once into a cache folder and returns the local path: interrupted downloads resume (HTTP ranges),
/// failures are retried, a file appears only when complete. The cache is <c>NEURALSHARP_CACHE</c> or
/// <c>~/.cache/neuralsharp</c>, under <c>downloads/</c>. Credentials in headers are not part of the cache key.
/// </summary>
public sealed class Downloader
{
    private static readonly Lazy<Downloader> SharedInstance = new(() => new Downloader());
    private readonly HttpClient _http;

    /// <summary>A downloader using <paramref name="http"/> (default: its own client) and <paramref name="cacheFolder"/>.</summary>
    public Downloader(HttpClient? http = null, string? cacheFolder = null)
    {
        _http = http ?? new HttpClient(new HttpClientHandler { AutomaticDecompression = DecompressionMethods.None }) { Timeout = Timeout.InfiniteTimeSpan };
        if (http is null)
        {
            _http.DefaultRequestHeaders.UserAgent.Add(new ProductInfoHeaderValue("NeuralSharp", "1.0"));
        }

        CacheFolder = cacheFolder ?? Path.Combine(DefaultCacheRoot, "downloads");
    }

    /// <summary>The downloader <see cref="Dataset"/> sources use unless given another.</summary>
    public static Downloader Shared => SharedInstance.Value;

    /// <summary><c>NEURALSHARP_CACHE</c>, or <c>~/.cache/neuralsharp</c>.</summary>
    public static string DefaultCacheRoot => Environment.GetEnvironmentVariable("NEURALSHARP_CACHE")
        ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "neuralsharp");

    /// <summary>Where files are kept.</summary>
    public string CacheFolder { get; }

    /// <summary>Called as downloads progress (about every megabyte).</summary>
    public IProgress<DownloadProgress>? Progress { get; init; }

    /// <summary>Re-download files even when cached.</summary>
    public bool Refresh { get; init; }

    /// <summary>Attempts per download.</summary>
    public int Attempts { get; init; } = 5;

    /// <summary>The wait before the first retry; it doubles with each attempt (at most 30 s).</summary>
    public TimeSpan RetryDelay { get; init; } = TimeSpan.FromSeconds(1);

    /// <summary>The client, for sources that call web APIs.</summary>
    public HttpClient Http => _http;

    /// <summary>Downloads (or finds in the cache) <paramref name="url"/>; see <see cref="DownloadAsync"/>.</summary>
    public string Download(string url, IReadOnlyDictionary<string, string>? headers = null, string? fileName = null) =>
        DownloadAsync(url, headers, fileName).GetAwaiter().GetResult();

    /// <summary>
    /// The local path of <paramref name="url"/>, downloading it first unless cached. <paramref name="fileName"/> names the
    /// file (its extension decides how it is read; default: the URL's last segment).
    /// </summary>
    public async Task<string> DownloadAsync(string url, IReadOnlyDictionary<string, string>? headers = null, string? fileName = null,
        CancellationToken cancellationToken = default)
    {
        fileName = Sanitize(fileName ?? Path.GetFileName(new Uri(url).LocalPath));
        string folder = Path.Combine(CacheFolder, Key(url));
        string target = Path.Combine(folder, fileName.Length > 0 ? fileName : "download");
        if (File.Exists(target) && !Refresh)
        {
            return target;
        }

        Directory.CreateDirectory(folder);
        string partial = target + ".part";
        for (int attempt = 1; ; attempt++)
        {
            try
            {
                await DownloadOnceAsync(url, headers, partial, cancellationToken).ConfigureAwait(false);
                File.Move(partial, target, overwrite: true);
                await File.WriteAllTextAsync(Path.Combine(folder, "source.json"), new JsonObject { ["url"] = url, ["file"] = Path.GetFileName(target) }.ToJsonString(),
                    cancellationToken).ConfigureAwait(false);
                return target;
            }
            catch (Exception ex) when (attempt < Attempts && ex is HttpRequestException { StatusCode: null or >= HttpStatusCode.InternalServerError or HttpStatusCode.TooManyRequests }
                                           or IOException && ex is not FileNotFoundException)
            {
                await Task.Delay(Backoff(attempt), cancellationToken).ConfigureAwait(false);
            }
        }
    }

    /// <summary>A GET with <paramref name="headers"/>, for API calls; throws with the server's message on failure.</summary>
    public async Task<string> GetStringAsync(string url, IReadOnlyDictionary<string, string>? headers = null, CancellationToken cancellationToken = default)
    {
        for (int attempt = 1; ; attempt++)
        {
            using var request = Request(url, headers);
            using var response = await _http.SendAsync(request, cancellationToken).ConfigureAwait(false);
            string body = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
            if (response.IsSuccessStatusCode)
            {
                return body;
            }

            if (attempt >= Attempts || response.StatusCode is not (HttpStatusCode.TooManyRequests or >= HttpStatusCode.InternalServerError))
            {
                throw Failure(url, response.StatusCode, body);
            }

            await Task.Delay(Backoff(attempt), cancellationToken).ConfigureAwait(false);
        }
    }

    /// <summary>GETs <paramref name="url"/> and every following page named by a <c>Link: &lt;…&gt;; rel="next"</c> header.</summary>
    public async Task<List<string>> GetPagesAsync(string url, IReadOnlyDictionary<string, string>? headers = null, CancellationToken cancellationToken = default)
    {
        var pages = new List<string>();
        string? next = url;
        while (next is not null && pages.Count < 10_000)
        {
            using var request = Request(next, headers);
            using var response = await _http.SendAsync(request, cancellationToken).ConfigureAwait(false);
            string body = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
            if (!response.IsSuccessStatusCode)
            {
                throw Failure(next, response.StatusCode, body);
            }

            pages.Add(body);
            next = response.Headers.TryGetValues("Link", out var links) ? NextLink(string.Join(",", links)) : null;
        }

        return pages;
    }

    private static string? NextLink(string header)
    {
        foreach (var part in header.Split(','))
        {
            int open = part.IndexOf('<'), close = part.IndexOf('>');
            if (open >= 0 && close > open && part.Contains("rel=\"next\"", StringComparison.Ordinal))
            {
                return part[(open + 1)..close].Trim();
            }
        }

        return null;
    }

    private async Task DownloadOnceAsync(string url, IReadOnlyDictionary<string, string>? headers, string partial, CancellationToken cancellationToken)
    {
        long existing = File.Exists(partial) ? new FileInfo(partial).Length : 0;
        using var request = Request(url, headers);
        if (existing > 0)
        {
            request.Headers.Range = new RangeHeaderValue(existing, null);
        }

        using var response = await _http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
        if (response.StatusCode == HttpStatusCode.RequestedRangeNotSatisfiable && existing > 0)
        {
            return;                                                 // the partial file is already complete
        }

        if (!response.IsSuccessStatusCode)
        {
            throw Failure(url, response.StatusCode, await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false));
        }

        bool resumed = existing > 0 && response.StatusCode == HttpStatusCode.PartialContent;
        long? total = response.Content.Headers.ContentLength is { } length ? length + (resumed ? existing : 0) : null;
        await using var source = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
        await using var file = new FileStream(partial, resumed ? FileMode.Append : FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20, useAsync: true);
        var buffer = new byte[1 << 20];
        long received = resumed ? existing : 0, reported = 0;
        int read;
        while ((read = await source.ReadAsync(buffer, cancellationToken).ConfigureAwait(false)) > 0)
        {
            await file.WriteAsync(buffer.AsMemory(0, read), cancellationToken).ConfigureAwait(false);
            received += read;
            if (received - reported >= 1 << 20)
            {
                reported = received;
                Progress?.Report(new DownloadProgress(url, received, total));
            }
        }

        if (total is { } expected && received != expected)
        {
            throw new IOException($"{url}: received {received} of {expected} bytes.");
        }

        Progress?.Report(new DownloadProgress(url, received, total));
    }

    private TimeSpan Backoff(int attempt) => TimeSpan.FromTicks(Math.Min(TimeSpan.FromSeconds(30).Ticks, RetryDelay.Ticks << Math.Min(attempt - 1, 20)));

    private static HttpRequestMessage Request(string url, IReadOnlyDictionary<string, string>? headers)
    {
        var request = new HttpRequestMessage(HttpMethod.Get, url);
        foreach (var (name, value) in headers ?? new Dictionary<string, string>())
        {
            request.Headers.TryAddWithoutValidation(name, value);
        }

        return request;
    }

    private static HttpRequestException Failure(string url, HttpStatusCode status, string body)
    {
        string hint = status switch
        {
            HttpStatusCode.Unauthorized => " (sign-in needed: set the token)",
            HttpStatusCode.Forbidden => " (no access: a gated or private resource, or the token lacks permission)",
            HttpStatusCode.NotFound => " (not found, or private without a token)",
            _ => "",
        };
        string detail = body.Length > 300 ? body[..300] + "…" : body;
        return new HttpRequestException($"{url}: {(int)status} {status}{hint}. {detail}".TrimEnd(), null, status);
    }

    private static string Key(string url)
    {
        string hash = Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(url)))[..16];
        var uri = new Uri(url);
        return Sanitize($"{uri.Host}-{hash}");
    }

    private static string Sanitize(string name)
    {
        var invalid = Path.GetInvalidFileNameChars();
        return new string([.. name.Select(c => invalid.Contains(c) || c is ':' or '?' or '*' ? '_' : c)]);
    }
}

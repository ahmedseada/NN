using System.Diagnostics;

namespace Qasd;

/// <summary>
/// A one-line progress bar for the console: the bar and percentage, how many units are done, still to do and in total,
/// what is happening now, the time elapsed and the estimated time left. Redrawn in place on a terminal; when the output goes to a file or a
/// pipe, one line is written every tenth of the way instead.
/// </summary>
/// <example>
/// <code>
/// var bar = new ConsoleProgress("training", total: 1200, unit: "steps");
/// bar.Report(done: 300, current: "epoch 2, loss 0.4120");
/// bar.WriteLine("epoch 2: validation accuracy 81.2%");   // a line above the bar
/// bar.Complete("best epoch 7");
/// </code>
/// </example>
public sealed class ConsoleProgress
{
    private const int BarWidth = 24;
    private readonly string _label;
    private readonly string _unit;
    private readonly Stopwatch _clock = Stopwatch.StartNew();
    private readonly bool _redraw = !Console.IsOutputRedirected;
    private readonly object _gate = new();
    private long _lastDraw = long.MinValue;
    private int _lastTenth = -1;
    private int _drawnWidth;
    private int _done;
    private string? _current;
    private bool _completed;

    /// <param name="label">What is progressing ("training").</param>
    /// <param name="total">How many units make up the whole.</param>
    /// <param name="unit">The units' name ("steps", "messages").</param>
    public ConsoleProgress(string label, int total, string unit = "steps")
    {
        _label = label;
        _unit = unit;
        Total = Math.Max(1, total);
    }

    /// <summary>How many units make up the whole (can be lowered, e.g. when training stops early).</summary>
    public int Total { get; set; }

    /// <summary>Time since the bar was created.</summary>
    public TimeSpan Elapsed => _clock.Elapsed;

    /// <summary>Records <paramref name="done"/> units finished and, optionally, what is happening now.</summary>
    public void Report(int done, string? current = null)
    {
        lock (_gate)
        {
            _done = Math.Clamp(done, 0, Total);
            _current = current ?? _current;
            long now = _clock.ElapsedMilliseconds;
            if (_redraw)
            {
                if (now - _lastDraw >= 100 || _done == Total)
                {
                    _lastDraw = now;
                    Draw();
                }
            }
            else
            {
                int tenth = _done * 10 / Total;
                if (tenth > _lastTenth)
                {
                    _lastTenth = tenth;
                    Console.WriteLine(Text());
                }
            }
        }
    }

    /// <summary>Writes <paramref name="line"/> above the bar (the bar is drawn again below it).</summary>
    public void WriteLine(string line)
    {
        lock (_gate)
        {
            if (_redraw && _drawnWidth > 0)
            {
                Console.Write('\r' + new string(' ', _drawnWidth) + '\r');
                _drawnWidth = 0;
            }

            Console.WriteLine(line);
            if (_redraw && !_completed)
            {
                Draw();
            }
        }
    }

    /// <summary>Marks the whole done and ends the bar's line.</summary>
    public void Complete(string? current = null)
    {
        lock (_gate)
        {
            if (_completed)
            {
                return;
            }

            Total = Math.Max(1, _done);
            _done = Total;
            _current = current ?? _current;
            if (_redraw)
            {
                Draw();
                Console.WriteLine();
            }
            else
            {
                Console.WriteLine(Text());
            }

            _completed = true;
        }
    }

    private void Draw()
    {
        string text = Text();
        int width = 0;
        try
        {
            width = Console.WindowWidth - 1;
        }
        catch (IOException)
        {
        }

        if (width > 20 && text.Length > width)
        {
            text = text[..(width - 1)] + "…";
        }

        Console.Write('\r' + text + (text.Length < _drawnWidth ? new string(' ', _drawnWidth - text.Length) : ""));
        _drawnWidth = text.Length;
    }

    private string Text()
    {
        double fraction = (double)_done / Total;
        int filled = (int)Math.Round(fraction * BarWidth);
        var elapsed = _clock.Elapsed;
        string eta = _done == Total ? "done"
            : _done == 0 ? "ETA --:--"
            : "ETA " + Format(TimeSpan.FromSeconds(elapsed.TotalSeconds / _done * (Total - _done)));
        return $"{_label} [{new string('█', filled)}{new string('░', BarWidth - filled)}] {fraction,6:P1}  "
               + $"{_unit}: done {_done:N0}, still {Total - _done:N0}, total {Total:N0}"
               + (string.IsNullOrEmpty(_current) ? "" : $" | current: {_current}")
               + $" | elapsed {Format(elapsed)} | {eta}";
    }

    private static string Format(TimeSpan time) =>
        time.TotalHours >= 1 ? $"{(int)time.TotalHours}:{time.Minutes:D2}:{time.Seconds:D2}" : $"{time.Minutes:D2}:{time.Seconds:D2}";
}

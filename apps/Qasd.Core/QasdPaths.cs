namespace Qasd;

/// <summary>
/// Where the trained models live by default: in their own projects, apps/Qasd/models/intents.nsm (the classifier) and
/// apps/Qasd.Tuned/models/qasd-tuned (the tuned model). The apps folder is found from the running program (the folder
/// holding Qasd.slnx, looking up from it); a published copy without the sources uses models/ next to itself.
/// </summary>
public static class QasdPaths
{
    private static readonly Lazy<string?> Apps = new(() =>
    {
        foreach (string start in new[] { AppContext.BaseDirectory, Directory.GetCurrentDirectory() })
        {
            for (var folder = new DirectoryInfo(start); folder is not null; folder = folder.Parent)
            {
                if (File.Exists(Path.Combine(folder.FullName, "Qasd.slnx")))
                {
                    return folder.FullName;
                }

                if (File.Exists(Path.Combine(folder.FullName, "apps", "Qasd.slnx")))
                {
                    return Path.Combine(folder.FullName, "apps");
                }
            }
        }

        return null;
    });

    /// <summary>The classifier's model file: apps/Qasd/models/intents.nsm.</summary>
    public static string Classifier => Apps.Value is { } apps
        ? Path.Combine(apps, "Qasd", "models", "intents.nsm")
        : Path.Combine(AppContext.BaseDirectory, "models", "intents.nsm");

    /// <summary>The tuned model's folder: apps/Qasd.Tuned/models/qasd-tuned.</summary>
    public static string Tuned => Apps.Value is { } apps
        ? Path.Combine(apps, "Qasd.Tuned", "models", "qasd-tuned")
        : Path.Combine(AppContext.BaseDirectory, "models", "qasd-tuned");
}

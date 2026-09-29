namespace Qasd;

/// <summary>
/// Where Qasd's models and data live by default, in the Qasd project: apps/Qasd/models/intents.qasd (the classifier),
/// apps/Qasd/models/tuned (the chat model tuned with idrak-tune) and apps/Qasd/data (the intent data and its split). The apps folder is found from the running program (the folder
/// holding Qasd.slnx, looking up from it). A published copy uses models/ next to itself when it has one.
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

    // models/ next to the program: a published copy (dotnet publish puts the trained models there).
    private static readonly string? Published = Directory.Exists(Path.Combine(AppContext.BaseDirectory, "models"))
        ? Path.Combine(AppContext.BaseDirectory, "models") : null;

    /// <summary>The classifier's model file: apps/Qasd/models/intents.qasd (models/intents.qasd in a published copy).</summary>
    public static string Classifier => Published is { } published ? Path.Combine(published, "intents.qasd") : Apps.Value is { } apps
        ? Path.Combine(apps, "Qasd", "models", "intents.qasd")
        : Path.Combine(AppContext.BaseDirectory, "models", "intents.qasd");

    /// <summary>The tuned model's folder (the adapter idrak-tune writes): apps/Qasd/models/tuned (models/tuned in a published copy).</summary>
    public static string Tuned => Published is { } published ? Path.Combine(published, "tuned") : Apps.Value is { } apps
        ? Path.Combine(apps, "Qasd", "models", "tuned")
        : Path.Combine(AppContext.BaseDirectory, "models", "tuned");

    /// <summary>The data folder: apps/Qasd/data (the split written by qasd split goes to apps/Qasd/data/split).</summary>
    public static string Data => Apps.Value is { } apps
        ? Path.Combine(apps, "Qasd", "data")
        : Path.Combine(AppContext.BaseDirectory, "data");
}

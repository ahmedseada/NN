using NeuralSharp;
using NeuralSharp.Generation;

// Coding tools: the workspace sandbox, reading, searching, exact edits, writes and allowlisted commands.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] CodingToolsGroup =
    [
        ("coding tools: list, read, search, edit and write stay inside the workspace and explain mistakes", CodingToolsFiles),
        ("coding tools: run_command starts allowlisted programs without a shell and trims long output", CodingToolsCommands),
    ];

    private static void CodingToolsFiles(Device device)
    {
        _ = device;
        string root = Path.Combine(Path.GetTempPath(), "ns-coding-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.Combine(root, "src", "app"));
        Directory.CreateDirectory(Path.Combine(root, "node_modules", "lib"));
        Directory.CreateDirectory(Path.Combine(root, "bin"));
        try
        {
            File.WriteAllText(Path.Combine(root, "src", "app", "user.service.ts"), "export class UserService {\n  getUsers() {\n    return [];\n  }\n}\n");
            File.WriteAllText(Path.Combine(root, "src", "Program.cs"), "class Program\r\n{\r\n    static void Main() { }\r\n}\r\n");
            File.WriteAllText(Path.Combine(root, "node_modules", "lib", "index.ts"), "export class UserService {}\n");
            File.WriteAllText(Path.Combine(root, "bin", "app.dll"), "x\0y");
            File.WriteAllText(Path.Combine(root, "long.txt"), string.Join('\n', Enumerable.Range(1, 1000).Select(i => $"line {i}")));
            var tools = new CodingTools(root, new CodingToolOptions { MaxReadLines = 50 });
            var registry = tools.Registry();
            string Run(string name, string json)
            {
                var result = registry.InvokeAsync(Call(name, json)).GetAwaiter().GetResult();
                return result.Content ?? result.Error!;
            }

            Check(tools.All.Select(t => t.Definition.Name).SequenceEqual(["list_files", "read_file", "search", "edit_file", "write_file", "run_command"]), "tool names");
            Check(Run("list_files", "{}") == "long.txt\nsrc/Program.cs\nsrc/app/user.service.ts", $"list skips ignored folders: {Run("list_files", "{}")}");
            Check(Run("list_files", "{\"pattern\":\"*.ts\"}") == "src/app/user.service.ts", "list pattern");
            Check(Run("read_file", "{\"path\":\"src/app/user.service.ts\",\"start_line\":2,\"end_line\":3}") == "2|   getUsers() {\n3|     return [];", "read range");
            string longRead = Run("read_file", "{\"path\":\"long.txt\"}");
            Check(longRead.Contains("50| line 50", StringComparison.Ordinal) && !longRead.Contains("line 51\n", StringComparison.Ordinal)
                  && longRead.EndsWith("start_line 51)", StringComparison.Ordinal), "long files come in parts");
            Check(Run("read_file", "{\"path\":\"../secret.txt\"}").Contains("outside the workspace", StringComparison.Ordinal), "no escaping the workspace");
            Check(Run("read_file", "{\"path\":\"/etc/passwd\"}").Contains("outside the workspace", StringComparison.Ordinal), "no absolute paths out");
            Check(Run("read_file", "{\"path\":\"missing.cs\"}").Contains("does not exist", StringComparison.Ordinal), "missing file");

            Check(Run("search", "{\"pattern\":\"class \\\\w+Service\"}") == "src/app/user.service.ts:1: export class UserService {", "search skips node_modules");
            Check(Run("search", "{\"pattern\":\"MAIN\",\"ignore_case\":true,\"glob\":\"*.cs\"}") == "src/Program.cs:3: static void Main() { }", "search options");
            Check(Run("search", "{\"pattern\":\"(\"}").Contains("invalid regular expression", StringComparison.Ordinal), "bad regex");

            string edited = Run("edit_file", "{\"path\":\"src/app/user.service.ts\",\"old_text\":\"    return [];\",\"new_text\":\"    return this.http.get('/api/users');\"}");
            Check(edited.StartsWith("Edited src/app/user.service.ts", StringComparison.Ordinal) && edited.Contains("3|     return this.http.get", StringComparison.Ordinal), $"edit shows the result: {edited}");
            Check(Run("edit_file", "{\"path\":\"src/app/user.service.ts\",\"old_text\":\"return [];\",\"new_text\":\"x\"}").Contains("not found", StringComparison.Ordinal), "stale edit");
            Check(Run("edit_file", "{\"path\":\"src/app/user.service.ts\",\"old_text\":\"getUsers() {\\n return\",\"new_text\":\"x\"}").Contains("whitespace is ignored", StringComparison.Ordinal), "whitespace hint");
            Check(Run("edit_file", "{\"path\":\"src/app/user.service.ts\",\"old_text\":\"s\",\"new_text\":\"S\"}").Contains("occurs", StringComparison.Ordinal), "ambiguous edit");

            // Line endings: the model writes \n; a CRLF file keeps CRLF.
            Run("edit_file", "{\"path\":\"src/Program.cs\",\"old_text\":\"{\\n    static void Main() { }\\n}\",\"new_text\":\"{\\n    static void Main() => System.Console.WriteLine(1);\\n}\"}");
            Check(File.ReadAllText(Path.Combine(root, "src", "Program.cs")) == "class Program\r\n{\r\n    static void Main() => System.Console.WriteLine(1);\r\n}\r\n", "CRLF kept");

            Check(Run("write_file", "{\"path\":\"src/app/user.ts\",\"content\":\"export interface User { id: number; }\\n\"}") == "Created src/app/user.ts (1 lines).", "write");
            Check(File.Exists(Path.Combine(root, "src", "app", "user.ts")), "written");
            Check(Run("write_file", "{\"path\":\"../x.ts\",\"content\":\"\"}").Contains("outside", StringComparison.Ordinal), "no writing outside");

            var readOnly = new CodingTools(root, new CodingToolOptions { ReadOnly = true });
            Check(readOnly.All.Count == 3, "read-only toolset");
        }
        finally
        {
            Directory.Delete(root, true);
        }
    }

    private static void CodingToolsCommands(Device device)
    {
        _ = device;
        string root = Path.Combine(Path.GetTempPath(), "ns-coding-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            var tools = new CodingTools(root, new CodingToolOptions { Commands = ["dotnet", "git"], MaxOutputCharacters = 300 });
            string Run(string command, int? timeout = null) => tools.RunCommandAsync(command, ".", timeout).GetAwaiter().GetResult();

            Check(Run("rm -rf /").Contains("not an allowed program", StringComparison.Ordinal), "allowlist");
            Check(Run("/usr/bin/dotnet --version").Contains("not an allowed program", StringComparison.Ordinal), "no paths to programs");
            Check(Run("dotnet build && rm x").Contains("not run by a shell", StringComparison.Ordinal), "no shell operators");
            Check(Run("git push").Contains("only git", StringComparison.Ordinal), "git read-only");

            string version = Run("dotnet --version");
            Check(version.StartsWith("$ dotnet --version\nexit code 0", StringComparison.Ordinal), $"dotnet runs: {version}");
            string help = Run("dotnet --help");
            Check(help.Contains("characters cut", StringComparison.Ordinal) && help.Length < 500, "long output is trimmed");
            string failed = Run("dotnet \"no such command\"");
            Check(!failed.Contains("exit code 0", StringComparison.Ordinal), $"failure exit code: {failed}");
        }
        finally
        {
            Directory.Delete(root, true);
        }
    }
}

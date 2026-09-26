using System.Diagnostics;
using System.Globalization;
using System.Text.Json.Nodes;
using NeuralSharp;
using NeuralSharp.Generation;
using NeuralSharp.Layers;
using NeuralSharp.Pretrained;

// Pretrained models in the Hugging Face layout, run by NeuralSharp's own engine (CPU or CUDA):
//
//   info  <folder>                  what the loader made of the model (architecture, parameters, notes)
//   chat  <folder>                  chat with the model in its own chat template (reasoning and tool calls shown)
//   check <reference.json>          compare with transformers: token ids, chat templates, logits, greedy output
//                                   (make the reference with tools/pytorch/pretrained_reference.py)
//
// Options: --cuda / --cpu, --int8 (int8 weights), --kv8 (int8 KV cache), --context N (default 4096),
//          --folder F (check: read the model from F instead of the folder named in the reference), --no-think.
var positional = new List<string>();
bool int8 = false, kv8 = false, noThink = false;
int context = 4096;
string? folderOverride = null;
Device device = Device.IsCudaAvailable ? Device.Cuda() : Device.Cpu;
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--cuda" or "--gpu": device = Device.Cuda(); break;
        case "--cpu": device = Device.Cpu; break;
        case "--int8": int8 = true; break;
        case "--kv8": kv8 = true; break;
        case "--no-think": noThink = true; break;
        case "--context": context = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
        case "--folder": folderOverride = args[++i]; break;
        default: positional.Add(args[i]); break;
    }
}

if (positional.Count < 2 || positional[0] is not ("info" or "chat" or "check"))
{
    Console.WriteLine("usage: info <folder> | chat <folder> | check <reference.json>   [--cuda|--cpu] [--int8] [--kv8] [--context N] [--folder F] [--no-think]");
    return 1;
}

Device.Default = device;
var cacheFormat = kv8 ? KeyValueFormat.Int8 : KeyValueFormat.Float32;

PretrainedModel Load(string folder)
{
    var watch = Stopwatch.StartNew();
    var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device, Int8 = int8, MaxPositions = context });
    Console.WriteLine($"Loaded {model.Config["architectures"]?[0]} from {folder} in {watch.Elapsed.TotalSeconds:F1} s on {device}{(int8 ? ", int8 weights" : "")}");
    Console.WriteLine($"  {model.Spec.ParameterCount / 1e6:F0}M parameters, {model.Spec.Layers} layers, dim {model.Spec.Dim}, heads {model.Spec.Heads}/{model.Spec.KvHeads}, "
                      + $"vocabulary {model.Spec.Vocabulary}, context {model.MaxPositions}");
    foreach (var note in model.Notes)
    {
        Console.WriteLine($"  note: {note}");
    }

    return model;
}

switch (positional[0])
{
    case "info":
    {
        using var model = Load(positional[1]);
        Console.WriteLine(model.Spec.ToJson());
        Console.WriteLine(model.ChatTemplate is { } t ? $"chat template: {t.Source.Length} characters, stops [{string.Join(", ", t.StopSequences)}]" : "no chat template");
        return 0;
    }

    case "chat":
    {
        using var model = Load(positional[1]);
        var chat = model.CreateChat(cacheFormat, context);
        var messages = new List<ChatMessage>();
        Console.WriteLine("Type a message (empty line to quit, /system <text> to set the system prompt).");
        while (Console.ReadLine() is { Length: > 0 } line)
        {
            if (line.StartsWith("/system ", StringComparison.Ordinal))
            {
                messages.RemoveAll(m => m.Role == "system");
                messages.Insert(0, new ChatMessage("system", line[8..]));
                continue;
            }

            messages.Add(new ChatMessage("user", line));
            var options = new GenerationOptions { Temperature = 0.6f, TopK = 20, TopP = 0.95f, RepeatPenalty = 1f, NumCtx = context };
            ChatMessage? reply = null;
            bool thinking = false;
            foreach (var chunk in chat.Stream(new ChatRequest(messages, Think: noThink ? false : null, Options: options)))
            {
                if (chunk.Delta.Thinking.Length > 0)
                {
                    if (!thinking)
                    {
                        Console.ForegroundColor = ConsoleColor.DarkGray;
                        thinking = true;
                    }

                    Console.Write(chunk.Delta.Thinking);
                }

                if (chunk.Delta.Content.Length > 0)
                {
                    if (thinking)
                    {
                        Console.ResetColor();
                        Console.WriteLine();
                        thinking = false;
                    }

                    Console.Write(chunk.Delta.Content);
                }

                foreach (var call in chunk.Delta.ToolCalls)
                {
                    Console.Write($"\n[tool call] {call.Name}({call.Arguments.ToJsonString()})");
                }

                if (chunk.Done)
                {
                    reply = chunk.Message;
                    Console.ResetColor();
                    Console.WriteLine($"\n[{chunk.Stats?.GeneratedTokens} tokens, {chunk.Stats?.TokensPerSecond:F1} tok/s]");
                }
            }

            if (reply is not null)
            {
                messages.Add(reply);
            }
        }

        return 0;
    }

    default:
        return Check(positional[1]);
}

int Check(string referencePath)
{
    var reference = JsonNode.Parse(File.ReadAllText(referencePath))!.AsObject();
    using var model = Load(folderOverride ?? (string)reference["folder"]!);
    var tokenizer = (BpeTokenizer)(model.Tokenizer ?? throw new InvalidOperationException("The model has no tokenizer.json."));
    int failures = 0;

    // 1. Tokenizer: the same ids as transformers, and decoding gives the text back.
    Console.WriteLine("\nTokenizer");
    foreach (var item in reference["texts"]!.AsArray())
    {
        string text = (string)item!["text"]!;
        int[] expected = [.. item["ids"]!.AsArray().Select(i => (int)i!)];
        var actual = tokenizer.Encode(text);
        bool same = actual.SequenceEqual(expected);
        string decoded = tokenizer.Decode(expected), expectedDecoded = (string?)item["decoded"] ?? text;
        bool decodes = decoded == expectedDecoded;
        Console.WriteLine($"  {(same && decodes ? "ok  " : "DIFF")} {expected.Length,4} ids{(decodes ? "" : $"  decoded differently: {Shorten(decoded)}")}  {Shorten(text)}");
        failures += same && decodes ? 0 : 1;
        if (!same)
        {
            int at = Enumerable.Range(0, Math.Min(actual.Count, expected.Length)).FirstOrDefault(i => actual[i] != expected[i], Math.Min(actual.Count, expected.Length));
            Console.WriteLine($"       first difference at {at}: expected [{string.Join(", ", expected.Skip(at).Take(4).Select(tokenizer.TokenOf))}], "
                              + $"got [{string.Join(", ", actual.Skip(at).Take(4).Select(tokenizer.TokenOf))}]");
        }
    }

    // 2. Chat templates: the model's own template renders conversations (tools, tool calls, reasoning) identically.
    Console.WriteLine("\nChat template");
    var tools = reference["tools"]!.AsArray().Select(t => new ToolDefinition((string)t!["name"]!, (string?)t["description"], t["parameters"]?.DeepClone())).ToList();
    foreach (var chat in reference["chats"]!.AsArray())
    {
        var messages = chat!["messages"]!.AsArray().Select(m => new ChatMessage((string)m!["role"]!, (string)m["content"]!, (string?)m["thinking"],
            m["tool_calls"]?.AsArray().Select(c => new ToolCall((string)c!["name"]!, (JsonObject)c["arguments"]!.DeepClone())).ToList(),
            (string?)m["tool_name"])).ToList();
        var chatTools = chat["tools"]!.AsArray().Count > 0 ? tools : [];
        bool? think = chat["think"] is JsonValue v ? (bool)v : null;
        bool generationPrompt = (bool)chat["add_generation_prompt"]!;
        string expected = (string)chat["rendered"]!;
        string actual;
        try
        {
            actual = model.ChatTemplate?.Render(messages, chatTools, think, generationPrompt) ?? "ERROR: no chat template";
        }
        catch (InvalidOperationException ex)
        {
            actual = "ERROR: " + ex.Message;
        }

        bool same = actual == expected || actual.StartsWith("ERROR", StringComparison.Ordinal) && expected.StartsWith("ERROR", StringComparison.Ordinal);
        Console.WriteLine($"  {(same ? "ok  " : "DIFF")} {messages.Count} messages, {chatTools.Count} tools, think {think?.ToString() ?? "default"}, "
                          + $"generation prompt {generationPrompt}: {expected.Length} characters");
        if (!same)
        {
            failures++;
            int at = Enumerable.Range(0, Math.Min(actual.Length, expected.Length)).FirstOrDefault(i => actual[i] != expected[i], Math.Min(actual.Length, expected.Length));
            Console.WriteLine($"       first difference at {at}:\n       expected …{Shorten(expected[Math.Max(0, at - 30)..])}\n       got      …{Shorten(actual[Math.Max(0, at - 30)..])}");
        }
    }

    // 3. Logits for the same ids, and 4. greedy continuations (through the generator and its KV cache).
    Console.WriteLine("\nLogits and greedy output");
    foreach (var run in reference["runs"]!.AsArray())
    {
        int[] ids = [.. run!["ids"]!.AsArray().Select(i => (int)i!)];
        float[] expected = [.. run["logits"]!.AsArray().Select(x => (float)x!)];
        using var input = Tensor.From([.. ids.Select(i => (float)i)], [1, ids.Length], device);
        var watch = Stopwatch.StartNew();
        float[] all;
        using (var logits = model.Network.Predict(input))
        {
            all = logits.ToArray();
        }

        var actual = all.AsSpan(all.Length - expected.Length).ToArray();
        float maxDiff = 0, scale = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            maxDiff = Math.Max(maxDiff, Math.Abs(actual[i] - expected[i]));
            scale = Math.Max(scale, Math.Abs(expected[i]));
        }

        var topExpected = TopK(expected, 10);
        var topActual = TopK(actual, 10);
        bool top1 = topExpected[0] == topActual[0];
        int overlap = topExpected.Intersect(topActual).Count();

        bool encodes = tokenizer.Encode((string)run["prompt"]!).SequenceEqual(ids);
        int[] greedyExpected = [.. run["generated"]!.AsArray().Select(i => (int)i!)];
        var generator = model.CreateGenerator(cacheFormat, context);
        var options = new GenerationOptions { TopK = 1, Temperature = 1f, RepeatPenalty = 1f, NumPredict = greedyExpected.Length, NumCtx = context, Seed = 0 };
        var (text, _, stats) = generator.Generate((string)run["prompt"]!, options);
        string expectedText = (string)run["generated_text"]!;
        int agree = 0;
        while (agree < Math.Min(text.Length, expectedText.Length) && text[agree] == expectedText[agree])
        {
            agree++;
        }

        bool greedySame = text == expectedText;
        bool ok = encodes && top1 && (int8 || maxDiff <= 2e-3f * Math.Max(1f, scale)) && (greedySame || int8 || kv8);
        failures += ok ? 0 : 1;
        Console.WriteLine($"  {(ok ? "ok  " : "DIFF")} {ids.Length,4} ids: max |Δlogit| {maxDiff:G3} (largest logit {scale:F1}), top-1 {(top1 ? "same" : "DIFFERENT")}, "
                          + $"top-10 overlap {overlap}/10, forward {watch.Elapsed.TotalMilliseconds:F0} ms; greedy {(greedySame ? "identical" : $"first {agree} of {expectedText.Length} characters agree")} "
                          + $"({stats.TokensPerSecond:F1} tok/s){(encodes ? "" : "; the prompt text encodes to different ids")}");
        if (!greedySame)
        {
            Console.WriteLine($"       transformers: {Shorten(expectedText)}\n       NeuralSharp:  {Shorten(text)}");
        }
    }

    Console.WriteLine(failures == 0 ? "\nEverything matches transformers." : $"\n{failures} check(s) differ.");
    return failures == 0 ? 0 : 2;
}

static int[] TopK(float[] values, int k) =>
    [.. values.Select((v, i) => (v, i)).OrderByDescending(p => p.v).Take(k).Select(p => p.i)];

static string Shorten(string text)
{
    string flat = text.Replace("\r", "\\r").Replace("\n", "\\n").Replace("\t", "\\t");
    return flat.Length > 100 ? flat[..100] + "…" : flat;
}

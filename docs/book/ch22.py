"""Chapter 22 — The Generation Layer: Text, Chat, Tools and Model Hosting."""
from gen import *

PART = "IV"


def pipeline_svg():
    w, h = 480, 150
    boxes = [
        (8, "messages", "tools, think"), (100, "ChatTemplate", "→ prompt text"), (192, "ITokenizer", "text ⇄ ids"),
        (284, "TextGenerator", "KV cache, sampler"), (376, "ChatOutputParser", "content, thinking"),
    ]
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">']
    for x, name, sub in boxes:
        p.append(f'<rect x="{x}" y="40" width="88" height="44" rx="4" fill="#e6f2ef" stroke="#0f6b5c" stroke-width="0.8"/>')
        p.append(svg_text(x + 44, 56, name, 8.2, "#0f6b5c"))
        p.append(svg_text(x + 44, 70, sub, 7.6, "#56606a"))
    p.append(svg_text(420, 82, "tool calls", 7.6, "#56606a"))
    for x in (96, 188, 280, 372):
        p.append(f'<path d="M{x} 62 L{x + 4} 62" stroke="#1a1f23" stroke-width="1"/><path d="M{x + 4} 59 L{x + 7} 62 L{x + 4} 65 Z" fill="#1a1f23"/>')
    p.append(svg_text(240, 20, "ChatGenerator: one request in, a stream of ChatChunks out", 8.4, "#1a1f23"))
    p.append(f'<rect x="4" y="30" width="472" height="74" rx="6" fill="none" stroke="#56606a" stroke-width="0.6" stroke-dasharray="3 2"/>')
    p.append(svg_text(240, 124, "ModelHost keeps the model loaded between requests (keep_alive);", 7.6, "#56606a"))
    p.append(svg_text(240, 138, "an HTTP endpoint maps a JSON request onto ChatRequest + GenerationOptions", 7.6, "#56606a"))
    p.append("</svg>")
    return "".join(p)


FILTERS = """
    string[] words = ["the", "a", "one", "this", "every", "zebra"];
    float[] scores = [3.0f, 2.2f, 1.6f, 0.5f, -0.5f, -1.5f];          // the model's logits for one position
    const int Rows = 20000;                                          // 20,000 independent draws in one call
    var flat = new float[Rows * words.Length];
    for (int r = 0; r < Rows; r++) scores.CopyTo(flat, r * words.Length);

    void Run(string name, Action<TokenSampler> set, int[]? history = null)
    {
        using var logits = Tensor.From(flat, [Rows, words.Length], device);
        using var sampler = new TokenSampler(device, Rows, words.Length, maxSteps: 1) { Seed = 42 };
        set(sampler);
        if (history is not null) sampler.SetHistory(history);           // "recent tokens" for the penalties
        sampler.Sample(logits);
        var ids = sampler.Read(0, 1)[0].Select(t => t.Id).ToArray();
        Console.WriteLine($"{name,-26}" + string.Join(" ",
            Enumerable.Range(0, words.Length).Select(w => $"{ids.Count(i => i == w) * 100.0 / Rows,5:F1}")));
    }

    Run("temperature 1 (no filter)", s => { });
    Run("temperature 0.5", s => s.Temperature = 0.5f);
    Run("temperature 1.5", s => s.Temperature = 1.5f);
    Run("top_k 2", s => s.TopK = 2);
    Run("top_p 0.9", s => s.TopP = 0.9f);
    Run("min_p 0.05", s => s.MinP = 0.05f);
    Run("repeat_penalty 1.5", s => s.RepeatPenalty = 1.5f, [0, 0, 0, 1]);      // history: the the the a
    Run("presence_penalty 1", s => s.PresencePenalty = 1f, [0, 0, 0, 1]);
    Run("frequency_penalty 0.5", s => s.FrequencyPenalty = 0.5f, [0, 0, 0, 1]);
"""

FILTERS_OUT = """
    setting                     the     a   one  this every zebra
    temperature 1 (no filter)  54.9  25.1  13.2   4.4   1.7   0.6
    temperature 0.5            79.0  15.6   4.7   0.6   0.0   0.0
    temperature 1.5            43.0  25.4  17.1   8.1   4.1   2.2
    top_k 2                    69.1  30.9   0.0   0.0   0.0   0.0
    top_p 0.9                  58.9  26.7  14.5   0.0   0.0   0.0
    min_p 0.05                 56.3  25.5  13.7   4.5   0.0   0.0
    repeat_penalty 1.5         38.7  22.5  26.0   8.6   3.2   1.1
    presence_penalty 1         40.7  18.2  27.4   9.1   3.3   1.2
    frequency_penalty 0.5      25.8  31.5  28.5   9.5   3.4   1.2
"""

STREAM = """
    using var gpt = CharGpt.Load("models/transformer.weights", device);   // the grammar GPT of Chapter 35
    gpt.Model.Eval();
    var generator = new TextGenerator(gpt.Model, new CharTokenizer(gpt.Config.Vocabulary), gpt.Config.Context);
    var options = new GenerationOptions { Seed = 3, Temperature = 0.7f, NumPredict = 120, Stop = ["\\n"] };

    foreach (var chunk in generator.Stream("the old dog ", options))
    {
        Console.Write(chunk.Text + (chunk.Done ? "" : "|"));                  // '|' marks chunk boundaries
        if (chunk.Done)
        {
            var s = chunk.Stats!;
            Console.WriteLine($"\\ndone_reason {chunk.DoneReason}; prompt {s.PromptTokens} tokens in " +
                $"{s.PromptDuration.TotalMilliseconds:F1} ms; {s.GeneratedTokens} tokens in " +
                $"{s.GenerationDuration.TotalMilliseconds:F1} ms ({s.TokensPerSecond:F0}/s); resets {s.ContextResets}");
        }
    }
""".replace("Chapter 35", "CHAPTER_GPT")


def build():
    return page(
        chapter_open(
            "textgen",
            "The previous chapter built the fast machinery: a KV cache, an on-device sampler and replayable decoding "
            "steps. The <code>NeuralSharp.Generation</code> namespace turns that machinery into what applications ask "
            "for: text in, text out, streamed as it is produced, with the sampling and length options that local LLM "
            "servers use; a chat format with system prompts, a separate thinking channel and tool calls; and a host "
            "that keeps models loaded between requests. The repository's GPT Web API uses all of it to answer "
            "Ollama-style <code>/api/chat</code> requests.",
            "<code>ITokenizer</code>: <code>CharTokenizer</code> (one token per character) and <code>WordTokenizer</code> (words, numbers, punctuation, special tokens).",
            "<code>GenerationOptions</code>: temperature, top_k, top_p, min_p, repeat_penalty, repeat_last_n, presence and frequency penalties, seed, num_ctx, num_predict, stop.",
            "<code>TextGenerator.Stream(prompt, options)</code> yields text chunks, then a final chunk with <code>DoneReason</code> (<code>stop</code> or <code>length</code>) and timing statistics.",
            "<code>ChatGenerator</code> + <code>ChatMLTemplate</code> + <code>ChatOutputParser</code>: messages and tools in; content, thinking and tool calls out.",
            "<code>ModelHost</code> + <code>KeepAlive</code>: load on first use, unload after the keep-alive time (<code>\"30m\"</code>, <code>0</code>, <code>-1</code>).",
        ),
        h2("22.1 The pieces"),
        diagram("Figure 22.1 — From a chat request to streamed output", pipeline_svg(),
                "Each box is a class you can use on its own; <code>ChatGenerator</code> wires them together."),
        reftable(["Type", "Role", "Key members"], [
            ["<code>ITokenizer</code>", "Text ⇄ token ids", "<code>VocabularySize</code>, <code>Encode(text)</code>, <code>Decode(ids)</code>"],
            ["<code>GenerationOptions</code>", "Sampling and length settings (a record)", "See Table 22.2"],
            ["<code>TextGenerator(model, tokenizer, contextLength)</code>", "Prompt → streamed continuation", "<code>Stream</code>, <code>Generate</code>, <code>MaxTokens</code> (4,096)"],
            ["<code>GenerationChunk</code>", "One piece of streamed text", "<code>Text</code>, <code>Done</code>, <code>DoneReason</code>, <code>Stats</code>"],
            ["<code>GenerationStats</code>", "Timing of one generation", "<code>PromptTokens</code>, <code>PromptDuration</code>, <code>GeneratedTokens</code>, <code>GenerationDuration</code>, <code>TotalDuration</code>, <code>ContextResets</code>, <code>TokensPerSecond</code>"],
            ["<code>ChatMessage</code>, <code>ToolDefinition</code>, <code>ToolCall</code>", "Chat data (records)", "<code>Role</code>, <code>Content</code>, <code>Thinking</code>, <code>ToolCalls</code>, <code>ToolName</code>"],
            ["<code>ChatTemplate</code> / <code>ChatMLTemplate</code>", "Messages → prompt text", "<code>Render</code>, <code>StopSequences</code>, <code>ThinkTags</code>, <code>ToolCallTags</code>"],
            ["<code>ChatOutputParser</code>", "Raw text → content, thinking, tool calls", "<code>Feed(text)</code>, <code>Finish()</code>"],
            ["<code>ChatGenerator</code>", "The whole chat turn", "<code>Stream(request)</code>, <code>Chat(request)</code>, <code>RenderPrompt</code>"],
            ["<code>ModelHost&lt;T&gt;</code>, <code>KeepAlive</code>", "Keep models loaded between requests", "<code>Acquire(name, keepAlive)</code>, <code>Loaded</code>, <code>Unload</code>, <code>KeepAlive.Parse</code>"],
        ], caption="Table 22.1 — The Generation namespace"),
        h2("22.2 Tokenizers"),
        para("A tokenizer fixes what one token is. The GPT of " + ch("gpt") + " reads characters: a tiny vocabulary "
             "(29 symbols) but long sequences. <code>WordTokenizer</code> reads words: sequences are about five times "
             "shorter, so a model with the same context sees much more text, at the cost of a larger vocabulary and of "
             "words it has never seen (they become <code>&lt;unk&gt;</code>). Special tokens in angle brackets, such as "
             "<code>&lt;sum&gt;</code> or <code>&lt;end&gt;</code>, are always single tokens (glossary <b>Tokenizer</b>, "
             "<b>Unknown token</b>)."),
        mex("a word vocabulary from example texts",
            "<code>FromTexts</code> puts the special tokens first and then the words by frequency.",
            """
            var tokenizer = WordTokenizer.FromTexts(
                ["the lions beat the eagles 3 to 1 .", "rain is likely , bring a coat !"],
                specials: ["<pad>", "<sum>", "<end>"]);

            var ids = tokenizer.Encode("The Lions beat the wolves 3 to 1. <sum> <end>");
            Console.WriteLine(string.Join(",", ids));
            Console.WriteLine($"'{tokenizer.Decode(ids)}'");
            string pieces = string.Concat(ids.Select(id => tokenizer.Decode([id])));
            Console.WriteLine(pieces == tokenizer.Decode(ids));       // decoding token by token gives the same text
            """,
            out="""
            3,5,6,3,19,9,11,10,8,1,2
            ' the lions beat the <unk> 3 to 1. <sum> <end>'
            True
            """,
            after="Text is lower-cased (by default), \"wolves\" was not in the example texts, and a space is put before "
                  "every token except closing punctuation. That last rule makes decoding <b>streaming-safe</b>: the "
                  "generator decodes each chunk separately, and the pieces join into exactly the whole text."),
        cpugpu("tokenizers run on the CPU; the model runs where its weights are",
               """
               var generator = new TextGenerator(model, tokenizer, contextLength: 96);
               // model built with Device.Cpu: prompt ids, sampling and cache all on the CPU
               """,
               """
               model.To(Device.Cuda());                                // or build it on the GPU
               var generator = new TextGenerator(model, tokenizer, contextLength: 96);
               // Encode/Decode stay on the CPU; ids are uploaded once per prompt, sampled ids read per chunk
               """),
        h2("22.3 Sampling options, measured"),
        para("Every option changes the scores (logits) of the vocabulary before one token is drawn. The worked example "
             "draws 20,000 tokens from one fixed set of six scores with each option, so the shares show exactly what each "
             "option does. Without any option the shares follow the softmax of the scores (glossary <b>Softmax</b>; the "
             "exponential function is in the Pre-Calc volume)."),
        mex("nine sampler settings on the same scores", None, FILTERS, out=FILTERS_OUT,
            after="Temperature reshapes the whole distribution. The three filters remove the unlikely tail in different "
                  "ways and renormalize what is left: <code>top_k 2</code> keeps two tokens, <code>top_p 0.9</code> keeps the "
                  "fewest tokens reaching 90 % (55 + 25 = 80 % is not enough, so three), and <code>min_p 0.05</code> keeps "
                  "every token at least 5 % as likely as the best (four). The penalties look at recent tokens: \"the\" "
                  "appeared three times and \"a\" once, so both lose share and \"one\" gains."),
        deriv("What each option does to a score x (before the softmax)", [
            "Temperature T: x → x / T. Below 1 sharpens, above 1 flattens; 0 means greedy (always the best token).",
            "top_k k: only the k highest scores stay (0 = all).",
            "top_p p: sort by probability and keep the shortest prefix whose probabilities add up to at least p (1 = off).",
            "min_p m: keep tokens with probability ≥ m × (probability of the best token) (0 = off). Unlike top_p, it adapts "
            "to how confident the model is.",
            "repeat_penalty r, for a token seen in the last <code>repeat_last_n</code> tokens: x → x / r if x &gt; 0, else x × r (1 = off).",
            "presence_penalty a and frequency_penalty b, for a token seen c times in that window: x → x − a − b × c.",
        ]),
        reftable(["Option (JSON name)", "Property", "Default", "Typical"], [
            ["<code>temperature</code>", "<code>Temperature</code>", "0.8", "0 (exact answers) to 1.2 (creative)"],
            ["<code>top_k</code>", "<code>TopK</code>", "40", "20–100; 1 = greedy"],
            ["<code>top_p</code>", "<code>TopP</code>", "0.9", "0.8–0.95"],
            ["<code>min_p</code>", "<code>MinP</code>", "0", "0.02–0.1"],
            ["<code>repeat_penalty</code>, <code>repeat_last_n</code>", "<code>RepeatPenalty</code>, <code>RepeatLastN</code>", "1.1, 64", "1.0–1.3; 64–256 tokens"],
            ["<code>presence_penalty</code>, <code>frequency_penalty</code>", "<code>PresencePenalty</code>, <code>FrequencyPenalty</code>", "0, 0", "0–1"],
            ["<code>seed</code>", "<code>Seed</code>", "random", "Fixed for reproducible output (same on CPU and GPU)"],
            ["<code>num_ctx</code>", "<code>NumCtx</code>", "2,048", "Capped by the model's context length"],
            ["<code>num_predict</code>", "<code>NumPredict</code>", "−1 (up to 4,096)", "The answer's maximum length"],
            ["<code>stop</code>", "<code>Stop</code>", "none", "Strings that end generation (not returned)"],
        ], caption="Table 22.2 — GenerationOptions"),
        para("On the GPU all of this runs inside the sampling kernel: the recent-token history is a device buffer that "
             "the kernel appends to after each draw, so penalties need no copy to the CPU and a recorded decoding step "
             "(" + ch("generation") + ") replays them unchanged. The top-p cut-off is found by bisection on the score "
             "(24 halvings of an interval), which needs no sorting."),
        h2("22.4 TextGenerator: streaming, stop sequences and limits"),
        mex("streaming a continuation", "The grammar GPT, a stop sequence at the end of the line, chunks of 8 tokens.",
            STREAM.replace("CHAPTER_GPT", ch("gpt")),
            out="""
            stream: an old m|ap under| the bri|dge.|
            done_reason stop; prompt 12 tokens in 68.3 ms; 28 tokens in 30.8 ms (910/s); resets 0
            """,
            after="Chunks arrive as the sampler produces them (<code>ChunkSize</code> tokens between reads from the "
                  "device). The last chunk is shorter because text that could be the start of a stop sequence is held "
                  "back until it is certain it is not; the stop text itself is never returned."),
        output("""
            num_predict 40, no stop:   [length, 40 tokens, resets 0]  "an old map under the bridge.\\nthe little "
            stop ".":                  [stop, 27 tokens, resets 0]    "an old map under the bridge"
            num_ctx 16, 200 tokens:    [length, 200 tokens, resets 22] "an old map under the bridge.\\nthe little robot found a
                                       tiny garden and then found the broken clock.\\nthe young pilot built a tiny garden ..."
            greedy (temperature 0):    [length, 60 tokens, resets 1]  "an old map and then found a shiny key. the little robot buil"
            greedy + repeat_penalty 1.3: [length, 60 tokens, resets 1] "an old map.\\nthe young pilot built a shiny key and then forgo"
            cache + graph        1319 tokens/s   first 30: an old map under the bridge.\\nt
            cache, no graph      1021 tokens/s   first 30: an old map under the bridge.\\nt
            no cache              316 tokens/s   first 30: an old map under the bridge.\\nt
            """, caption="Same prompt and seed with other options (CPU)"),
        para("<code>DoneReason</code> is <code>stop</code> when a stop sequence ended the text and <code>length</code> when "
             "<code>num_predict</code> did. With <code>num_ctx 16</code> the window fills after 16 tokens; the generator then "
             "keeps the most recent half and re-reads it (a <b>context reset</b>, counted in the statistics), so it can "
             "generate far beyond its context, remembering only the recent text. A prompt longer than the window keeps "
             "its last <code>num_ctx − 1</code> tokens. The three execution modes give identical text, at very different "
             "speeds."),
        trap("sharing one generator between concurrent requests",
             "<p>A generation owns the model's KV caches while it runs, so two generations on the same model at the same "
             "time would overwrite each other's caches. Serialize them (a <code>SemaphoreSlim(1)</code> per model, as the "
             "Web API does) or load one model per concurrent request.</p>"),
        h2("22.5 Chat: templates, thinking and tools"),
        para("A chat model is a language model trained on conversations written in a fixed text format, the <b>chat "
             "template</b>. <code>ChatMLTemplate</code> writes the widely used ChatML format: each message between "
             "<code>&lt;|im_start|&gt;role</code> and <code>&lt;|im_end|&gt;</code>; tool definitions as JSON inside "
             "<code>&lt;tools&gt;</code> in the system message; the model's reasoning between <code>&lt;think&gt;</code> tags; "
             "tool calls as JSON between <code>&lt;tool_call&gt;</code> tags; tool results as <code>tool</code> messages "
             "wrapped in <code>&lt;tool_response&gt;</code>. The template also supplies the stop sequences that end the "
             "assistant's turn."),
        snippet(CHAT_REQUEST, caption="A chat request with a system prompt, thinking and one tool"),
        output(CHAT_PROMPT, caption="The prompt text the template renders (the tool JSON is shortened here)"),
        para("<code>think</code> has three states: <code>true</code> lets the model reason first and returns the reasoning "
             "in <code>Message.Thinking</code>; <code>false</code> closes an empty think block in the prompt so the model "
             "answers directly; <code>null</code> leaves the choice to the model. The <code>ChatOutputParser</code> reads "
             "the generated text as it streams and routes it: text inside the think tags to <code>Delta.Thinking</code>, "
             "complete tool calls (parsed JSON) to <code>Delta.ToolCalls</code>, everything else to "
             "<code>Delta.Content</code>. It works for any chunking, even one character at a time, because it holds back "
             "text that could be the start of a tag."),
        CHAT_DEMO_STATIONS,
        honestbox("What a small model can and cannot do here",
                  "<p>The chat model above is a character-level GPT trained for about an hour on the CPU on synthetic "
                  "conversations; it learned the format (thinking, a tool call with the right JSON, an answer citing the "
                  "tool result) but not to copy facts from the tool result (Table 22.5). The Generation layer is the same for a real chat model: what changes is "
                  "the size of the network and the data it was trained on. The model decides <i>whether</i> to call a "
                  "tool; your code runs the tool and sends the result back as a <code>tool</code> message.</p>"),
        h2("22.6 Keeping models loaded"),
        para("Loading weights takes time (100 ms for the small GPT from disk; seconds for large models on a GPU), and a "
             "loaded model holds memory. <code>ModelHost</code> loads a model on first use, lends it out with a "
             "<b>lease</b>, and unloads it once the lease's keep-alive time has passed without use, freeing its device "
             "memory. Requests choose the keep-alive with the same values Ollama accepts."),
        mex("load once, reuse, expire",
            None,
            """
            using var host = new ModelHost<CharGpt>(name => CharGpt.Load($"models/{name}.weights", device),
                defaultKeepAlive: KeepAlive.Parse("5m"));

            using (var lease = host.Acquire("grammar", KeepAlive.Parse("2s")))
            {
                Console.WriteLine($"first request: load took {lease.LoadDuration.TotalMilliseconds:F1} ms");
                // ... generate with lease.Model ...
            }                                              // the 2 s countdown starts when the lease ends
            using (var lease = host.Acquire("grammar", KeepAlive.Parse("2s")))
                Console.WriteLine($"second request: load took {lease.LoadDuration.TotalMilliseconds:F1} ms");

            Thread.Sleep(3500);                            // host.Loaded is now empty: the model was unloaded
            using (host.Acquire("grammar", KeepAlive.Parse("-1"))) { }     // loads again, then stays forever
            using (host.Acquire("grammar", KeepAlive.Parse("0"))) { }      // unloads as soon as the lease ends
            """,
            out="""
            first request: load took 100.3 ms
            during request               loaded: grammar (users 1, expires never/in use)  loads 1
            after request (keep 2s)      loaded: grammar (users 0, expires in 2 s)  loads 1
            second request: load took 0.0 ms
            3.5 s later                  loaded:   loads 1
            after keep_alive -1          loaded: grammar (users 0, expires never/in use)  loads 2
            after keep_alive 0           loaded:   loads 2
            """),
        reftable(["keep_alive value", "Meaning", "<code>KeepAlive.Parse</code>"], [
            ["<code>\"30m\"</code>, <code>\"1h30m\"</code>, <code>\"250ms\"</code>", "Duration with units (ns, us, ms, s, m, h)", "00:30:00, 01:30:00, 00:00:00.25"],
            ["<code>90</code> or <code>\"90\"</code>", "Seconds", "00:01:30"],
            ["<code>0</code>", "Unload as soon as the request ends", "00:00:00"],
            ["<code>-1</code> (any negative)", "Keep loaded forever", "null"],
            ["missing", "The host's default (the Web API uses 5 minutes)", "—"],
        ], caption="Table 22.3 — keep_alive"),
        h2("22.7 An Ollama-compatible chat endpoint"),
        para("The GPT Web API (" + ch("webapi") + ") maps <code>POST /api/chat</code> onto the classes above, using "
             "Ollama's request and response format, so tools and clients written for Ollama can talk to a NeuralSharp "
             "model. The request's <code>options</code> object becomes <code>GenerationOptions</code> (unknown keys are "
             "ignored), <code>think</code> and <code>tools</code> go to the <code>ChatRequest</code>, "
             "<code>keep_alive</code> to the lease, and the response streams as newline-delimited JSON (one object per "
             "line, <code>application/x-ndjson</code>), ending with a line that has <code>\"done\": true</code>, the "
             "<code>done_reason</code> and the timings in nanoseconds."),
        API_STATIONS,
        reftable(["Endpoint", "Returns"], [
            ["<code>POST /api/chat</code>", "Streamed NDJSON (default) or one JSON object with <code>\"stream\": false</code>; 400 for invalid requests, 404 when no model is available"],
            ["<code>GET /api/tags</code>", "The served model with its size and details"],
            ["<code>GET /api/ps</code>", "Loaded models with <code>expires_at</code> (from keep_alive)"],
            ["<code>GET /api/version</code>", "The API version"],
        ], caption="Table 22.4 — The Ollama-style endpoints"),
        practice([
            (1, "Which option makes generation deterministic without fixing a seed?",
             "<code>Temperature = 0</code> (or <code>TopK = 1</code>): the best token is always chosen."),
            (1, "A request has <code>num_predict 50</code> and <code>stop [\".\"]</code>; the model writes a full stop after 20 tokens. What are <code>done_reason</code> and the returned text?",
             "<code>stop</code>; the text up to, but not including, the full stop."),
            (2, "Why can <code>min_p</code> keep more tokens than <code>top_p</code> on a flat distribution and fewer on a peaked one?",
             "Its threshold is relative to the best token: when no token dominates, many tokens are within the fraction of the best; "
             "when one dominates, few are. <code>top_p</code> instead always collects a fixed share of probability."),
            (2, "Write a tool loop: send a request with tools, run each returned tool call, append the results and ask again until the answer has no tool calls.",
             "Loop: <code>var reply = chat.Chat(request).Message!</code>; if <code>reply.ToolCalls</code> is empty, stop; otherwise add "
             "<code>reply</code> to the messages, and for each call add <code>new ChatMessage(\"tool\", Run(call), ToolName: call.Name)</code>; "
             "repeat with the extended message list (set a maximum number of rounds)."),
            (3, "Serve two different models from one Web API with a 10-minute keep-alive each and at most one generation per model at a time.",
             "One <code>ModelHost</code> whose load function builds either model by name; per model a <code>SemaphoreSlim(1)</code> "
             "(for example in a dictionary); per request <code>Acquire(name, keepAlive ?? 10 min)</code>, wait on that model's "
             "semaphore, generate, release both."),
        ], PART),
        footer("Tokenizer", "Unknown token", "Temperature", "Top-k sampling", "Top-p sampling", "Min-p sampling",
               "Repetition penalty", "Stop sequence", "Context window", "Streaming", "Chat template", "Tool calling",
               "Thinking", "Keep-alive", "NDJSON", "Hallucination"),
    )


# Measured: the chat model saved by the Transformer sample's --chat mode, and the GptApi sample serving it.
CHAT_REQUEST = """
    var webFetch = new ToolDefinition("web_fetch", "Fetch a page from the allowlisted search results.",
        JsonNode.Parse(\"\"\"{"type":"object","properties":{"url":{"type":"string",
            "description":"Absolute http(s) URL from the allowlist."}},"required":["url"]}\"\"\"));
    var messages = new List<ChatMessage>
    {
        new("system", "You are a helpful assistant. Cite sources as [1], [2] when a research pack is present."),
        new("user", "What is the latest Ollama version?"),
    };
    var options = new GenerationOptions { Temperature = 1f, TopK = 20, TopP = 0.95f, MinP = 0f,
        RepeatPenalty = 1f, NumCtx = 4096, NumPredict = 300, Seed = 7 };

    var chat = new ChatGenerator(new TextGenerator(gpt.Model, new CharTokenizer(gpt.Config.Vocabulary), gpt.Config.Context));
    var request = new ChatRequest(messages, [webFetch], Think: true, Options: options);
    Console.WriteLine(chat.RenderPrompt(request));
"""

CHAT_PROMPT = """
    <|im_start|>system
    You are a helpful assistant. Cite sources as [1], [2] when a research pack is present.

    # Tools

    You may call one or more functions. Function signatures:
    <tools>
    {"type":"function","function":{"name":"web_fetch","description":"Fetch a page from the allowlisted search ...}}}
    </tools>

    For each call, return <tool_call>{"name": <function-name>, "arguments": <args-json-object>}</tool_call><|im_end|>
    <|im_start|>user
    What is the latest Ollama version?<|im_end|>
    <|im_start|>assistant
"""

CHAT_DEMO_STATIONS = mex("two turns with a tool",
    "The model is the character-level chat model that the Transformer sample trains with <code>--chat true</code> "
    "(350,380 parameters, context 256, trained 3,244 s on synthetic ChatML conversations). The loop streams each turn, "
    "runs the tool call by hand, and sends the result back as a <code>tool</code> message.",
    """
    for (int turn = 1; turn <= 2; turn++)
    {
        ChatMessage? final = null;
        foreach (var chunk in chat.Stream(new ChatRequest(messages, [webFetch], Think: true, Options: options)))
        {
            if (chunk.Delta.Thinking.Length > 0) Console.WriteLine($"thinking: {JsonSerializer.Serialize(chunk.Delta.Thinking)}");
            if (chunk.Delta.Content.Length > 0) Console.WriteLine($"content:  {JsonSerializer.Serialize(chunk.Delta.Content)}");
            foreach (var c in chunk.Delta.ToolCalls) Console.WriteLine($"tool_call: {c.Name} {c.Arguments.ToJsonString()}");
            if (chunk.Done) final = chunk.Message;                     // Content, Thinking and ToolCalls together
        }
        messages.Add(final!);
        if (final!.ToolCalls is not { Count: > 0 } calls) break;
        string url = calls[0].Arguments["url"]!.GetValue<string>();
        string page = $"[1] {url}: Ollama 0.12.3 is the latest release.";   // your code runs the tool
        messages.Add(new("tool", page, ToolName: "web_fetch"));
    }
    """,
    out="""
    --- turn 1
    content:  "\\n"
    tool_call: web_fetch {"url":"https://ollama.com/releases"}
    done: stop, prompt 255, generated 94
    --- turn 2
    thinking: "\\nThe p"   thinking: "age says"   thinking: " Ollama "   thinking: "6.29.7 i"
    thinking: "s the la"   thinking: "test rel"   thinking: "ease.\\n"
    content:  "The late"   content:  "st Ollam"   content:  "a versio"   content:  "n is 7.2"   content:  "8.7 [1]."
    done: stop, prompt 255, generated 109
    """,
    after="The format is right in both turns: a well-formed tool call with a sensible URL, then reasoning and an "
          "answer that cites the source as [1]. The fact is wrong: the page said 0.12.3, and the model wrote two "
          "different, invented versions. Section 22.7 measures how often each happens.")

API_STATIONS = "".join([
    snippet("""
        $ dotnet run -c Release --project samples/NeuralSharp.Samples.GptApi -- \\
              --Gpt:ChatModelPath=models/chat.weights
        $ curl -N localhost:5080/api/chat -d '{
            "model": "qwen3.8:27b", "stream": true, "think": true, "keep_alive": "30m",
            "options": { "temperature": 1, "top_k": 20, "top_p": 0.95, "min_p": 0, "repeat_penalty": 1,
                         "presence_penalty": 0, "num_ctx": 4096, "num_predict": 2048 },
            "messages": [
              { "role": "system", "content": "You are a helpful assistant. Cite sources as [1], [2] when a research pack is present." },
              { "role": "user", "content": "What is the latest Ollama version?" } ],
            "tools": [ { "type": "function", "function": { "name": "web_fetch",
              "description": "Fetch a page from the allowlisted search results.",
              "parameters": { "type": "object", "required": ["url"],
                "properties": { "url": { "type": "string", "description": "Absolute http(s) URL from the allowlist." } } } } } ]
          }'
        """, caption="An Ollama request, unchanged (no Content-Type header needed)"),
    output("""
        {"model":"qwen3.8:27b","created_at":"2026-09-25T17:28:08.72Z","message":{"role":"assistant","content":"","thinking":"\\nThe u"},"done":false}
        {"model":"qwen3.8:27b","created_at":"2026-09-25T17:28:08.73Z","message":{"role":"assistant","content":"","thinking":"ser want"},"done":false}
        ... 7 more thinking lines: "The user wants the latest Ollama version. I should fetch the release page."
        {"model":"qwen3.8:27b","created_at":"2026-09-25T17:28:08.88Z","message":{"role":"assistant","content":"",
          "tool_calls":[{"function":{"name":"web_fetch","arguments":{"url":"https://ollama.com/releases"},"index":0}}]},"done":false}
        {"model":"qwen3.8:27b","created_at":"2026-09-25T17:28:08.88Z","message":{"role":"assistant","content":""},"done":true,
          "done_reason":"stop","total_duration":316002600,"load_duration":0,"prompt_eval_count":255,
          "prompt_eval_duration":95529200,"eval_count":192,"eval_duration":220047600}
        """, caption="The streamed response: 12 NDJSON lines (timestamps shortened)"),
    para("The response echoes the requested model name; any name selects the served model. <code>num_ctx 4096</code> is "
         "capped by the model's context of 256 characters, so the prompt (about 560 characters with the tool "
         "definition) is cut to its last 255 (<code>prompt_eval_count</code>). <code>GET /api/ps</code> right after the "
         "request shows <code>\"expires_at\":\"2026-09-25T17:58:21Z\"</code>: 30 minutes, from <code>keep_alive</code>."),
    reftable(["20 runs of each request (no seed)", "Result"], [
        ["Turn 1: calls <code>web_fetch</code> with <code>https://ollama.com/releases</code>", "20 of 20"],
        ["Turn 1: thinks before calling", "10 of 20"],
        ["Turn 2 (tool result added, <code>\"stream\": false</code>): answer cites [1]", "20 of 20"],
        ["Turn 2: thinks before answering", "17 of 20"],
        ["Turn 2: answer states the version from the page (0.12.3)", "0 of 20; 20 different invented versions"],
    ], caption="Table 22.5 — What the small chat model learned, measured through the API"),
    para("A typical second answer: <code>{\"thinking\":\"The page says Ollama 19.12.0 is the latest release.\", "
         "\"content\":\"The latest Ollama version is 1.20.3 [1].\"}</code>. The invented versions follow the pattern of the "
         "training data (a number up to 19, then up to 29, then up to 9): the model learned what versions look like, "
         "not to copy the one in the tool result. Copying a string from earlier in the context is a separate skill "
         "that small models acquire late; a larger model, more training, or word-level tokens (a version as one "
         "token) all help. This is exactly the failure called <b>hallucination</b> in large models, and why answers "
         "that cite a source should be checked against it."),
])

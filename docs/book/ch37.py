"""Chapter 37 — Text Summarization: Extractive and Abstractive."""
from gen import *

PART = "VI"

# Measured by: dotnet run -c Release --project samples/NeuralSharp.Samples.Summarizer -- --cpu  (see RESULTS)
M = {k: "…" for k in ("params", "r1", "lead_r1", "exact", "numbers", "train_s", "ms", "vocab")}   # filled after the full run
RESULTS = ""
EXAMPLES = ""

LAYOUT = """
    Training sequence (96 token ids):
      the tigers played the foxes in elgin on friday . ... <sum> the tigers beat the foxes 4 to 0 in elgin . <end> <pad> ...
    Targets (next token), · = ignored:
      ·   ·      ·      ·   ·     ·  ·     ·  ·      ·  ...  the   tigers beat the foxes 4 to 0 in elgin . <end>  ·    ·     ·
"""

DATASET = """
    // Inputs are "<report> <sum> <summary> <end> <pad>..."; the target of each position is the next token, but only
    // where that next token belongs to the summary. Every other position gets the id `vocabulary` (one past the end).
    int ignore = words.VocabularySize;
    var features = new float[reports.Count * Context];
    var targets = new float[reports.Count * Context];
    Array.Fill(targets, ignore);
    for (int i = 0; i < reports.Count; i++)
    {
        var prompt = words.Encode(reports[i].Document + " <sum>");
        var ids = prompt.Concat(words.Encode(reports[i].Summary + " <end>")).Take(Context + 1).ToList();
        for (int t = 0; t < Context; t++)
        {
            features[i * Context + t] = t < ids.Count ? ids[t] : words["<pad>"];
            if (t + 1 < ids.Count && t + 1 >= prompt.Count)
                targets[i * Context + t] = ids[t + 1];
        }
    }
"""

LOSS = """
    // Cross-entropy over summary tokens only, averaged per summary token.
    static Tensor SummaryLoss(Tensor logits, Tensor next, double summaryTokensPerRow)
    {
        int vocabulary = logits.Shape[^1];
        // One-hot with one extra class, then cut it off: the "ignore" id becomes an all-zero row that adds nothing.
        var targets = Tensor.OneHot(next, vocabulary + 1).Narrow(2, 0, vocabulary);
        return (targets * logits.LogSoftmax()).Sum() * (float)(-1.0 / (next.Shape[0] * summaryTokensPerRow));
    }
"""

GENERATE = """
    var greedy = new GenerationOptions
    {
        Temperature = 0f, TopK = 1, RepeatPenalty = 1f,       // deterministic: always the most likely word
        NumPredict = 30, Stop = ["<end>"], NumCtx = Context,
    };
    model.Eval();
    var generator = new TextGenerator(model, tokenizer, Context);          // tokenizer: a WordTokenizer
    var (summary, reason, stats) = generator.Generate(report + " <sum>", greedy);
"""


def build():
    return page(
        chapter_open(
            "summarizer",
            "A summary keeps what matters and drops the rest. <b>Extractive</b> summarizers select sentences from the "
            "text; <b>abstractive</b> ones write new sentences, which lets them combine facts that are spread over the "
            "text, compare numbers and rephrase. This project, <code>NeuralSharp.Samples.Summarizer</code>, compares two "
            "extractive baselines with a word-level transformer that writes the summary token by token through the "
            "Generation layer of " + ch("textgen") + ", and scores all three with ROUGE and with checks of the facts.",
            "Data: short reports of four kinds (sports matches, weather, company results, fires) with one-sentence reference summaries.",
            f"Model: a decoder-only transformer over words ({M['params']} parameters), trained on \"report &lt;sum&gt; summary &lt;end&gt;\" with the loss masked to the summary.",
            "Generation: <code>WordTokenizer</code> + <code>TextGenerator</code>, greedy, stopping at <code>&lt;end&gt;</code>.",
            f"Result: ROUGE-1 {M['r1']} against {M['lead_r1']} for the best extractive baseline; {M['exact']} of summaries exactly right, {M['numbers']} with every number right.",
            f"Cost: training {M['train_s']} s on the CPU; {M['ms']} ms per summary.",
        ),
        h2("37.1 Reports that need more than copying"),
        para("<code>Reports.cs</code> generates the data. Each report starts with a scene-setting sentence, followed by "
             "two sentences with the key facts and two or three filler sentences, in random order. The reference "
             "summaries are designed so that no sentence of the report contains them: the winner of a match must be found "
             "by comparing two scores, \"rose\" or \"fell\" by comparing two revenues, and \"rain likely\" by checking "
             "whether a percentage is at least 50."),
        reftable(["Kind", "Key facts in the report", "Reference summary"], [
            ["match", "the tigers scored 4 goals . … the foxes scored 0 goals .", "the tigers beat the foxes 4 to 0 in elgin ."],
            ["weather", "the high will be 22 degrees … there is a 0 percent chance of rain .", "granton will reach 22 degrees on tuesday with little chance of rain ."],
            ["company", "revenue was 21 million dollars . … a year earlier revenue was 15 million dollars .", "ionix revenue rose from 15 to 21 million dollars in the third quarter ."],
            ["fire", "firefighters rescued 30 people from the building .", "30 people were rescued from a factory fire in elgin ."],
        ], caption="Table 37.1 — The four kinds of report"),
        para(f"8,000 reports are used for training (5 % of them for validation) and 400 new ones for testing. The "
             f"<code>WordTokenizer</code> built from the training texts has {M['vocab']} words, numbers 0–99 included, so "
             "a number is one token that the model can copy."),
        h2("37.2 Two extractive baselines"),
        snippet("""
            static List<string> Sentences(string document) =>
                [.. document.Split(" . ", StringSplitOptions.RemoveEmptyEntries).Select(s => s.TrimEnd(' ', '.') + " .")];

            // 1. The first sentence (news articles often put the gist first).
            string First(string doc) => Sentences(doc)[0];

            // 2. The sentence whose words occur most often in the whole report.
            static string Central(string document)
            {
                var counts = WordTokenizer.Split(document).GroupBy(w => w).ToDictionary(g => g.Key, g => g.Count());
                return Sentences(document).MaxBy(s => WordTokenizer.Split(s).Average(w => (double)counts.GetValueOrDefault(w)))!;
            }
            """, caption="Extractive summaries: choose one sentence"),
        para("Both are instant and can never state something false, but here neither can be right: the summary sentence "
             "is not in the report. They set the floor that a learned model must beat."),
        h2("37.3 Training only on the summary"),
        para("The abstractive model is a GPT like the one in " + ch("gpt") + ", over words instead of characters: "
             "embedding, positional encoding, three causal transformer layers of width 96, and a linear layer to the "
             "vocabulary. It learns from sequences \"report &lt;sum&gt; summary &lt;end&gt;\". Predicting the report "
             "itself would be wasted effort (its sentences are random), so only the summary positions have targets "
             "(<b>loss masking</b>)."),
        output(LAYOUT, caption="What the model reads and what it is trained to predict"),
        snippet(DATASET, caption="Building masked training sequences"),
        snippet(LOSS, caption="The masked loss, from ordinary tensor operations"),
        trap("averaging over positions that do not count",
             "<p>Dividing the masked loss by all 96 positions would make it about eight times smaller than the true loss "
             "per summary token, and hide how well the model really does. Divide by the number of summary tokens (here "
             "their average per row, a constant computed from the data).</p>"),
        h2("37.4 Writing the summary"),
        para("At inference time the report plus <code>&lt;sum&gt;</code> is the prompt, and the Generation layer does the "
             "rest: the KV cache processes the prompt once, the sampler picks the most likely word, and generation stops "
             "when the text contains <code>&lt;end&gt;</code>, which is not returned."),
        snippet(GENERATE, caption="Summarizing with TextGenerator"),
        h2("37.5 Results"),
        output(RESULTS, caption="dotnet run -c Release --project samples/NeuralSharp.Samples.Summarizer -- --cpu"),
        para("ROUGE compares a summary with the reference by overlapping words (ROUGE-1), word pairs (ROUGE-2) and the "
             "longest common word sequence (ROUGE-L), each as the F1 of precision and recall (glossary <b>ROUGE</b>). "
             "It rewards the right words, not the right facts: \"the tigers beat the tigers 4 to 0\" still scores well. "
             "That is why the sample also counts exact summaries and summaries with every number right and in order."),
        output(EXAMPLES, caption="Summaries of new reports"),
        honestbox("What these numbers do and do not show",
                  "<p>The reports are synthetic and follow four templates, so a small model can learn them almost "
                  "perfectly; on real news the same model would need far more data and a larger vocabulary (sub-word "
                  "tokens). What transfers is the method: masked training on \"input &lt;sep&gt; output\" sequences, "
                  "generation with a stop token, and evaluation that checks facts, not only word overlap.</p>"),
        cpugpu("training and inference commands",
               """
               dotnet run -c Release --project samples/NeuralSharp.Samples.Summarizer -- --cpu
               dotnet run -c Release --project samples/NeuralSharp.Samples.Summarizer -- --cpu --predict \\
                   --input "the lions played the owls in kelso on friday . the owls scored 2 goals . the lions scored 4 goals ."
               """,
               """
               dotnet run -c Release --project samples/NeuralSharp.Samples.Summarizer -- --cuda
               dotnet run -c Release --project samples/NeuralSharp.Samples.Summarizer -- --cuda --predict
               """,
               note="Without <code>--input</code>, prediction mode summarizes a freshly generated report. On the GPU the "
                    "decoding step is recorded once as a CUDA graph and replayed for every word."),
        practice([
            (1, "Why can neither extractive baseline produce the reference summary here?",
             "The reference sentence does not occur in the report: it combines facts from several sentences and states a "
             "comparison (winner, rose/fell) that no sentence contains."),
            (1, "What does the model receive as the target at a position inside the report?",
             "The ignore id (the vocabulary size), which becomes an all-zero one-hot row, so the position adds nothing to the loss."),
            (2, "Add a fifth report kind (for example election results) and check that the model learns it.",
             "Add a generator in <code>Reports.cs</code> with a lead, two fact sentences, fillers and a summary that needs a "
             "comparison (the winner); include it in the <code>switch</code> with <code>i % 5</code>; retrain and look at "
             "the per-kind exact-match lines."),
            (2, "Why does greedy decoding suit this task better than sampling with temperature 0.8?",
             "A summary should state the facts; sampling can pick a less likely word, such as a wrong number or \"fell\" "
             "instead of \"rose\", with no benefit from variety."),
            (3, "Adapt the sample to real text: what must change?",
             "A much larger corpus of (document, summary) pairs; a sub-word tokenizer so rare words and numbers are not "
             "&lt;unk&gt;; a longer context (reports longer than 96 tokens); a bigger model trained on the GPU; and "
             "evaluation by people or by fact checks, since ROUGE alone rewards fluent but wrong summaries."),
        ], PART),
        footer("Summarization", "Extractive summarization", "Abstractive summarization", "ROUGE", "Loss masking",
               "Greedy decoding", "Tokenizer", "Unknown token"),
    )

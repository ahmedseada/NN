"""Station library for the NeuralSharp book. Every page is built from these functions (Rule 3).

All content strings are HTML fragments: write literal < and > as &lt; / &gt; (Rule 5); esc_check() fails the
build otherwise. Code passed to code()/mex() is plain text and is escaped here.
"""
import html
import re

# ---------------------------------------------------------------- safety (Rule 5)

ALLOWED_TAGS = {"b", "i", "em", "strong", "code", "span", "br", "ul", "ol", "li", "p", "div", "table", "thead",
                "tbody", "tr", "th", "td", "pre", "h1", "h2", "h3", "a", "sub", "sup", "svg", "g", "rect", "line",
                "circle", "text", "path", "polygon", "polyline", "defs", "marker", "dl", "dt", "dd", "section", "tspan"}
_TAG = re.compile(r"</?([a-zA-Z][a-zA-Z0-9]*)(\s[^<>]*)?/?>")


def esc(text):
    """Escapes plain text for HTML."""
    return html.escape(text, quote=False)


def esc_check(fragment, where):
    """Fails on any '<' that does not start an allowed tag (an unescaped <b... can swallow text)."""
    for m in re.finditer("<", fragment):
        tag = _TAG.match(fragment, m.start())
        if not tag or tag.group(1).lower() not in ALLOWED_TAGS:
            snippet = fragment[max(0, m.start() - 30): m.start() + 40].replace("\n", " ")
            raise ValueError(f"esc_check: bare '<' in {where}: ...{snippet}...")
    for bad in ("ﬁ", "ﬂ", "ﬀ", "ﬃ"):
        if bad in fragment:
            raise ValueError(f"esc_check: ligature character {bad!r} in {where}; use plain letters")
    return fragment


def brk(symbol):
    """Wraps an exotic glyph so it renders from the math font."""
    return f'<span class="f">{symbol}</span>'


def f(formula):
    """Inline formula (mono, weight 800). Pass HTML-safe text."""
    return f'<span class="f">{formula}</span>'


# ---------------------------------------------------------------- C# highlighting

_CS = re.compile(r"""(?P<c>//[^\n]*)|(?P<s>\$?@?"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)')|(?P<n>\b\d[\d_]*(?:\.\d+)?[fFdDmML]?\b)|(?P<w>[A-Za-z_][A-Za-z0-9_]*)""")
_KEYWORDS = set("""using var new return if else for foreach in while do break continue public private internal static
readonly const class record struct void int float double bool string true false null this base await async
sealed override virtual abstract interface namespace out ref is not and or with default throw try catch finally
typeof params yield get set init when switch case lock""".split())


def highlight(code_text):
    out, pos = [], 0
    for m in _CS.finditer(code_text):
        out.append(esc(code_text[pos:m.start()]))
        tok = m.group(0)
        if m.group("c"):
            out.append(f'<span class="c">{esc(tok)}</span>')
        elif m.group("s"):
            out.append(f'<span class="s">{esc(tok)}</span>')
        elif m.group("n"):
            out.append(f'<span class="n">{esc(tok)}</span>')
        elif tok in _KEYWORDS:
            out.append(f'<span class="k">{esc(tok)}</span>')
        elif tok[0].isupper():
            out.append(f'<span class="t">{esc(tok)}</span>')
        else:
            out.append(esc(tok))
        pos = m.end()
    out.append(esc(code_text[pos:]))
    return "".join(out)


def _dedent(text):
    lines = text.strip("\n").split("\n")
    indent = min((len(l) - len(l.lstrip()) for l in lines if l.strip()), default=0)
    return "\n".join(l[indent:] for l in lines)


def code(text, caption=None, lang="cs"):
    body = highlight(_dedent(text)) if lang == "cs" else esc(_dedent(text))
    cap = f'<div class="cap">{caption}</div>' if caption else ""
    return f'{cap}<pre>{body}</pre>'


def output(text, caption="Output"):
    return f'<div class="cap">{caption}</div><pre class="out">{esc(_dedent(text))}</pre>'


# ---------------------------------------------------------------- stations

def topbar(left, right):
    return f'<div class="topbar"><span>{left}</span><span>{right}</span></div>'


def _anchor(text):
    return "a-" + re.sub(r"[^a-z0-9]+", "-", re.sub(r"<[^>]+>", "", text).lower()).strip("-")


def h1(number, title, toc=True):
    """Chapter/Part heading; registers a Contents entry whose page number the renderer resolves."""
    aid = _anchor(f"{number} {title}")
    if toc:
        TOC.append((1, f"{number.title()} · {title}", aid))
    return f'<div class="h1" id="{aid}"><div class="num">{number}</div><h1>{title}</h1></div>'


def h2(title, toc=True):
    aid = _anchor(title)
    if toc:
        TOC.append((2, title, aid))
    return f'<h2 id="{aid}">{title}</h2>'


def h3(title):
    return f"<h3>{title}</h3>"


def para(*paragraphs):
    return "".join(f"<p>{p}</p>" for p in paragraphs)


def glance(*items, title="At a glance"):
    lis = "".join(f"<li>{i}</li>" for i in items)
    return f'<div class="station box glance"><div class="title">{title}</div><ul>{lis}</ul></div>'


def brief(text):
    return f'<div class="station brief"><p>{text}</p></div>'


def defbox(term, text):
    return f'<div class="station box defbox"><div class="title">Definition · {term}</div>{text}</div>'


def honestbox(title, text):
    """Deferrals and honest limits (never a defbox)."""
    return f'<div class="station box honestbox"><div class="title">{title}</div>{text}</div>'


def trap(title, text):
    return f'<div class="station box trap"><div class="title">Trap · {title}</div>{text}</div>'


def reftable(headers, rows, caption=None, atomic=True):
    """atomic=False lets a long table (e.g. an Answer Key) run across pages; rows never split."""
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    cap = f'<div class="cap">{caption}</div>' if caption else ""
    cls = "station" if atomic else "flow"
    return f'<div class="{cls}">{cap}<table class="ref"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def deriv(title, steps):
    lis = "".join(f"<li>{s}</li>" for s in steps)
    return f'<div class="station deriv"><div class="title">{title}</div><ol>{lis}</ol></div>'


def mex(title, intro, snippet, out=None, caption=None, after=None):
    """Worked example: a runnable snippet, optionally with its output."""
    parts = [f'<div class="title">Worked example · {title}</div>']
    if intro:
        parts.append(f"<p>{intro}</p>")
    parts.append(code(snippet, caption))
    if out:
        parts.append(output(out))
    if after:
        parts.append(f"<p>{after}</p>")
    return f'<div class="{_cls(snippet, out)} mex">{"".join(parts)}</div>'


LONG_LINES = 40   # stations with more code lines than this may break across pages (they cannot fit on one)


def _cls(*texts):
    lines = sum(t.count("\n") for t in texts if t)
    return "station long" if lines > LONG_LINES else "station"


def snippet(text, caption=None):
    """A standalone code block, kept on one page unless it is longer than a page can hold."""
    return f'<div class="{_cls(text)}">{code(text, caption)}</div>'


def diagram(title, svg, caption):
    # SVG user units are CSS px (0.75pt); draw at 4/3 so a label of size 8 prints at 8pt.
    svg = re.sub(r'<svg width="([\d.]+)" height="([\d.]+)"',
                 lambda m: f'<svg width="{float(m.group(1)) * 4 / 3:.0f}" height="{float(m.group(2)) * 4 / 3:.0f}"', svg, count=1)
    return f'<div class="station diagram"><div class="title">{title}</div>{svg}<div class="cap">{caption}</div></div>'


def diffcircle(level, of=3):
    return f'<span class="diff">{"●" * level}{"○" * (of - level)}</span>'


def practice(items, part_key):
    """items: (difficulty, question, answer). Answers are collected into the Part's Answer Key."""
    lis = []
    for i, (level, question, answer) in enumerate(items, 1):
        ANSWERS.setdefault(part_key, []).append((f"{CURRENT['chapter']}.{i}", answer))
        lis.append(f"<li>{diffcircle(level)} {question}</li>")
    return f'<div class="station practice"><div class="title">Practice</div><ol>{"".join(lis)}</ol></div>'


def footer(*terms):
    """Key-terms strip; build.py extracts these calls with ast to build the Glossary."""
    return f'<div class="terms"><b>Key terms</b> · {" · ".join(terms)}</div>'


def page(*stations, new=False):
    stations = list(stations)
    # keep the last Practice block and the key-terms strip on the same page
    if len(stations) >= 2 and stations[-1].startswith('<div class="terms">') and 'class="station practice"' in stations[-2]:
        stations[-2:] = [f'<div class="chapter-end">{stations[-2]}{stations[-1]}</div>']
    cls = "page newpage" if new else "page"
    return f'<section class="{cls}">{"".join(stations)}</section>'


# ---------------------------------------------------------------- SVG helpers (Rule 2: fonts on every <text>)

SVG_FONT = 'font-family="JetBrains Mono, Noto Sans Math, Noto Sans" font-weight="800"'


def svg_text(x, y, s, size=9, fill="#1a1f23", anchor="middle"):
    return f'<text x="{x}" y="{y}" {SVG_FONT} font-size="{size}" fill="{fill}" text-anchor="{anchor}">{s}</text>'


# ---------------------------------------------------------------- roadmap (single source for cross-references)

PARTS = {
    "I": ("Foundations", "Devices, tensors, memory and gradients: the four ideas under every NeuralSharp program."),
    "II": ("Building Networks", "Every layer in the library: what it computes, its shapes, and how to combine it."),
    "III": ("Training", "Losses, metrics, optimizers, schedules, data pipelines, the Trainer and telemetry."),
    "IV": ("Performance and Internals", "Making it fast, generating text quickly, and how the CPU and CUDA backends work."),
    "V": ("Projects: Tabular Data and Time Series", "Complete projects on numbers in rows and in time."),
    "VI": ("Projects: Vision, Sequences and Text", "Complete projects on images, characters, sentences and language models."),
    "VII": ("Projects: Shipping and Integration", "Putting trained models to work in real .NET applications."),
}

CHAPTERS = [  # (key, part, title)
    ("start", "I", "Getting Started: Your First Network on CPU and GPU"),
    ("devices", "I", "Devices, Backends and Compute Resources"),
    ("tensors", "I", "Tensors"),
    ("memory", "I", "Memory and Tensor Lifetime"),
    ("autograd", "I", "Autograd: How Gradients Are Computed"),
    ("modules", "II", "Modules, Sequential and Model Files"),
    ("dense", "II", "Dense Layers and Activations"),
    ("norm", "II", "Normalization and Dropout"),
    ("embedding", "II", "Embeddings"),
    ("conv", "II", "Convolution and Pooling"),
    ("recurrent", "II", "Recurrent Layers: LSTM and GRU"),
    ("attention", "II", "Attention and Transformers"),
    ("losses", "III", "Losses"),
    ("metrics", "III", "Metrics"),
    ("optimizers", "III", "Optimizers"),
    ("schedules", "III", "Learning-Rate Schedules and Gradient Clipping"),
    ("data", "III", "Data: Datasets, CSV, Scalers and Loaders"),
    ("trainer", "III", "The Trainer"),
    ("telemetry", "III", "Telemetry: Logging and Tracking Everything"),
    ("performance", "IV", "Performance Tuning"),
    ("generation", "IV", "Fast Generation: KV Cache, Sampling and CUDA Graphs"),
    ("textgen", "IV", "The Generation Layer: Text, Chat, Tools and Model Hosting"),
    ("cpu", "IV", "Inside the CPU Backend"),
    ("cuda", "IV", "Inside the CUDA Backend"),
    ("testing", "IV", "Testing, Gradient Checking and Debugging"),
    ("regression", "V", "Regression: Predicting House Prices"),
    ("binary", "V", "Binary Classification"),
    ("multiclass", "V", "Multi-Class Classification"),
    ("timeseries", "V", "Time-Series Forecasting"),
    ("anomaly", "V", "Anomaly Detection with Autoencoders"),
    ("recommender", "V", "Recommenders and Categorical Embeddings"),
    ("cnn", "VI", "Image Classification with CNNs"),
    ("ocr", "VI", "OCR: Reading Characters"),
    ("sentiment", "VI", "Sequence Classification and Sentiment"),
    ("gpt", "VI", "GPT Text Generation"),
    ("reranker", "VI", "Search Re-Ranking with a Cross-Encoder"),
    ("summarizer", "VI", "Text Summarization: Extractive and Abstractive"),
    ("inference", "VII", "Inference Modes and Model Files in Practice"),
    ("webapi", "VII", "Serving Models over HTTP"),
    ("finetune", "VII", "Fine-Tuning, Freezing and Multi-Output Models"),
    ("projecttypes", "VII", "NeuralSharp in Every .NET Project Type"),
]
CH = {key: i for i, (key, _, _) in enumerate(CHAPTERS, 1)}


def ch(key, word="Chapter"):
    """Cross-reference to a chapter by key, e.g. ch('trainer') -> 'Chapter 18'."""
    return f"{word} {CH[key]}" if word else str(CH[key])


def chapter_open(key, brief_text, *glance_items):
    """Topbar + heading + brief + at-a-glance for chapter `key`; also sets CURRENT for the Answer Key."""
    n = CH[key]
    part, title = CHAPTERS[n - 1][1], CHAPTERS[n - 1][2]
    CURRENT["chapter"] = str(n)
    CURRENT["part"] = part
    return (topbar(f"PART {part} · {PARTS[part][0].upper()}", f"CHAPTER {n}") + h1(f"CHAPTER {n}", title)
            + brief(brief_text) + glance(*glance_items))


def partpage(part):
    """Part title page: name, purpose and the chapters it contains."""
    name, purpose = PARTS[part]
    rows = "".join(f'<li><span class="pn">{i}</span>{t}</li>'
                   for i, (_, p, t) in enumerate(CHAPTERS, 1) if p == part)
    aid = _anchor(f"part {part} {name}")
    TOC.append((0, f"Part {part} · {name}", aid))
    return (f'<section class="page newpage partpage" id="{aid}"><div class="partnum">PART {part}</div>'
            f'<div class="partname">{name}</div><p class="partpurpose">{purpose}</p>'
            f'<ol class="partch">{rows}</ol></section>')


def cpugpu(title, cpu, gpu, note=None):
    """The CPU and GPU versions of the same code, labelled, in one atomic station."""
    n = f"<p>{note}</p>" if note else ""
    return (f'<div class="{_cls(cpu, gpu)} box cpugpu"><div class="title">CPU and GPU · {title}</div>'
            f'<div class="lbl cpu">CPU</div>{code(cpu)}<div class="lbl gpu">GPU</div>{code(gpu)}{n}</div>')


# collected while building
ANSWERS = {}
TOC = []
CURRENT = {"chapter": "0", "part": "I"}

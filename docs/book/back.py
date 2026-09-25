"""Back matter: Answer Key (one continuous page per Part) and Glossary (terms from footer() calls)."""
import ast
import json
import pathlib
import re
from gen import *

HERE = pathlib.Path(__file__).parent


def answer_key(part):
    rows = [[num, ans] for num, ans in ANSWERS.get(part, [])]
    return page(topbar("BACK MATTER", "ANSWER KEY"), h1("ANSWER KEY", f"Part {part}"),
                reftable(["No.", "Answer"], rows, atomic=False).replace('class="ref"', 'class="ref key"'), new=True)


def footer_terms(chapter_files):
    """Collects footer("...") terms from chapter sources with ast (Rule: glossary from footer calls)."""
    terms = set()
    for path in chapter_files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "footer":
                terms.update(a.value for a in node.args if isinstance(a, ast.Constant))
    return sorted(terms, key=str.lower)


def glossary(chapter_files):
    data = json.loads((HERE / "data" / "glossary.json").read_text(encoding="utf-8"))
    entries = []
    missing = []
    for term in footer_terms(chapter_files):
        if term not in data:
            missing.append(term)
            continue
        e = data[term]
        hint = f' <i>Hint:</i> {e["hint"]}' if e.get("hint") else ""
        see_text = re.sub(r"\{ch:(\w+)\}", lambda m: ch(m.group(1)), e.get("see", ""))
        see = f' <span class="cap">See: {see_text}.</span>' if see_text else ""
        entries.append(f'<div class="entry"><dt>{term}</dt><dd>{e["def"]}{hint}{see}</dd></div>')
    if missing:
        raise ValueError(f"glossary.json lacks: {missing}")
    (HERE / "data" / "glossary_terms.json").write_text(json.dumps(footer_terms(chapter_files), indent=1))
    return page(topbar("BACK MATTER", "GLOSSARY"), h1("GLOSSARY", "Terms, hints and where to read more"),
                brief("Math is only hinted at here. For the full treatment see the Pre-Calc volume "
                      "<i>Numbers, Shapes, and Functions</i> and the <i>Activation Functions</i> volume of this series."),
                f'<dl class="gloss">{"".join(entries)}</dl>', new=True)


def outro(version, commit, date):
    return page(
        topbar("BACK MATTER", "OUTRO"),
        h1("OUTRO", "Where to Go from Here"),
        brief("You have now seen every part of NeuralSharp, from a single tensor to a GPT served over HTTP, and "
              "built a dozen kinds of project with it, each on the CPU and on the GPU."),
        h2("What you can build now", toc=False),
        reftable(["If your data is…", "Start from"], [
            ["rows of numbers → a number", ch("regression")],
            ["rows of numbers → yes/no", ch("binary")],
            ["rows of numbers → one of K classes", ch("multiclass")],
            ["a series over time → its future", ch("timeseries")],
            ["normal records → alerts on unusual ones", ch("anomaly")],
            ["users, items, categories", ch("recommender")],
            ["images", ch("cnn") + ", " + ch("ocr")],
            ["sentences or event sequences", ch("sentiment")],
            ["text to continue", ch("gpt")],
            ["a trained model to ship", ch("inference") + "–" + ch("projecttypes")],
        ], caption="Table O.1 — Choosing a starting point"),
        h2("Going deeper", toc=False),
        para("The mathematics that this book only hints at is developed in the other volumes of the series: the "
             "Pre-Calc volume <i>Numbers, Shapes, and Functions</i> for functions, exponentials, logarithms, "
             "trigonometry and rates of change, and the <i>Activation Functions</i> volume for every activation, its "
             "derivative and its effect on gradients. Inside the library, " + ch("cpu") + " and " + ch("cuda") +
             " show where to add your own operations, and the test project shows how to verify them."),
        h2("Colophon", toc=False),
        reftable(["", ""], [
            ["Book", f"NeuralSharp · The Library and Its Projects, version {version}, {date}"],
            ["Library", f"NeuralSharp for .NET 10, repository commit {commit}"],
            ["Examples", "Every printed output comes from a real run on a 4-core cloud CPU (AVX-512, 8-wide Vector&lt;float&gt;) "
                         "unless marked otherwise; GPU figures quoted from runs on a laptop GeForce RTX 5050"],
            ["Typesetting", "Generated from Python sources with WeasyPrint; Noto Sans, Noto Sans Math and JetBrains Mono"],
        ], caption="Table O.2 — About this edition"),
        new=True,
    )


def index_pages(entries):
    """entries: list of (term, [pages]) sorted by term."""
    groups = {}
    for term, pages in entries:
        key = term[0].upper() if term[0].isalpha() else "#"
        groups.setdefault(key, []).append((term, pages))
    blocks = []
    for key in sorted(groups):
        rows = "".join(f'<div class="ix"><span class="ixt">{t}</span> <span class="ixp">{", ".join(map(str, p))}</span></div>'
                       for t, p in groups[key])
        blocks.append(f'<div class="ixg"><div class="ixk">{key}</div>{rows}</div>')
    return page(topbar("BACK MATTER", "INDEX"), h1("INDEX", "Index"),
                para("Page numbers refer to the chapters (at most ten per term). Definitions are in the Glossary."),
                f'<div class="index">{"".join(blocks)}</div>', new=True)

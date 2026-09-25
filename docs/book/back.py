"""Back matter: Answer Key (one continuous page per Part) and Glossary (terms from footer() calls)."""
import ast
import json
import pathlib
from gen import *

HERE = pathlib.Path(__file__).parent


def answer_key(part):
    rows = [[num, ans] for num, ans in ANSWERS.get(part, [])]
    return page(topbar("BACK MATTER", "ANSWER KEY"), h1("ANSWER KEY", f"Part {part}"),
                reftable(["No.", "Answer"], rows).replace('class="ref"', 'class="ref key"'), new=True)


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
        see = f' <span class="cap">See: {e["see"]}.</span>' if e.get("see") else ""
        entries.append(f'<div class="entry"><dt>{term}</dt><dd>{e["def"]}{hint}{see}</dd></div>')
    if missing:
        raise ValueError(f"glossary.json lacks: {missing}")
    (HERE / "data" / "glossary_terms.json").write_text(json.dumps(footer_terms(chapter_files), indent=1))
    return page(topbar("BACK MATTER", "GLOSSARY"), h1("GLOSSARY", "Terms, hints and where to read more"),
                brief("Math is only hinted at here. For the full treatment see the Pre-Calc volume "
                      "<i>Numbers, Shapes, and Functions</i> and the <i>Activation Functions</i> volume of this series."),
                f'<dl class="gloss">{"".join(entries)}</dl>', new=True)

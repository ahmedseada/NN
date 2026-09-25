"""Front matter: cover and Contents (page numbers resolved from the final layout)."""
from gen import *

SERIES = "NeuralSharp · The Library and Its Projects"


def cover():
    return ('<section class="cover">'
            '<div class="band"><div class="series">NeuralSharp Series · Volume 1</div>'
            '<div class="title">NeuralSharp<br/>The Library and Its Projects</div>'
            '<div class="subtitle">A self-sufficient guide to building, training and serving neural networks in '
            'pure C# on .NET 10, on the CPU and on NVIDIA GPUs, from the first tensor to a GPT served over HTTP.</div>'
            '<div class="author">Ahmed Seada</div></div>'
            '<div class="meta"><b>Part I</b> · The Library: every part, what it does and how to use it<br/>'
            '<b>Part II</b> · The Projects: every kind of project you can build, end to end<br/>'
            '<b>Back matter</b> · Answer Key · Glossary · Index</div>'
            + code("""
                using NeuralSharp;
                Device.Default = Device.IsCudaAvailable ? Device.Cuda() : Device.Cpu;   // GPU when present
            """, "Every program in this book starts the same way")
            + '</section>')


def contents():
    items = []
    for level, title, aid in TOC:
        items.append(f'<li class="l{level}"><a href="#{aid}">{title}</a><span class="dots"></span>'
                     f'<a class="pn" href="#{aid}"></a></li>')
    return page(topbar("NEURALSHARP", "CONTENTS"), h1("SAMPLE EDITION", "Contents", toc=False),
                f'<ul class="toc">{"".join(items)}</ul>', new=True)


def how_to_use():
    return page(
        topbar("NEURALSHARP", "HOW TO USE THIS BOOK"),
        h1("BEFORE YOU START", "How to Use This Book"),
        brief("This book is the complete manual for NeuralSharp, a neural-network library written in pure C# for .NET 10 "
              "that runs on the CPU and on NVIDIA GPUs. It is meant to be enough on its own: every part of the library is "
              "explained with code you can run, and every kind of project you can build with it is shown end to end."),
        h2("Two halves", toc=False),
        reftable(["Parts", "What they contain", "Read them"], [
            ["I–IV · The Library", "Every part of the library, bottom up: devices, tensors, memory, autograd (I); every "
             "layer (II); losses, metrics, optimizers, data, the Trainer and telemetry (III); performance, fast "
             "generation and the internals of both backends (IV).", "In order the first time; later as a reference"],
            ["V–VII · The Projects", "Complete projects: tabular data and time series (V); images, characters, sentences "
             "and language models (VI); inference, web APIs, fine-tuning and every .NET project type (VII).",
             "Pick the project closest to yours; each one lists the chapters it builds on"],
        ], caption="Table 0.1 — The structure of the book"),
        h2("CPU and GPU, always both", toc=False),
        para("Every chapter shows how its code runs on the CPU and on the GPU. Usually the difference is one line, "
             "shown in a box like this one:"),
        cpugpu("example",
               """
               Device.Default = Device.Cpu;
               """,
               """
               Device.Default = Device.IsCudaAvailable ? Device.Cuda() : Device.Cpu;
               """,
               "Output printed in the book comes from real runs. Unless stated otherwise it was produced on the CPU; on "
               "the GPU the numbers agree up to rounding in the last digit, and tensors print <code>device=cuda:0</code>."),
        h2("The boxes", toc=False),
        reftable(["Box", "Contains"], [
            ["<b>At a glance</b>", "The chapter's key facts, for a quick review."],
            ["<b>Definition</b>", "A precise meaning for a term used from then on."],
            ["<b>Worked example</b>", "A runnable program or fragment, usually with its real output."],
            ["<b>CPU and GPU</b>", "The two versions of the same code side by side."],
            ["<b>Step by step</b>", "A procedure or what happens inside one call, in order."],
            ["<b>Trap</b>", "A common mistake, its symptom and its fix."],
            ["<b>Honest note</b> (amber)", "A limit of the library, or a topic deliberately left to another book."],
            ["<b>Practice</b>", "Exercises marked ●○○ (routine) to ●●● (project-sized); answers are in the Answer Key of each Part."],
            ["<b>Key terms</b>", "The chapter's vocabulary; every term is defined in the Glossary."],
        ], caption="Table 0.2 — Reading the page"),
        h2("Where the math is", toc=False),
        para("This book uses gradients, activation functions and probability as tools. It gives the idea and a one-line "
             "hint, never a derivation. The Glossary entry for each term says where the full treatment is: the Pre-Calc "
             "volume <i>Numbers, Shapes, and Functions</i> (functions, exponentials, logarithms, rates of change) and the "
             "<i>Activation Functions</i> volume (every activation, its derivative, its range and its effect on gradients)."),
        h2("Running the code", toc=False),
        deriv("Setting up a project for the examples", [
            "<code>dotnet new console -n Playground</code> and <code>cd Playground</code>.",
            "<code>dotnet add reference ../NN/src/NeuralSharp/NeuralSharp.csproj</code> (adjust the path).",
            "Put the example in <code>Program.cs</code> after <code>using NeuralSharp;</code> (and "
            "<code>using NeuralSharp.Layers;</code>, <code>.Optimizers</code>, <code>.Data</code>, <code>.Training</code> as needed).",
            "<code>dotnet run -c Release</code>. Debug builds work but are several times slower.",
        ]),
        para("Code uses current C#: top-level statements, collection expressions (<code>[1f, 2, 3]</code>), "
             "<code>using var</code> declarations and target-typed <code>new</code>. Fragments assume the variables of "
             "the preceding example in the same section."),
        new=True,
    )

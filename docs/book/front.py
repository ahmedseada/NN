"""Front matter: cover and Contents (page numbers resolved from the final layout)."""
from gen import *

SERIES = "NeuralSharp · The Library and Its Projects"


def cover():
    return ('<section class="cover">'
            '<div class="band"><div class="series">NeuralSharp Series · Volume 1</div>'
            '<div class="title">NeuralSharp<br/>The Library and Its Projects</div>'
            '<div class="subtitle">A self-sufficient guide to building, training and serving neural networks in '
            'pure C# on .NET 10, on the CPU and on NVIDIA GPUs, from the first tensor to a GPT served over HTTP.</div></div>'
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

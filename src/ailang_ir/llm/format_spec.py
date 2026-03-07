"""
LLM-native format specification for AILang-IR.

Defines the compact format that LLMs can directly read and write.
The spec is designed for system prompt injection (≤300 tokens).
"""

FORMAT_SPEC = """\
# AILang-IR LLM Format

Encode semantic statements as: `HEADER object_key [>target_key]`

## Header (6 positional chars)

| Pos | Field     | Values |
|-----|-----------|--------|
| 0   | Speaker   | U=user S=system A=agent T=third ?=unknown |
| 1   | Mode      | a=assert o=opinion h=hypothesis r=request c=command q=question k=commit b=observe f=reflect |
| 2:4 | Act       | bl=believe pr=prefer sg=suggest nd=need dc=decide rj=reject ag=agree dg=disagree cr=create md=modify dl=delete qu=query ob=observe cm=compare pl=plan wn=warn ex=explain uk=unknown |
| 4   | Certainty | hex 0-f (0=0% f=100%) |
| 5   | Time      | n=now p=past f=future a=atemporal ?=unknown |

## Object Key

1-3 lowercase words, underscore separated. Core concept only.

## Target (optional)

`>target_key` for comparisons (e.g., prefer X over Y).

## Examples

```
Uoblbn graph_mem
Uandbn sem_compress
Uaprbn typed_models >dicts
Uqqu7n parser
Ukplbf persistence
```\
"""


def get_format_spec() -> str:
    """Return the LLM format specification string for system prompt injection."""
    return FORMAT_SPEC

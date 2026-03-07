"""
LLM-native format specification for AILang-IR.

Defines the compact format that LLMs can directly read and write.
The spec is designed for system prompt injection (≤300 tokens).
"""

FORMAT_SPEC = """\
# AILang-IR Compressed Format

Each line: `HEADER object_key [>target] [#act_label]`

Header = 6 chars: Speaker(U/A/S/T/?) Mode(a/o/h/r/c/q/k/b/f) Act(2ch) Certainty(0-f hex) Time(n/p/f/a/?)
Act codes: bl=believe sg=suggest pr=prefer ag=agree dg=disagree nd=need dc=decide rj=reject cr=create md=modify dl=delete qu=query ob=observe cm=compare pl=plan wn=warn ex=explain uk=unknown

Lines ending with `#act_label` clarify the stance (e.g. #disagree, #prefer).

Examples:
  Uoblbn postgresql_main          — User believes postgresql is main choice
  Uadgbn monolith #disagree       — User disagrees, prefers monolith
  Uaprbn rest >graphql #prefer    — User prefers rest over graphql
  Aasgbn redis_caching             — Agent suggests redis caching\
"""


FORMAT_SPEC_FULL = """\
# AILang-IR LLM Format (Full Reference)

Encode semantic statements as: `HEADER object_key [>target_key] [#act_label]`

## Header (6 chars, layout: S M AA C T)

```
pos:  0 1 2 3 4 5
field: S M A A C T
ex:   U o b l b n  = User, opinion, believe, 73%, now
```

| Pos | Field     | Values |
|-----|-----------|--------|
| 0   | Speaker   | U=user S=system A=agent T=third ?=unknown |
| 1   | Mode      | a=assert o=opinion h=hypothesis r=request c=command q=question k=commit b=observe f=reflect |
| 2-3 | Act (2ch) | bl=believe pr=prefer sg=suggest nd=need dc=decide rj=reject ag=agree dg=disagree cr=create md=modify dl=delete qu=query ob=observe cm=compare pl=plan wn=warn ex=explain uk=unknown |
| 4   | Certainty | hex 0-f (0=0% f=100%) |
| 5   | Time      | n=now p=past f=future a=atemporal ?=unknown |

## Object Key

1-3 lowercase words, underscore separated. Core concept only.

## Target (optional)

`>target_key` for comparisons (e.g., prefer X over Y).

## Act Label (optional)

`#act_label` appended to lines with stance-critical acts for clarity.

## Examples

```
Uoblbn graph_mem
Uandbn sem_compress
Uaprbn typed_models >dicts #prefer
Uadgbn microservices #disagree
Uqqu7n parser
Ukplbf persistence
```\
"""


def get_format_spec() -> str:
    """Return the compact LLM format specification for system prompt injection."""
    return FORMAT_SPEC


def get_format_spec_full() -> str:
    """Return the full LLM format specification with complete reference tables."""
    return FORMAT_SPEC_FULL

"""Synthetic cue-block generation via OpenRouter (Claude Opus).

Real 'advanced' cue blocks are scarce at n≈32, so the generative (BioMistral) track needs
augmentation. This package generates schema-matched, framework-grounded synthetic cue
blocks, quarantines them (provenance tier = 'synthetic_claude_opus', train-only), and
validates them before use. See DATA_CONTRACT.md §4 (hard rule 3).
"""

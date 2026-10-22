"""Regression for the recorded first-target metadata failure; no new GPU inference."""
import ast
from pathlib import Path
def test_sampler_trace_and_manifest_have_no_duplicate_semantic_SHA_keyword():
    tree=ast.parse(Path('scripts/stage11c0/worker.py').read_text())
    matches=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='dict':
            if any(k.arg is None and isinstance(k.value,ast.Name) and k.value.id=='tr' for k in node.keywords):
                matches.append(node);assert not any(k.arg=='semantic_SHA' for k in node.keywords)
    assert matches

"""Dependency-free regression checks replacing the invalid historical demo script.

The previous file imported several heavyweight ML stacks and downloaded GPT-2
weights at module execution time. Those side effects do not belong in pytest
collection. Model-specific examples should live in separately selected demos.
"""


def reactive_machine(input_value):
    return "Positive" if input_value > 0 else "Negative"


def test_reactive_machine():
    assert reactive_machine(3) == "Positive"
    assert reactive_machine(0) == "Negative"
    assert reactive_machine(-2) == "Negative"

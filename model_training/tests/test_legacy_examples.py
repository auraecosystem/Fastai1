"""Small, dependency-free checks for retained example utilities."""


def reactive_machine(input_value):
    if input_value > 0:
        return "Positive"
    return "Negative"


def test_reactive_machine_examples():
    assert reactive_machine(1) == "Positive"
    assert reactive_machine(0) == "Negative"
    assert reactive_machine(-1) == "Negative"

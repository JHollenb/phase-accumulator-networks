from foo.core import say_hello


def test_say_hello() -> None:
    assert say_hello("Jake") == "Hello, Jake!"

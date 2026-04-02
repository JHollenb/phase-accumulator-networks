from pan.config import get_settings


def say_hello(name: str | None = None) -> str:
    settings = get_settings()
    target = name or settings.greeting_target
    return f"Hello, {target}! (app={settings.app_name}, debug={settings.debug})"

import functools
import warnings


def deprecated(reason: str = ""):
    """Decorator to mark functions as deprecated with IDE hint."""

    def decorator(func):
        message = f"Function {func.__name__} is deprecated."
        if reason:
            message += f" {reason}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        wrapper.__deprecated__ = True  # IDE/mypy hint
        wrapper.__doc__ = (func.__doc__ or "") + f"\n\n DEPRECATED: {reason}\n"
        return wrapper

    return decorator

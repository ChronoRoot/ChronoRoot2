"""Plot-layout gallery: remix real report CSVs into factor scenarios and emit each figure kind."""

from .scenarios import SCENARIOS

__all__ = ['SCENARIOS', 'generate_gallery']


def generate_gallery(*args, **kwargs):
    from .generate import generate_gallery as _generate_gallery
    return _generate_gallery(*args, **kwargs)

"""Garante que a raiz do repositorio esteja no sys.path, para que `import econometria`
funcione mesmo quando o pacote nao foi instalado via `pip install -e .`.

Importe este modulo (`import _bootstrap`) antes de qualquer `from econometria import ...`
em qualquer arquivo de app/ ou app/pages/.
"""
import sys
from pathlib import Path

_RAIZ_REPO = Path(__file__).resolve().parent.parent
if str(_RAIZ_REPO) not in sys.path:
    sys.path.insert(0, str(_RAIZ_REPO))

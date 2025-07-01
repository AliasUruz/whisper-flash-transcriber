"""Funções utilitárias para verificação e instalação de dependências.
"""

from __future__ import annotations

import importlib
import importlib.metadata as importlib_metadata
import subprocess
import sys
from packaging import version
from tkinter import messagebox, Tk

REQUIRED_VERSIONS = {
    "torch": "2.7.1",
    "transformers": "4.52.4",
    "optimum": "1.26.1",
    "numpy": "2.3.0",
    "soundfile": "0.13.1",
}


def _prompt_user(msg: str) -> bool:
    """Exibe um prompt no terminal ou uma janela Tkinter.

    Retorna ``True`` se o usuário aceitar a instalação.
    """
    try:
        root = Tk()
        root.withdraw()
        return messagebox.askyesno("Dependências", msg)
    except Exception:
        resposta = input(f"{msg} [s/N]: ").strip().lower()
        return resposta.startswith("s") or resposta.startswith("y")
    finally:
        if 'root' in locals():
            root.destroy()


def ensure_dependencies() -> None:
    """Verifica dependências críticas e instala versões adequadas se necessário.

    Esta função checa se ``torch``, ``transformers``, ``optimum``, ``numpy`` e
    ``soundfile`` estão instaladas com as versões mínimas especificadas em
    ``REQUIRED_VERSIONS``. Quando alguma dependência está ausente ou com versão
    inferior, é solicitada a permissão do usuário para executarmos ``pip
    install`` com as versões recomendadas.

    Também é verificada a compatibilidade entre ``transformers`` e ``optimum``.
    Se ``transformers`` estiver na versão 4.49 ou superior e ``optimum`` não for
    ao menos a versão 1.26.1, será proposto o ajuste automático.
    """

    missing_or_old = []

    for pkg, min_version in REQUIRED_VERSIONS.items():
        try:
            module = importlib.import_module(pkg)
            installed_version = getattr(module, "__version__", "0")
            if version.parse(installed_version) < version.parse(min_version):
                missing_or_old.append(f"{pkg}>={min_version}")
        except ImportError:
            missing_or_old.append(f"{pkg}>={min_version}")

    # Checar compatibilidade transformers/optimum
    try:
        import transformers
        trans_ver = version.parse(transformers.__version__)
        if trans_ver >= version.parse("4.49"):
            try:
                import optimum
                try:
                    opt_ver_str = getattr(optimum, "__version__", None) or importlib_metadata.version("optimum")
                except Exception:
                    opt_ver_str = "0"
                opt_ver = version.parse(opt_ver_str)
                if opt_ver < version.parse("1.26.1"):
                    missing_or_old.append("optimum>=1.26.1")
            except ImportError:
                missing_or_old.append("optimum>=1.26.1")
    except ImportError:
        # Caso transformers nem esteja instalado, já será capturado acima
        pass

    if not missing_or_old:
        return

    msg = (
        "Dependências faltando ou desatualizadas: "
        + ", ".join(missing_or_old)
        + "\nDeseja instalar/atualizar agora?"
    )
    if _prompt_user(msg):
        for pkg in missing_or_old:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


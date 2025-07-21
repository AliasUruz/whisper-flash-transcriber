import psutil


def get_available_memory_mb() -> int:
    """Retorna memória disponível em megabytes."""
    bytes_disponiveis = psutil.virtual_memory().available
    return int(bytes_disponiveis / (1024 ** 2))


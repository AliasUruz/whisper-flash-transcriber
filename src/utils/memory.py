import psutil


def get_available_memory_mb() -> int:
    """Retorna memória disponível em megabytes."""
    bytes_disponiveis = psutil.virtual_memory().available
    return int(bytes_disponiveis / (1024 ** 2))


def get_total_memory_mb() -> int:
    """Retorna a memória total em megabytes."""
    bytes_totais = psutil.virtual_memory().total
    return int(bytes_totais / (1024 ** 2))


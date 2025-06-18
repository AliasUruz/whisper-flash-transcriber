import logging
from typing import Optional


def select_batch_size(gpu_index: int, fallback: int = 4) -> int:
    """Calcula batch size dinâmico baseado na VRAM disponível."""
    try:
        import torch
    except Exception as e:  # pragma: no cover - proteção caso torch não esteja instalado
        logging.error(f"Torch indisponível: {e}. Usando valor de fallback: {fallback}")
        return fallback

    if not torch.cuda.is_available() or gpu_index < 0:
        logging.info("GPU não disponível ou não selecionada, usando batch size de CPU (4).")
        return fallback

    try:
        device = torch.device(f"cuda:{gpu_index}")
        free_memory_bytes, total_memory_bytes = torch.cuda.mem_get_info(device)
        free_memory_gb = free_memory_bytes / (1024 ** 3)
        total_memory_gb = total_memory_bytes / (1024 ** 3)
        logging.info(
            f"Verificando VRAM para GPU {gpu_index}: {free_memory_gb:.2f}GB livres de {total_memory_gb:.2f}GB."
        )
        if free_memory_gb >= 10.0:
            bs = 32
        elif free_memory_gb >= 6.0:
            bs = 16
        elif free_memory_gb >= 4.0:
            bs = 8
        elif free_memory_gb >= 2.0:
            bs = 4
        else:
            bs = 2
        logging.info(f"VRAM livre ({free_memory_gb:.2f}GB) -> Batch size dinâmico selecionado: {bs}")
        return bs
    except Exception as e:  # pragma: no cover - captura erros ao consultar VRAM
        logging.error(
            f"Erro ao calcular batch size dinâmico: {e}. Usando valor padrão de fallback: {fallback}",
            exc_info=True,
        )
        return fallback

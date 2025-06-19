import logging


def select_batch_size(gpu_index: int, fallback: int = 4) -> int:
    """Calcula batch size dinâmico baseado na VRAM disponível."""
    try:
        import torch
    except Exception as e:  # pragma: no cover - torch pode não estar instalado
        logging.error(
            "Torch indisponível: %s. Usando valor de fallback: %s",
            e,
            fallback,
        )
        return fallback

    if not torch.cuda.is_available() or gpu_index < 0:
        logging.info(
            "GPU não disponível ou não selecionada, usando batch size de CPU"
        )
        return fallback

    try:
        device = torch.device(f"cuda:{gpu_index}")
        free_memory_bytes, total_memory_bytes = torch.cuda.mem_get_info(device)
        free_memory_gb = free_memory_bytes / (1024 ** 3)
        total_memory_gb = total_memory_bytes / (1024 ** 3)
        logging.info(
            "Verificando VRAM para GPU %s: %.2fGB livres de %.2fGB.",
            gpu_index,
            free_memory_gb,
            total_memory_gb,
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
        logging.info(
            "VRAM livre (%.2fGB) -> Batch size dinâmico selecionado: %s",
            free_memory_gb,
            bs,
        )
        return bs
    except Exception as e:  # pragma: no cover - erro ao consultar VRAM
        logging.error(
            (
                "Erro ao calcular batch size dinâmico: %s. "
                "Usando valor padrão de fallback: %s"
            ),
            e,
            fallback,
            exc_info=True,
        )
        return fallback

from PIL import Image, ImageDraw

def create_icon(state="idle", width=64, height=64):
    """
    Gera um ícone dinâmico para a bandeja do sistema.
    States: 'idle', 'recording', 'transcribing'
    """
    # Cores
    bg_color = (0, 0, 0, 0) # Transparente
    outer_color = (255, 255, 255) # Branco
    
    if state == "recording":
        inner_color = (255, 50, 50) # Vermelho
    elif state == "transcribing":
        inner_color = (150, 50, 255) # Roxo
    else: # idle
        inner_color = (200, 200, 200) # Cinza claro

    image = Image.new('RGBA', (width, height), bg_color)
    dc = ImageDraw.Draw(image)

    # Quadrado Maior (Borda Arredondada simulada ou simples)
    margin = width // 8
    dc.rectangle(
        (margin, margin, width - margin, height - margin),
        outline=outer_color,
        width=width // 12
    )

    # Quadrado Menor (Preenchido)
    inner_margin = width // 3
    dc.rectangle(
        (inner_margin, inner_margin, width - inner_margin, height - inner_margin),
        fill=inner_color
    )

    return image

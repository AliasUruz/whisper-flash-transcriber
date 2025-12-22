from PIL import Image, ImageDraw

def create_icon(state: str = "idle", width: int = 64, height: int = 64) -> Image.Image:
    """
    Generates a dynamic icon for the system tray.
    States: 'idle', 'recording', 'transcribing'
    """
    # Colors
    bg_color = (0, 0, 0, 0)  # Transparent
    outer_color = (255, 255, 255)  # White
    
    if state == "recording":
        inner_color = (255, 50, 50)  # Red
    elif state == "transcribing":
        inner_color = (150, 50, 255)  # Purple
    else:  # idle
        inner_color = (200, 200, 200)  # Light gray

    image = Image.new('RGBA', (width, height), bg_color)
    dc = ImageDraw.Draw(image)

    # Outer square (simulated rounded border or simple)
    margin = width // 8
    dc.rectangle(
        (margin, margin, width - margin, height - margin),
        outline=outer_color,
        width=width // 12
    )

    # Inner square (filled)
    inner_margin = width // 3
    dc.rectangle(
        (inner_margin, inner_margin, width - inner_margin, height - inner_margin),
        fill=inner_color
    )

    return image


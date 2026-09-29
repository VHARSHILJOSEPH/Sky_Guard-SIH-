from PIL import Image, ImageDraw
import math

def create_skyguard_icon(size: int) -> Image.Image:
    # Use 4x supersampling for ultra smooth anti-aliased drawing
    scale = 4
    s = size * scale
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Rounded rectangle background
    corner_radius = int(s * 0.22)
    # Background gradient or dark navy container
    draw.rounded_rectangle([0, 0, s - 1, s - 1], radius=corner_radius, fill=(11, 19, 41, 255))
    
    # Border
    border_w = max(2, int(scale * 1.5))
    draw.rounded_rectangle([1, 1, s - 2, s - 2], radius=corner_radius, outline=(45, 212, 191, 110), width=border_w)

    cx = s / 2
    # Shield coordinates (relative to s)
    top_y = s * 0.18
    shoulder_y = s * 0.28
    left_x = s * 0.24
    right_x = s * 0.76
    bottom_y = s * 0.80

    shield_pts = [
        (cx, top_y),
        (right_x, shoulder_y),
        (right_x, s * 0.48),
        (cx + (right_x - cx) * 0.6, s * 0.67),
        (cx, bottom_y),
        (cx - (right_x - cx) * 0.6, s * 0.67),
        (left_x, s * 0.48),
        (left_x, shoulder_y),
    ]

    # Fill shield interior
    draw.polygon(shield_pts, fill=(6, 32, 45, 220))
    # Outline shield
    shield_stroke = max(2, int(scale * 2.5))
    draw.polygon(shield_pts, outline=(45, 212, 191, 255))

    # Radar concentric arcs
    center_pulse_y = s * 0.58

    # Outer arc
    r1 = s * 0.20
    bbox1 = [cx - r1, center_pulse_y - r1, cx + r1, center_pulse_y + r1]
    arc_w1 = max(2, int(scale * 2))
    draw.arc(bbox1, start=205, end=335, fill=(45, 212, 191, 230), width=arc_w1)

    # Inner arc
    r2 = s * 0.12
    bbox2 = [cx - r2, center_pulse_y - r2, cx + r2, center_pulse_y + r2]
    arc_w2 = max(2, int(scale * 1.8))
    draw.arc(bbox2, start=205, end=335, fill=(56, 189, 248, 240), width=arc_w2)

    # Core emitter dot
    r_dot = s * 0.05
    draw.ellipse([cx - r_dot, center_pulse_y - r_dot, cx + r_dot, center_pulse_y + r_dot], fill=(94, 234, 212, 255))

    # Downsample with high quality Lanczos filter
    final_img = img.resize((size, size), Image.Resampling.LANCZOS)
    return final_img

if __name__ == "__main__":
    icon_16 = create_skyguard_icon(16)
    icon_32 = create_skyguard_icon(32)
    icon_48 = create_skyguard_icon(48)
    icon_64 = create_skyguard_icon(64)
    icon_180 = create_skyguard_icon(180)
    icon_192 = create_skyguard_icon(192)

    # Save favicon.ico with multiple resolutions
    icon_32.save(
        "public/favicon.ico",
        format="ICO",
        sizes=[(16, 16), (32, 32), (48, 48), (64, 64)],
        append_images=[icon_16, icon_48, icon_64]
    )
    print("Updated public/favicon.ico successfully.")

    icon_32.save("public/favicon-32x32.png", format="PNG")
    icon_16.save("public/favicon-16x16.png", format="PNG")
    icon_180.save("public/apple-touch-icon.png", format="PNG")
    print("Generated PNG icons successfully.")

import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

SRC_DIR = r"C:\Users\HP\.gemini\antigravity-ide\brain\0d663324-0651-4978-bdbe-ff0977463f0b"
DEST_DIR = r"c:\Users\HP\AI-Fitness\AI_Wellness_Intelligence\assets"
os.makedirs(DEST_DIR, exist_ok=True)

IMAGES = {
    "nova_high": os.path.join(SRC_DIR, "nova_wellness_high_1790064864919.jpg"),
    "nova_good": os.path.join(SRC_DIR, "nova_wellness_good_1790064893269.jpg"),
    "nova_fair": os.path.join(SRC_DIR, "nova_wellness_fair_1790064914558.jpg"),
    "nova_rest": os.path.join(SRC_DIR, "nova_wellness_rest_1790064942441.jpg"),
}

# Save avatar png for user profile / greeting
good_img = Image.open(IMAGES["nova_good"]).convert("RGBA").resize((256, 256), Image.Resampling.LANCZOS)
good_img.save(os.path.join(DEST_DIR, "nova_avatar.png"), "PNG")
print("Saved nova_avatar.png")

TARGET_SIZE = (480, 480)
NUM_FRAMES = 30  # 30 frames at 100ms = 3.0s seamless loop
FRAME_DURATION = 100  # ms

def draw_star(draw, cx, cy, size, color):
    """Draw a 4-point anime sparkle star."""
    points = [
        (cx, cy - size),
        (cx + size * 0.25, cy - size * 0.25),
        (cx + size, cy),
        (cx + size * 0.25, cy + size * 0.25),
        (cx, cy + size),
        (cx - size * 0.25, cy + size * 0.25),
        (cx - size, cy),
        (cx - size * 0.25, cy - size * 0.25),
    ]
    draw.polygon(points, fill=color)

def draw_leaf(draw, cx, cy, size, angle_deg, color):
    """Draw a small stylized anime leaf."""
    leaf_canvas = Image.new("RGBA", (int(size * 3), int(size * 3)), (0, 0, 0, 0))
    ldraw = ImageDraw.Draw(leaf_canvas)
    lcx, lcy = size * 1.5, size * 1.5
    bbox = [lcx - size * 0.5, lcy - size, lcx + size * 0.5, lcy + size]
    ldraw.pieslice(bbox, 45, 225, fill=color)
    ldraw.pieslice(bbox, 225, 45, fill=color)
    rotated = leaf_canvas.rotate(angle_deg, resample=Image.Resampling.BICUBIC)
    return rotated, int(cx - size * 1.5), int(cy - size * 1.5)

def create_looping_gif(image_path, output_path, theme="high"):
    base_raw = Image.open(image_path).convert("RGBA").resize(TARGET_SIZE, Image.Resampling.LANCZOS)
    
    # Particle specs: (base_x, base_y, sway_amp, rise_speed, size, phase, kind)
    if theme == "high":
        particles = [
            (80, 140, 15, 30, 9, 0.0, "star"),
            (400, 160, 18, 35, 11, 0.3, "star"),
            (110, 320, 12, 25, 7, 0.6, "leaf"),
            (380, 310, 14, 28, 8, 0.8, "leaf"),
            (60, 240, 10, 20, 6, 0.2, "sparkle"),
            (420, 240, 12, 22, 7, 0.5, "star"),
            (240, 60, 8, 15, 6, 0.7, "sparkle"),
        ]
        star_color = (70, 220, 255, 220)
        leaf_color = (90, 230, 170, 200)
    elif theme == "good":
        particles = [
            (90, 160, 12, 25, 8, 0.0, "star"),
            (390, 180, 14, 28, 9, 0.35, "star"),
            (120, 340, 10, 20, 7, 0.7, "leaf"),
            (370, 330, 12, 22, 7, 0.5, "leaf"),
            (70, 260, 8, 18, 5, 0.2, "sparkle"),
            (410, 260, 10, 20, 6, 0.8, "sparkle"),
        ]
        star_color = (60, 210, 240, 210)
        leaf_color = (100, 220, 150, 190)
    elif theme == "fair":
        particles = [
            (90, 180, 10, 20, 7, 0.1, "sparkle"),
            (390, 200, 12, 22, 7, 0.4, "sparkle"),
            (100, 320, 8, 18, 6, 0.7, "leaf"),
            (380, 300, 10, 20, 6, 0.3, "leaf"),
        ]
        star_color = (100, 200, 255, 190)
        leaf_color = (120, 210, 160, 180)
    else:  # rest
        particles = [
            (100, 200, 8, 15, 6, 0.0, "sparkle"),
            (380, 210, 10, 16, 6, 0.5, "sparkle"),
            (110, 330, 6, 14, 5, 0.3, "leaf"),
            (370, 320, 8, 15, 5, 0.8, "leaf"),
        ]
        star_color = (140, 190, 255, 170)
        leaf_color = (130, 220, 170, 160)

    frames = []
    w, h = TARGET_SIZE

    for frame_idx in range(NUM_FRAMES):
        t = frame_idx / float(NUM_FRAMES)  # 0.0 to 1.0 (seamless)
        
        # 1. Subtle character breathing & energetic micro-movement
        # Use seamless sine cycles
        bob_y = int(-4.0 * math.sin(2.0 * math.pi * t))
        scale = 1.0 + 0.010 * math.sin(2.0 * math.pi * t)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        scaled_char = base_raw.resize((new_w, new_h), Image.Resampling.BICUBIC)
        
        # Create frame canvas
        frame_canvas = Image.new("RGBA", (w, h), (255, 255, 255, 255))
        
        # Paste centered with bob_y
        paste_x = (w - new_w) // 2
        paste_y = (h - new_h) // 2 + bob_y
        frame_canvas.paste(scaled_char, (paste_x, paste_y), scaled_char)
        
        # 2. Chest emblem glow pulse
        # The leaf badge is around center-chest (x ~ 275, y ~ 270 on scaled character)
        chest_cx = int(w * 0.57)
        chest_cy = int(h * 0.56 + bob_y)
        glow_pulse = 0.5 + 0.5 * math.sin(2.0 * math.pi * t)
        glow_r = int(18 + 8 * glow_pulse)
        
        glow_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        gdraw = ImageDraw.Draw(glow_layer)
        glow_alpha = int(40 + 50 * glow_pulse)
        gdraw.ellipse(
            [chest_cx - glow_r, chest_cy - glow_r, chest_cx + glow_r, chest_cy + glow_r],
            fill=(60, 230, 255, glow_alpha)
        )
        glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(8))
        frame_canvas.alpha_composite(glow_layer)
        
        # 3. Particle layer (stars, sparkles, leaves)
        particle_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        pdraw = ImageDraw.Draw(particle_layer)
        
        for px, py, sway, rise, psize, phase, pkind in particles:
            local_t = (t + phase) % 1.0
            cur_x = px + sway * math.sin(2.0 * math.pi * local_t)
            cur_y = py - rise * local_t
            # wrap seamlessly around vertical zone
            if cur_y < py - rise:
                cur_y += rise
            
            p_alpha_mul = math.sin(math.pi * local_t)  # smooth fade in & out
            if p_alpha_mul < 0:
                p_alpha_mul = 0
            
            cur_size = psize * (0.8 + 0.4 * p_alpha_mul)
            
            if pkind == "star":
                col = (star_color[0], star_color[1], star_color[2], int(star_color[3] * p_alpha_mul))
                draw_star(pdraw, cur_x, cur_y, cur_size, col)
            elif pkind == "leaf":
                col = (leaf_color[0], leaf_color[1], leaf_color[2], int(leaf_color[3] * p_alpha_mul))
                rot_deg = 20 * math.sin(2.0 * math.pi * local_t)
                leaf_img, lx, ly = draw_leaf(pdraw, cur_x, cur_y, cur_size, rot_deg, col)
                particle_layer.alpha_composite(leaf_img, (lx, ly))
            elif pkind == "sparkle":
                col = (star_color[0], star_color[1], star_color[2], int(star_color[3] * p_alpha_mul * 0.9))
                r = cur_size * 0.5
                pdraw.ellipse([cur_x - r, cur_y - r, cur_x + r, cur_y + r], fill=col)
        
        frame_canvas.alpha_composite(particle_layer)
        
        # Convert to RGB with high quality palette quantization
        frame_rgb = frame_canvas.convert("RGB")
        frames.append(frame_rgb)
    
    # Save animated GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=FRAME_DURATION,
        loop=0,
        optimize=True
    )
    file_size_kb = os.path.getsize(output_path) / 1024.0
    print(f"Created {os.path.basename(output_path)} ({file_size_kb:.1f} KB)")

# Generate all 4 looping animations
create_looping_gif(IMAGES["nova_high"], os.path.join(DEST_DIR, "nova_companion_high.gif"), "high")
create_looping_gif(IMAGES["nova_good"], os.path.join(DEST_DIR, "nova_companion_good.gif"), "good")
create_looping_gif(IMAGES["nova_fair"], os.path.join(DEST_DIR, "nova_companion_fair.gif"), "fair")
create_looping_gif(IMAGES["nova_rest"], os.path.join(DEST_DIR, "nova_companion_rest.gif"), "rest")

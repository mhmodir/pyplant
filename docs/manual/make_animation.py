"""Render a 20 s walkthrough animation from the manual screenshots.

    python make_animation.py    ->  pyplant_walkthrough.mp4 and .gif
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
FIG = HERE / "figures"
W, H, FPS = 1280, 720, 25
SLATE, PANEL, ACCENT = (59, 68, 80), (44, 51, 61), (47, 123, 208)
GREEN, RED, WHITE, MUTED = (46, 154, 88), (204, 58, 64), (255, 255, 255), (190, 198, 208)

INTRO, OUTRO, STEP, FADE = 1.5, 1.5, 1.0, 0.25

STEPS = [
    ("s01_scenario", "Run the default scenario", "Quick start > Run scenario",
     ["30 min of plant time, offline", "Level holds, OP +9.6 %"]),
    ("s02_runtime_stopped", "Runtime tab, stack down", "Click Runtime",
     ["4 services: plant, PLC,", "historian, HMI - all down"]),
    ("s03_runtime_services", "Start the stack", "Speed 1x > Start stack",
     ["4/4 services up", "ports 5400 - 5405"]),
    ("s04_runtime_mimic", "Read the healthy plant", "Scroll to Unit 1",
     ["Tank 1 300 mm, valve 66 %", "FT101 = FT102 = 10 L/min"]),
    ("s05_faceplate_trend", "Setpoint step to 340 mm", "LIC101 > Setpoint 340 > Write",
     ["PV 339.7 mm, no overshoot", "OP settles at 68.6 %"]),
    ("s06_faults_empty", "Open the Faults tab", "Click Faults",
     ["One-click presets", "or build your own fault"]),
    ("s07_fault_injected", "Inject a leak", "Leak in tank 1",
     ["faults 1 - leak_168 on TANK1", "5e-5 m2, step at 169 s"]),
    ("s08_truth", "Ground truth", "Scroll to Ground truth",
     ["Instruments agree with truth:", "a process fault, not a sensor"]),
    ("s09_runtime_leak", "Controller fights the leak", "Runtime > LIC101",
     ["OP jumps 68 % -> 93 %", "level dips, no alarm"]),
    ("s10_hmi", "Operator HMI", "Open HMI (port 5405)",
     ["Inflow 18.3 L/min (was 10)", "still no alarm"]),
    ("s11_data_form", "Clear fault, open Data", "Clear all faults > Data",
     ["Presets Quick / Standard /", "Thorough"]),
    ("s12_data_running", "Generate a dataset", "Quick > Generate",
     ["Fast runner, own process", "progress bar fills"]),
    ("s13_data_done", "Dataset ready", "Wait ~45 s",
     ["14 runs, 16 814 samples", "dataset.parquet"]),
    ("s14_model_form", "Set up training", "Click Model",
     ["4 methods ticked", "window 60, quantile 0.995"]),
    ("s15_model_results", "Train and compare", "Train (~12 s)",
     ["PCA selected: 88 % detected", "0 false alarms / day"]),
    ("s16_by_mode", "Results per fault mode", "Scroll to By fault mode",
     ["Leak: 100 % detected, 139 s", "isolated: FLOW_IMBALANCE"]),
    ("s17_stopped", "Stop the stack", "Stop",
     ["0/4 services", "results kept in work/manual"]),
]
TOTAL = INTRO + len(STEPS) * STEP + OUTRO


def font(size, bold=False):
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    try:
        return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)
    except OSError:
        return ImageFont.load_default()


F_BIG, F_TITLE, F_TEXT, F_SMALL = font(64, True), font(38, True), font(26), font(20)
F_NUM = font(120, True)
SHOT_BOX = (40, 40, 720, 640)  # left, top, right, bottom


def load(name):
    img = Image.open(FIG / f"{name}.png").convert("RGB")
    bw, bh = SHOT_BOX[2] - SHOT_BOX[0], SHOT_BOX[3] - SHOT_BOX[1]
    s = min(bw / img.width, bh / img.height)
    return img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)


SHOTS = {name: load(name) for name, *_ in STEPS}


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def progress_bar(d, t):
    d.rectangle([0, H - 8, W, H], fill=PANEL)
    d.rectangle([0, H - 8, int(W * t / TOTAL), H], fill=ACCENT)


def intro_frame(t):
    img = Image.new("RGB", (W, H), SLATE)
    d = ImageDraw.Draw(img)
    a = ease(t / 0.6)
    y = int(250 - 30 * (1 - a))
    d.text((W // 2, y), "pyplant workbench", font=F_BIG, fill=WHITE, anchor="mm")
    d.text((W // 2, y + 70), "from step 1 to a trained fault detector",
           font=F_TEXT, fill=MUTED, anchor="mm")
    d.text((W // 2, y + 150), "scenario: a leak in tank 1", font=F_TITLE,
           fill=(255, 160, 160), anchor="mm")
    flow = ["scenario", "live stack", "inject leak", "dataset", "train", "stop"]
    x0 = W // 2 - (len(flow) * 150) // 2
    for i, label in enumerate(flow):
        on = t > 0.4 + i * 0.12
        x = x0 + i * 150
        d.rounded_rectangle([x + 5, 500, x + 140, 545], radius=8,
                            fill=ACCENT if on else PANEL)
        d.text((x + 72, 522), label, font=F_SMALL, fill=WHITE, anchor="mm")
    return img


def outro_frame(t):
    img = Image.new("RGB", (W, H), SLATE)
    d = ImageDraw.Draw(img)
    a = ease(t / 0.5)
    shot = SHOTS["s15_model_results"].resize((420, int(420 * 0.88)))
    img.paste(shot, (W // 2 - 210, 70 + int(20 * (1 - a))))
    d.text((W // 2, 490), "Leak found by PCA in 43 s - before any alarm",
           font=F_TITLE, fill=WHITE, anchor="mm")
    d.text((W // 2, 545), "python -m pyplant.workbench.app --workspace ./work/manual",
           font=F_TEXT, fill=MUTED, anchor="mm")
    d.text((W // 2, 600), "Full manual: main.pdf  |  Quick guide: quickguide.pdf",
           font=F_SMALL, fill=MUTED, anchor="mm")
    return img


def step_frame(i, t):
    name, title, action, bullets = STEPS[i]
    img = Image.new("RGB", (W, H), SLATE)
    d = ImageDraw.Draw(img)

    shot = SHOTS[name]
    z = 1.0 + 0.05 * (t / STEP)  # slow push-in
    cw, ch = int(shot.width / z), int(shot.height / z)
    crop = shot.crop((0, 0, cw, ch)).resize(shot.size, Image.LANCZOS)
    sx = SHOT_BOX[0] + (SHOT_BOX[2] - SHOT_BOX[0] - shot.width) // 2
    sy = SHOT_BOX[1] + (SHOT_BOX[3] - SHOT_BOX[1] - shot.height) // 2
    d.rectangle([sx - 4, sy - 4, sx + shot.width + 4, sy + shot.height + 4], fill=WHITE)
    img.paste(crop, (sx, sy))

    a = ease(t / 0.35)
    px = 760 + int(30 * (1 - a))
    tone = tuple(int(MUTED[k] * a + SLATE[k] * (1 - a)) for k in range(3))
    white = tuple(int(WHITE[k] * a + SLATE[k] * (1 - a)) for k in range(3))
    colour = RED if name in ("s07_fault_injected", "s08_truth", "s09_runtime_leak",
                             "s10_hmi") else GREEN if name in (
                             "s15_model_results", "s16_by_mode") else ACCENT
    d.text((px, 60), f"{i + 1:02d}", font=F_NUM, fill=colour)
    d.text((px + 175, 95), f"of {len(STEPS)}", font=F_TEXT, fill=tone)
    d.text((px, 220), title, font=F_TITLE, fill=white)
    d.rounded_rectangle([px, 285, px + 480, 335], radius=8, fill=PANEL)
    d.text((px + 14, 310), "> " + action, font=F_TEXT, fill=white, anchor="lm")
    for k, b in enumerate(bullets):
        d.ellipse([px + 2, 382 + k * 48, px + 14, 394 + k * 48], fill=colour)
        d.text((px + 28, 388 + k * 48), b, font=F_TEXT, fill=tone, anchor="lm")

    for k in range(len(STEPS)):  # step dots
        cx = 780 + k * 26
        r = 7 if k == i else 5
        d.ellipse([cx - r, 650 - r, cx + r, 650 + r],
                  fill=colour if k <= i else PANEL)
    return img


def frame_at(t):
    if t < INTRO:
        return intro_frame(t)
    t2 = t - INTRO
    if t2 < len(STEPS) * STEP:
        i, local = int(t2 // STEP), t2 % STEP
        return step_frame(i, local)
    return outro_frame(t2 - len(STEPS) * STEP)


def main():
    n = int(round(TOTAL * FPS))
    frames, prev = [], None
    boundaries = [INTRO + k * STEP for k in range(len(STEPS) + 1)]
    for f in range(n):
        t = f / FPS
        img = frame_at(t)
        for b in boundaries:  # crossfade into each new scene
            if b <= t < b + FADE and f > 0:
                before = frame_at(b - 1e-3)
                img = Image.blend(before, img, ease((t - b) / FADE))
        progress_bar(ImageDraw.Draw(img), t)
        frames.append(img)

    mp4 = HERE / "pyplant_walkthrough.mp4"
    out = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for img in frames:
        out.write(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
    out.release()

    gif = HERE / "pyplant_walkthrough.gif"
    small = [im.resize((800, 450), Image.LANCZOS) for im in frames[::3]]
    small[0].save(gif, save_all=True, append_images=small[1:],
                  duration=int(1000 * 3 / FPS), loop=0, optimize=True)
    print(f"{len(frames)} frames, {len(frames) / FPS:.1f} s -> {mp4.name}, {gif.name}")


if __name__ == "__main__":
    main()

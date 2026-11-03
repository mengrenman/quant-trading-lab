"""Generate app/static/css/themes.css: the dashboard's color themes as CSS variables.

The templates use Tailwind's color classes (bg-slate-900, text-emerald-400, ...).
tailwind.config.js points every shade of the families they use at a variable,
--c-<family>-<shade> (an "R G B" triplet, so opacity modifiers still work), and this
script writes those variables for each theme:

- midnight: Tailwind's own colors, the original look;
- daylight: a light theme; the neutral scale is reversed (dark panels become light) and
  accent shades move so text stays readable on white;
- terminal: neutral grays on black, amber accents, green and red, monospace text.

Independently, data-palette="colorblind" swaps buy/sell green and red for blue and orange
(and the accent and warning colors so they stay distinct), after Okabe and Ito.

Run by `make css`; the palette below is Tailwind v3.4's default.
"""

from pathlib import Path

SHADES = (50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950)
FAMILIES = ("slate", "emerald", "rose", "cyan", "amber", "violet", "sky", "lime")  # used by the templates
ACCENTS = FAMILIES[1:]

# Tailwind's colors, 50 to 950, as hex (bright to dark).
_HEX = {
    "slate": "f8fafc f1f5f9 e2e8f0 cbd5e1 94a3b8 64748b 475569 334155 1e293b 0f172a 020617",
    "neutral": "fafafa f5f5f5 e5e5e5 d4d4d4 a3a3a3 737373 525252 404040 262626 171717 0a0a0a",
    "emerald": "ecfdf5 d1fae5 a7f3d0 6ee7b7 34d399 10b981 059669 047857 065f46 064e3b 022c22",
    "green": "f0fdf4 dcfce7 bbf7d0 86efac 4ade80 22c55e 16a34a 15803d 166534 14532d 052e16",
    "rose": "fff1f2 ffe4e6 fecdd3 fda4af fb7185 f43f5e e11d48 be123c 9f1239 881337 4c0519",
    "red": "fef2f2 fee2e2 fecaca fca5a5 f87171 ef4444 dc2626 b91c1c 991b1b 7f1d1d 450a0a",
    "cyan": "ecfeff cffafe a5f3fc 67e8f9 22d3ee 06b6d4 0891b2 0e7490 155e75 164e63 083344",
    "amber": "fffbeb fef3c7 fde68a fcd34d fbbf24 f59e0b d97706 b45309 92400e 78350f 451a03",
    "yellow": "fefce8 fef9c3 fef08a fde047 facc15 eab308 ca8a04 a16207 854d0e 713f12 422006",
    "violet": "f5f3ff ede9fe ddd6fe c4b5fd a78bfa 8b5cf6 7c3aed 6d28d9 5b21b6 4c1d95 2e1065",
    "sky": "f0f9ff e0f2fe bae6fd 7dd3fc 38bdf8 0ea5e9 0284c7 0369a1 075985 0c4a6e 082f49",
    "lime": "f7fee7 ecfccb d9f99d bef264 a3e635 84cc16 65a30d 4d7c0f 3f6212 365314 1a2e05",
    "blue": "eff6ff dbeafe bfdbfe 93c5fd 60a5fa 3b82f6 2563eb 1d4ed8 1e40af 1e3a8a 172554",
    "orange": "fff7ed ffedd5 fed7aa fdba74 fb923c f97316 ea580c c2410c 9a3412 7c2d12 431407",
    "pink": "fdf2f8 fce7f3 fbcfe8 f9a8d4 f472b6 ec4899 db2777 be185d 9d174d 831843 500724",
}
PALETTE = {family: {shade: tuple(int(code[i:i + 2], 16) for i in (0, 2, 4))
                    for shade, code in zip(SHADES, codes.split(), strict=True)}
           for family, codes in _HEX.items()}

# Which palette family each template family takes its colors from.
SOURCES = {
    "midnight": {f: f for f in FAMILIES},
    "daylight": {f: f for f in FAMILIES},
    "terminal": {**{f: f for f in FAMILIES}, "slate": "neutral", "emerald": "green", "rose": "red",
                 "cyan": "amber", "amber": "yellow", "sky": "cyan"},
}
COLORBLIND = {"emerald": "sky", "rose": "orange", "amber": "yellow", "cyan": "pink"}
COLORBLIND_LIGHT = {"emerald": "blue", "rose": "orange", "amber": "yellow", "cyan": "pink"}

# Light theme: which shade of the source family each class shade shows.
DAYLIGHT_NEUTRAL = {50: 950, 100: 900, 200: 800, 300: 700, 400: 600, 500: 500,
                    600: 400, 700: 300, 800: 200, 900: 100, 950: 50}
DAYLIGHT_ACCENT = {50: 950, 100: 900, 200: 800, 300: 800, 400: 700, 500: 700,
                   600: 800, 700: 800, 800: 200, 900: 100, 950: 50}

SURFACES = {  # page background, cards, borders
    "midnight": {"darkBg": (9, 13, 22), "darkCard": (17, 23, 38), "darkBorder": (30, 41, 59)},
    "daylight": {"darkBg": (241, 245, 249), "darkCard": (255, 255, 255), "darkBorder": (226, 232, 240)},
    "terminal": {"darkBg": (0, 0, 0), "darkCard": (10, 10, 10), "darkBorder": (38, 38, 38)},
}
FONTS = {
    "midnight": "Inter, system-ui, -apple-system, sans-serif",
    "daylight": "Inter, system-ui, -apple-system, sans-serif",
    "terminal": "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace",
}


def shade_of(theme: str, family: str, shade: int) -> int:
    if theme != "daylight":
        return shade
    return (DAYLIGHT_NEUTRAL if family == "slate" else DAYLIGHT_ACCENT)[shade]


def variables(theme: str, sources: dict[str, str]) -> dict[str, tuple[int, int, int]]:
    out = {}
    for family in FAMILIES:
        for shade in SHADES:
            out[f"{family}-{shade}"] = PALETTE[sources[family]][shade_of(theme, family, shade)]
    # Buy and sell: the 500 shade (700 on the light theme) of the families behind emerald and rose.
    strong = 700 if theme == "daylight" else 500
    out["bidGreen"] = PALETTE[sources["emerald"]][strong]
    out["askRed"] = PALETTE[sources["rose"]][strong]
    out["bidBg"] = PALETTE[sources["emerald"]][shade_of(theme, "emerald", 900)]
    out["askBg"] = PALETTE[sources["rose"]][shade_of(theme, "rose", 900)]
    return out


def block(selector: str, values: dict, extra: str = "") -> str:
    body = "".join(f"  --c-{name}: {r} {g} {b};\n" for name, (r, g, b) in values.items())
    return f"{selector} {{\n{body}{extra}}}\n"


def build() -> str:
    parts = ["/* Generated by scripts/build_themes.py (make css): do not edit. */\n"]
    for theme in ("midnight", "daylight", "terminal"):
        selector = ':root, [data-theme="midnight"]' if theme == "midnight" else f'[data-theme="{theme}"]'
        base = {**variables(theme, SOURCES[theme]), **SURFACES[theme]}
        extra = (f"  --font-sans: {FONTS[theme]};\n"
                 f"  color-scheme: {'light' if theme == 'daylight' else 'dark'};\n")
        parts.append(block(selector, base, extra))
        swaps = COLORBLIND_LIGHT if theme == "daylight" else COLORBLIND
        sources = {**SOURCES[theme], **swaps}
        colorblind = {k: v for k, v in variables(theme, sources).items()
                      if k.split("-")[0] in swaps or k in ("bidGreen", "askRed", "bidBg", "askBg")}
        parts.append(block(f'[data-theme="{theme}"][data-palette="colorblind"]', colorblind))
    return "\n".join(parts)


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "app" / "static" / "css" / "themes.css"
    target.write_text(build())
    print(f"wrote {target}")

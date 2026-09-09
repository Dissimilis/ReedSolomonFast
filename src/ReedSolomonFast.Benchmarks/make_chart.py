"""Generate a throughput chart (SVG) from a BenchmarkDotNet competitive log.

No dependencies. Emits a light-background SVG that stays readable in GitHub's dark theme.
Usage: python make_chart.py <bdn-log> <out.svg> [Shape]   (default Shape 10+4)
"""
import math
import re
import sys

# Series to plot, in legend order: (substring of BDN method description, label, colour)
SERIES = [
    ("Witteborn Encode", "ReedSolomon (Witteborn), scalar", "#8c8c8c"),
    ("ReedSolomon.NET Encode", "ReedSolomon.NET, scalar", "#e8863c"),
    ("ours Encode", "ReedSolomonFast (this library)", "#2e7d32"),
]

UNITS = {"ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9}


def parse(path, shape):
    """Pull (description, shard_size, mean_ns) rows for one shape out of the BDN summary table."""
    rows = []
    for line in open(path, encoding="utf-8", errors="replace"):
        if not line.startswith("|") or "Shard_Size" in line or "---" in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 5 or cells[1] != shape:
            continue
        desc, size, mean = cells[0].strip("'"), cells[2], cells[3]
        m = re.match(r"^([\d.,]+)\s*(ns|us|ms|s)$", mean)
        if not m or not size.isdigit():
            continue
        value = float(m.group(1).replace(",", "").replace(" ", ""))
        rows.append((desc, int(size), value * UNITS[m.group(2)]))
    return rows


def main():
    shape = sys.argv[3] if len(sys.argv) > 3 else "10+4"
    data_shards, parity_shards = (int(x) for x in shape.split("+"))
    total = data_shards + parity_shards
    rows = parse(sys.argv[1], shape)
    sizes = sorted({s for _, s, _ in rows})

    # Throughput counts data + parity bytes per operation (klauspost's convention). bytes / ns = GB/s.
    data = {}
    for key, label, colour in SERIES:
        pts = [(s, s * total / ns) for d, s, ns in rows if key in d]
        if pts:
            data[label] = (sorted(pts), colour)

    W, H = 960, 480
    L, R, T, B = 78, 300, 54, 62
    pw, ph = W - L - R, H - T - B

    all_tp = [tp for pts, _ in data.values() for _, tp in pts]
    y_lo, y_hi = math.log10(min(all_tp) * 0.6), math.log10(max(all_tp) * 1.5)
    xs = [math.log2(s) for s in sizes]
    x_lo, x_hi = min(xs), max(xs)

    def X(size):
        return L + (math.log2(size) - x_lo) / (x_hi - x_lo) * pw if x_hi > x_lo else L + pw / 2

    def Y(tp):
        return T + ph - (math.log10(tp) - y_lo) / (y_hi - y_lo) * ph

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
        f'font-family="-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{L}" y="26" font-size="17" font-weight="600" fill="#1b1b1b">'
        f'Reed-Solomon encode throughput on .NET, {shape} shards. Higher is better</text>',
        f'<text x="{L}" y="45" font-size="12" fill="#5a5a5a">'
        f'Single thread, AMD Ryzen 7 PRO 7840U, .NET 10. Data + parity bytes per second, log scale.</text>',
    ]

    ticks = []
    d = math.floor(y_lo)
    while d <= math.ceil(y_hi):
        for mant in (1, 2, 5):
            v = mant * (10 ** d)
            if y_lo <= math.log10(v) <= y_hi:
                ticks.append(v)
        d += 1
    for v in ticks:
        y = Y(v)
        out.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L+pw}" y2="{y:.1f}" stroke="#e9e9e9" stroke-width="1"/>')
        label = f"{v:g}" if v >= 1 else f"{v:.2f}".rstrip("0")
        out.append(f'<text x="{L-10}" y="{y+4:.1f}" font-size="11" fill="#6a6a6a" text-anchor="end">{label}</text>')
    out.append(f'<text x="20" y="{T+ph/2:.0f}" font-size="12" fill="#4a4a4a" '
               f'transform="rotate(-90 20 {T+ph/2:.0f})" text-anchor="middle">Throughput (GB/s, log scale)</text>')

    def human(n):
        if n >= 1 << 20:
            return f"{n >> 20} MiB"
        if n >= 1 << 10:
            return f"{n >> 10} KiB"
        return f"{n} B"

    for s in sizes:
        x = X(s)
        out.append(f'<line x1="{x:.1f}" y1="{T+ph}" x2="{x:.1f}" y2="{T+ph+5}" stroke="#b0b0b0"/>')
        out.append(f'<text x="{x:.1f}" y="{T+ph+20}" font-size="10.5" fill="#6a6a6a" text-anchor="middle">{human(s)}</text>')
    out.append(f'<line x1="{L}" y1="{T+ph}" x2="{L+pw}" y2="{T+ph}" stroke="#9a9a9a" stroke-width="1.2"/>')
    out.append(f'<text x="{L+pw/2:.0f}" y="{H-16}" font-size="12" fill="#4a4a4a" text-anchor="middle">Shard size</text>')

    for label, (pts, colour) in data.items():
        path = " ".join(("M" if i == 0 else "L") + f"{X(s):.1f},{Y(tp):.1f}" for i, (s, tp) in enumerate(pts))
        width = 3.0 if "this library" in label else 1.9
        out.append(f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>')
        for s, tp in pts:
            out.append(f'<circle cx="{X(s):.1f}" cy="{Y(tp):.1f}" r="{3.4 if width > 2 else 2.6}" fill="{colour}"/>')

    ly = T + 8
    out.append(f'<text x="{L+pw+24}" y="{ly}" font-size="11.5" font-weight="600" fill="#3a3a3a">Implementation</text>')
    ly += 18
    for label, (pts, colour) in data.items():
        peak = max(tp for _, tp in pts)
        out.append(f'<line x1="{L+pw+24}" y1="{ly-4}" x2="{L+pw+52}" y2="{ly-4}" stroke="{colour}" stroke-width="3"/>')
        out.append(f'<circle cx="{L+pw+38}" cy="{ly-4}" r="3.2" fill="{colour}"/>')
        weight = "600" if "this library" in label else "400"
        out.append(f'<text x="{L+pw+60}" y="{ly}" font-size="11.5" font-weight="{weight}" fill="#2b2b2b">{label}</text>')
        ly += 15
        out.append(f'<text x="{L+pw+60}" y="{ly}" font-size="10" fill="#7a7a7a">peak {peak:.2f} GB/s</text>')
        ly += 21
    out.append("</svg>")

    open(sys.argv[2], "w", encoding="utf-8").write("\n".join(out))
    print(f"wrote {sys.argv[2]}: {len(data)} series, {len(sizes)} sizes, peak {max(all_tp):.2f} GB/s")


main()

"""Inject the multi-seed sweep results into the slide deck's RESULTS_TABLE marker."""
import json, re

with open('sweep_results.json') as f:
    sweep = json.load(f)

ORDER = ['mlp', 'cnn2d', 'lstm', 'transformer']
METRICS = ['x_pos', 'y_pos', 'x_vel', 'y_vel', 'average']
HEAD = ['X pos', 'Y pos', 'X vel', 'Y vel', 'Average']

# best by average mean
best = max(ORDER, key=lambda k: sweep[k]['metrics']['average']['mean'])

rows = []
for k in ORDER:
    m = sweep[k]['metrics']
    cells = ''.join(
        f"<td>{m[met]['mean']:.3f} <span class='muted small'>± {m[met]['std']:.3f}</span></td>"
        for met in METRICS
    )
    cls = " class='best'" if k == best else ""
    rows.append(f"<tr{cls}><td><strong>{sweep[k]['name']}</strong></td>{cells}</tr>")

table = (
    "<table>\n<thead><tr><th>Model</th>"
    + ''.join(f"<th>{h}</th>" for h in HEAD)
    + "</tr></thead>\n<tbody>\n"
    + '\n'.join(rows)
    + "\n</tbody></table>"
)

with open('presentation/index.html') as f:
    html = f.read()

html = re.sub(
    r"<!-- RESULTS_TABLE -->.*?<!-- /RESULTS_TABLE -->",
    f"<!-- RESULTS_TABLE -->\n    {table}\n    <!-- /RESULTS_TABLE -->",
    html,
    flags=re.DOTALL,
)
with open('presentation/index.html', 'w') as f:
    f.write(html)

print(f"Injected results table. Best model by avg: {sweep[best]['name']}")
for k in ORDER:
    a = sweep[k]['metrics']['average']
    print(f"  {sweep[k]['name']:12s} avg r = {a['mean']:.3f} ± {a['std']:.3f}")

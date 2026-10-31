"""ReorgLab: reorganization energies of hole/electron-transport materials from Gaussian logs.

Developed by Sania Ismaeel.   Run:  streamlit run app.py
"""
from __future__ import annotations

import html
import io
import zipfile

import altair as alt
import pandas as pd
import streamlit as st

import reorg_core as rc

APP_NAME = "ReorgLab"
DEVELOPER = "Sania Ismaeel"
VERSION = "1.0"

UNITS = {  # unit -> (Hartree factor, decimals)
    "eV": (rc.HARTREE_TO_EV, 4),
    "Hartree": (1.0, 7),
    "meV": (rc.HARTREE_TO_EV * 1000, 1),
    "kcal/mol": (rc.HARTREE_TO_KCAL, 2),
}
CALC_MODES = {
    "Holes & electrons": ("hole", "electron"),
    "Holes only (HTM)": ("hole",),
    "Electrons only (ETM)": ("electron",),
}
VIEW_MODES = ("Summary", "Detailed")
CARRIER = {"hole": ("λh", "#3b82c4"), "electron": ("λe", "#e07a1f")}  # colour-blind-safe pair

PALETTES = {
    "light": dict(card="#ffffff", border="#e3e8ee", text="#1f2933", muted="#66727f",
                  head="#f1f4f8", ok="#0f7b4f", ok_bg="#e3f5ec", warn="#9a5b00", warn_bg="#fff1d6"),
    "dark": dict(card="#161d26", border="#2a3441", text="#e6edf3", muted="#93a1b0",
                 head="#1d2631", ok="#5fd39b", ok_bg="#14352a", warn="#f2b155", warn_bg="#3b2d12"),
}
LOG_SUFFIXES = (".log", ".out")


# ----------------------------------------------------------------------------- data
def read_uploads(files) -> tuple[list[rc.LogData], list[str]]:
    """Parse uploaded logs (plain files or .zip archives); return logs and skipped-file messages."""
    raw: list[tuple[str, bytes]] = []
    for f in files:
        if f.name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(f.getvalue())) as zf:
                raw += [(n.rsplit("/", 1)[-1], zf.read(n)) for n in zf.namelist()
                        if n.lower().endswith(LOG_SUFFIXES)]
        else:
            raw.append((f.name, f.getvalue()))
    logs, skipped = [], []
    for name, data in raw:
        try:
            logs.append(rc.parse_log(name, data.decode(errors="ignore")))
        except ValueError as exc:
            skipped.append(f"{name}: {exc}")
    return logs, skipped


def results_frame(results, carriers, unit) -> pd.DataFrame:
    scale = UNITS[unit][0]
    rows = []
    for r in results:
        row = {"Molecule": r.label, "Formula": r.formula}
        for kind in carriers:
            car, symbol = r.carriers[kind], CARRIER[kind][0]
            lam = car.energies().get("lambda")
            row[f"{symbol} ({unit})"] = lam * scale if lam is not None else None
            row[f"{symbol} status"] = "ok" if car.complete else "missing: " + "; ".join(car.missing)
        rows.append(row)
    return pd.DataFrame(rows)


def components_frame(results, carriers, unit) -> pd.DataFrame:
    scale, rows = UNITS[unit][0], []
    for r in results:
        for kind in carriers:
            car = r.carriers[kind]
            if not car.complete:
                continue
            e = car.energies()
            rows.append({
                "Molecule": r.label, "Carrier": kind,
                f"λ1: ion G0→G± ({unit})": e["lambda1"] * scale,
                f"λ2: neutral G±→G0 ({unit})": e["lambda2"] * scale,
                f"λ total ({unit})": e["lambda"] * scale,
                "RMSD G0→ion (Å)": car.rmsd_to_neutral,
                "Ion at G0 from": car.ion_at_neutral_source,
            })
    return pd.DataFrame(rows)


def files_frame(results) -> pd.DataFrame:
    rows = [{
        "Molecule": r.label, "File": log.name, "Recognised as": r.roles[log.name],
        "Charge / mult.": f"{log.charge:+d} / {log.mult}",
        "Job": "optimization" if log.is_opt else "single point",
        "Energy (Hartree)": log.energy,
        "Status": "ok" if (log.terminated and log.converged) else "check",
    } for r in results for log in r.logs]
    return pd.DataFrame(rows).sort_values(["Molecule", "Recognised as"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- rendering
def theme_name() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except (AttributeError, KeyError):
        return "light"


def style_block(p: dict) -> str:
    return f"""<style>
.rl-hero{{padding:1.7rem 2rem;border-radius:18px;color:#fff;margin-bottom:1.1rem;
  background:linear-gradient(120deg,#0b5d57 0%,#1d5fa8 100%);}}
.rl-hero h1{{margin:0;font-size:2.2rem;letter-spacing:-.02em;color:#fff;padding:0}}
.rl-hero p{{margin:.35rem 0 0;font-size:1.02rem;opacity:.92}}
.rl-chip{{display:inline-block;margin-top:.9rem;padding:.25rem .8rem;border-radius:999px;
  background:rgba(255,255,255,.18);font-size:.85rem;font-weight:600}}
.rl-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:.4rem 0 1rem}}
.rl-card{{background:{p['card']};border:1px solid {p['border']};border-radius:14px;padding:.85rem 1rem}}
.rl-card .k{{font-size:.78rem;color:{p['muted']};letter-spacing:.02em}}
.rl-card .v{{font-size:1.55rem;font-weight:700;color:{p['text']};line-height:1.25}}
.rl-card .s{{font-size:.82rem;color:{p['muted']}}}
.rl-wrap{{overflow-x:auto;border:1px solid {p['border']};border-radius:14px;background:{p['card']}}}
table.rl{{border-collapse:collapse;width:100%;font-size:.92rem;color:{p['text']}}}
table.rl th{{background:{p['head']};text-align:left;padding:.6rem .8rem;font-weight:600;
  white-space:nowrap;border-bottom:1px solid {p['border']}}}
table.rl td{{padding:.55rem .8rem;border-bottom:1px solid {p['border']};white-space:nowrap}}
table.rl tr:last-child td{{border-bottom:none}}
table.rl td.num{{font-variant-numeric:tabular-nums;text-align:right}}
table.rl td.best{{font-weight:700}}
.rl-pill{{padding:.1rem .55rem;border-radius:999px;font-size:.78rem;font-weight:600}}
.rl-ok{{color:{p['ok']};background:{p['ok_bg']}}}
.rl-warn{{color:{p['warn']};background:{p['warn_bg']}}}
.rl-foot{{margin-top:2.2rem;padding-top:.9rem;border-top:1px solid {p['border']};
  color:{p['muted']};font-size:.85rem;text-align:center}}
.rl-side{{color:{p['muted']};font-size:.85rem;line-height:1.5}}
</style>"""


def html_table(df: pd.DataFrame, decimals: int, best: dict[str, str] | None = None) -> str:
    """Render a frame as a themed HTML table. Status-like columns become pills; the lowest
    value of each column named in `best` is bold."""
    best = best or {}
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, row in df.iterrows():
        cells = []
        for col, val in row.items():
            if val is None or (isinstance(val, float) and pd.isna(val)):
                cells.append("<td>—</td>")
            elif col.endswith("status") or col == "Status":
                ok = str(val) == "ok"
                cells.append(f"<td><span class='rl-pill {'rl-ok' if ok else 'rl-warn'}'>"
                             f"{html.escape(str(val))}</span></td>")
            elif isinstance(val, float):
                digits = 6 if col == "Energy (Hartree)" else (3 if "RMSD" in col else decimals)
                cls = "num best" if best.get(col) == row["Molecule"] else "num"
                cells.append(f"<td class='{cls}'>{val:.{digits}f}</td>")
            else:
                cells.append(f"<td>{html.escape(str(val))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='rl-wrap'><table class='rl'><tr>{head}</tr>{''.join(rows)}</table></div>"


def metric_cards(items: list[tuple[str, str, str]]) -> str:
    cards = "".join(f"<div class='rl-card'><div class='k'>{k}</div><div class='v'>{v}</div>"
                    f"<div class='s'>{s}</div></div>" for k, v, s in items)
    return f"<div class='rl-cards'>{cards}</div>"


def bar_chart(df: pd.DataFrame, carriers, unit: str, p: dict) -> alt.Chart | None:
    cols = {f"{CARRIER[k][0]} ({unit})": k for k in carriers}
    long = df.melt(id_vars="Molecule", value_vars=list(cols), var_name="Carrier", value_name="λ").dropna()
    if long.empty:
        return None
    colours = [CARRIER[cols[c]][1] for c in cols]
    return (alt.Chart(long, height=300).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(x=alt.X("Molecule:N", title=None, sort=None, axis=alt.Axis(labelAngle=0)),
                    xOffset="Carrier:N",
                    y=alt.Y("λ:Q", title=f"Reorganization energy ({unit})"),
                    color=alt.Color("Carrier:N", scale=alt.Scale(domain=list(cols), range=colours),
                                    legend=alt.Legend(title=None, orient="top")),
                    tooltip=["Molecule", "Carrier", alt.Tooltip("λ:Q", format=".4f")])
            .configure(background="transparent")
            .configure_axis(labelColor=p["muted"], titleColor=p["muted"], gridColor=p["border"],
                            domainColor=p["border"], tickColor=p["border"], labelFontSize=12)
            .configure_legend(labelColor=p["text"], labelFontSize=12)
            .configure_view(stroke=None))


# ----------------------------------------------------------------------------- page
def sidebar() -> tuple[tuple[str, ...], str, str]:
    with st.sidebar:
        st.markdown(f"### ⚛️ {APP_NAME}")
        calc = st.radio("Calculation mode", list(CALC_MODES),
                        help="Which charge carriers to evaluate.")
        view = st.radio("View mode", VIEW_MODES, horizontal=True,
                        help="Summary: results and chart. Detailed: adds contributions, file recognition and checks.")
        unit = st.selectbox("Energy unit", list(UNITS))
        with st.expander("Method"):
            st.markdown("Four-point (Nelsen) scheme, energies from the last SCF cycle.")
            st.latex(r"\lambda_{h}=[E_{+}(G_0)-E_{+}(G_{+})]+[E_{0}(G_{+})-E_{0}(G_0)]")
            st.latex(r"\lambda_{e}=[E_{-}(G_0)-E_{-}(G_{-})]+[E_{0}(G_{-})-E_{0}(G_0)]")
            st.caption("Light/dark theme: ⋮ menu → Settings → Theme.")
        st.markdown(f"<div class='rl-side'>Version {VERSION}<br>Developed by <b>{DEVELOPER}</b></div>",
                    unsafe_allow_html=True)
    return CALC_MODES[calc], view, unit


def main() -> None:
    st.set_page_config(page_title=f"{APP_NAME} · reorganization energies", page_icon="⚛️", layout="wide")
    p = PALETTES[theme_name()]
    st.html(style_block(p))
    carriers, view, unit = sidebar()

    st.html(f"""<div class='rl-hero'><h1>⚛️ {APP_NAME}</h1>
<p>Reorganization energies for hole- and electron-transport materials, straight from your Gaussian log files.
Upload everything together: each file is recognised automatically from its contents.</p>
<span class='rl-chip'>Developed by {DEVELOPER}</span></div>""")

    files = st.file_uploader("Drop Gaussian .log / .out files or a .zip", type=["log", "out", "zip"],
                             accept_multiple_files=True)
    if not files:
        st.info("Upload the neutral, cation and anion optimizations and single points for one or more molecules.")
        footer(p)
        return

    logs, skipped = read_uploads(files)
    for msg in skipped:
        st.warning(f"Skipped {msg}")
    if not logs:
        st.error("No usable Gaussian logs found.")
        return

    results = rc.analyse(logs)
    decimals = UNITS[unit][1]
    table = results_frame(results, carriers, unit)
    n_warn = sum(len(r.warnings) for r in results)
    n_missing = sum(not r.carriers[k].complete for r in results for k in carriers)

    cards = [("Molecules", str(len(results)), f"{len(logs)} files recognised")]
    best: dict[str, str] = {}
    for kind in carriers:
        col = f"{CARRIER[kind][0]} ({unit})"
        valid = table.dropna(subset=[col])
        if valid.empty:
            cards.append((f"Lowest {CARRIER[kind][0]}", "—", "no complete data"))
            continue
        top = valid.loc[valid[col].idxmin()]
        best[col] = top["Molecule"]
        cards.append((f"Lowest {CARRIER[kind][0]}", f"{top[col]:.{decimals}f} {unit}", top["Molecule"]))
    cards.append(("Checks", str(n_warn + n_missing), "warnings / missing items"))
    st.html(metric_cards(cards))

    def results_tab() -> None:
        st.html(html_table(table, decimals, best))
        chart = bar_chart(table, carriers, unit, p)
        if chart is not None:
            st.altair_chart(chart, width="stretch")
        st.download_button("⬇ Download results (CSV)", table.to_csv(index=False),
                           "reorganization_energies.csv", mime="text/csv")

    if view == "Summary":
        results_tab()
        if n_warn or n_missing:
            st.caption("Some checks need attention. Switch to Detailed view to see them.")
    else:
        t1, t2, t3, t4 = st.tabs(["Results", "Contributions", "File recognition",
                                  f"Checks ({n_warn + n_missing})"])
        with t1:
            results_tab()
        with t2:
            comp = components_frame(results, carriers, unit)
            st.html(html_table(comp, decimals) if not comp.empty else "<p>No complete molecules yet.</p>")
        with t3:
            st.html(html_table(files_frame(results), decimals))
        with t4:
            items = [(r.label, f"Missing for {CARRIER[k][0]}: {'; '.join(r.carriers[k].missing)}")
                     for r in results for k in carriers if not r.carriers[k].complete]
            items += [(r.label, w) for r in results for w in r.warnings]
            if not items:
                st.success("All checks passed.")
            for label, text in items:
                st.warning(f"**{label}**: {text}")
            st.caption("All files should share one level of theory (functional, basis set, solvent model). "
                       "The app flags any mismatch. Neutral energies are taken as printed (restricted or unrestricted).")
    footer(p)


def footer(p: dict) -> None:
    st.html(f"<div class='rl-foot'>{APP_NAME} v{VERSION} · Four-point reorganization energies · "
            f"Developed by <b>{DEVELOPER}</b></div>")


if __name__ == "__main__":
    main()

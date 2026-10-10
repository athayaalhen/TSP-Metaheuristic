import io, os, sys, html
import numpy as np, pandas as pd, streamlit as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from tsp_core import *

MAX_CITIES = 200
FONT = "Nunito Sans, Plus Jakarta Sans, system-ui, sans-serif"
C = dict(bg="#FFF8F0", cream="#FFF3E2", soft="#F3DCD3", pink="#E9B1B0", deep="#B4606B", gold="#D9A441", text="#4A2C2C")
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Nunito Sans", "Plus Jakarta Sans", "DejaVu Sans"]})

st.set_page_config(page_title="TSP Optimizer", page_icon="🌸", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito+Sans:wght@400;500;600;700&display=swap');
:root{--bg:#FFF8F0;--panel:#FFFDF9;--peach:#FEEAC9;--rose:#B4606B;--rose-d:#9C4D58;--text:#4A2C2C;--muted:#8A6F6A;--line:#F0E1D2}
.stApp,.stApp p,.stApp label,.stApp button,.stApp input,.stApp textarea,.stApp li,.stApp td,.stApp th,.stApp summary,.stApp h1,.stApp h2,.stApp h3,.stApp h4,[data-testid=stSidebar]{font-family:'Nunito Sans','Plus Jakarta Sans',system-ui,-apple-system,'Segoe UI',sans-serif!important}
.stApp h1,.stApp h2,.stApp h3,.stApp h4{color:var(--text);font-weight:700}
.block-container{padding-top:2rem;max-width:1250px}
.hero{display:flex;gap:16px;align-items:center;background:var(--peach);border-radius:20px;padding:18px 24px;margin-bottom:6px}
.hero .logo{width:50px;height:50px;border-radius:15px;background:var(--rose);display:flex;align-items:center;justify-content:center;font-size:26px}
.hero .t{font-size:1.65rem;font-weight:700;color:var(--text);line-height:1.2}.hero .d{color:var(--muted);font-size:.95rem;margin-top:2px}
.sec{display:flex;align-items:center;gap:10px;margin:30px 0 10px;font-weight:700;font-size:1.2rem;color:var(--text)}
.sec .num{background:var(--rose);color:#fff;border-radius:50%;width:27px;height:27px;display:flex;align-items:center;justify-content:center;font-size:.85rem}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:16px 20px;margin:10px 0;box-shadow:0 2px 10px rgba(120,70,60,.05)}
.panel.draft{border:1.5px dashed #E3C8B8;background:#FFFAF3;box-shadow:none}
.ptitle{font-size:.76rem;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);font-weight:700;margin-bottom:8px}
.mcard{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:12px 16px;min-height:88px;box-shadow:0 2px 8px rgba(120,70,60,.04)}
.ml{font-size:.78rem;color:var(--muted);font-weight:600}.mv{font-size:1.4rem;font-weight:700;color:var(--text);line-height:1.35}.ms{font-size:.78rem;color:var(--muted)}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:2px 0 8px}
.chip{background:var(--peach);color:var(--text);border-radius:999px;padding:3px 12px;font-size:.82rem;font-weight:500}.chip b{font-weight:700;margin-left:6px}
.chip.on{background:var(--rose);color:#fff}.chip.off{background:#F3EBE4;color:#A8958E}
.rl{font-size:.82rem;font-weight:700;color:var(--rose);margin:8px 0 2px}
.badge{display:inline-block;border-radius:999px;padding:3px 12px;font-size:.78rem;font-weight:700;margin-left:10px;vertical-align:middle}
.badge.todo{background:#FFF1D6;color:#9A6A12;border:1px dashed #E0B864}.badge.done{background:#E6F1E4;color:#3F6B3A}.badge.stale{background:#FBE3DF;color:#9C4D58}
.stButton button,.stDownloadButton button{border-radius:12px;border:1px solid var(--line);background:var(--peach);color:var(--text);font-weight:600}
.stButton button:hover,.stDownloadButton button:hover{border-color:var(--rose);color:var(--rose-d)}
.stButton button[kind=primary],.stButton button[data-testid=stBaseButton-primary]{background:var(--rose);color:#fff;border:0;font-size:1.05rem;padding:.65rem 1rem;box-shadow:0 4px 12px rgba(180,96,107,.3)}
.stButton button[kind=primary]:hover,.stButton button[data-testid=stBaseButton-primary]:hover{background:var(--rose-d);color:#fff}
[data-testid=stMetric]{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:12px 18px}
[data-testid=stSidebar]{border-right:1px solid var(--line)}
button[role=tab][aria-selected=true]{color:var(--rose);font-weight:700}
</style>""", unsafe_allow_html=True)
ss = st.session_state
st.markdown('<div class="hero"><div class="logo">🌸</div><div><div class="t">TSP Optimizer</div>'
            '<div class="d">Cari rute terpendek dan adu Simulated Annealing vs Tabu Search. Mulai dari sidebar, lalu ikuti langkah 1 sampai 4.</div></div></div>', unsafe_allow_html=True)

def card(label, value, sub=""): return f'<div class="mcard"><div class="ml">{label}</div><div class="mv">{value}</div><div class="ms">{sub}</div></div>'
def chips(items): return '<div class="chips">' + "".join(f'<span class="chip {c}">{k}{f"<b>{v}</b>" if v != "" else ""}</span>' for k, v, c in items) + "</div>"
def sec(n_, t, badge=""): st.markdown(f'<div class="sec"><span class="num">{n_}</span>{t}{badge}</div>', unsafe_allow_html=True)

# ================= SIDEBAR (ringkas) =================
sb = st.sidebar
sb.markdown("### 🌸 Pengaturan dasar")
sb.markdown("**Data kota**")
method = sb.radio("Metode input data", ["Klik kanvas X-Y", "Generate acak", "Input manual", "Upload Excel/CSV", "Dataset contoh (data1.xlsx)"], label_visibility="collapsed")
df = None
if method == "Klik kanvas X-Y":
    cw = sb.number_input("Lebar kanvas (X max)", 10, 10000, 100); ch = sb.number_input("Tinggi kanvas (Y max)", 10, 10000, 100)
elif method == "Generate acak":
    n_rand = sb.number_input("Jumlah kota", 3, MAX_CITIES, 20); xr = sb.number_input("X max", 10, 10000, 100)
    yr = sb.number_input("Y max", 10, 10000, 100); sd = sb.number_input("Random seed data", 0, 99999, 42)
    g = np.random.default_rng(sd)
    df = pd.DataFrame({"City": range(1, n_rand + 1), "X": g.uniform(0, xr, n_rand).round(2), "Y": g.uniform(0, yr, n_rand).round(2)})
elif method == "Input manual":
    sb.caption("Edit tabel koordinat di halaman utama (tambah/ubah/hapus baris).")
elif method == "Upload Excel/CSV":
    up = sb.file_uploader("File", type=["xlsx", "xls", "csv"])
    if up is not None:
        if up.name.endswith("csv"): raw = pd.read_csv(up)
        else:
            xl = pd.ExcelFile(up); raw = xl.parse(sb.selectbox("Sheet", xl.sheet_names))
        cols = list(raw.columns); num = [c for c in cols if pd.api.types.is_numeric_dtype(raw[c])]
        cc = sb.selectbox("Kolom City/ID", ["(otomatis 1..n)"] + cols)
        cx = sb.selectbox("Kolom X", num, index=0 if num else None); cy = sb.selectbox("Kolom Y", num, index=min(1, len(num) - 1) if num else None)
        if cx and cy:
            df = raw[[cx, cy]].copy(); df.columns = ["X", "Y"]
            df.insert(0, "City", raw[cc] if cc in cols else range(1, len(raw) + 1)); df = df.dropna(subset=["X", "Y"]).reset_index(drop=True)
else:
    xl = pd.ExcelFile(os.path.join(ROOT, "data", "data1.xlsx")); raw = xl.parse("Sheet7")
    case = sb.selectbox("Case", ["Case I", "Case II", "Case III"])
    df = raw[[f"{case} - x", f"{case} - y"]].dropna(); df.columns = ["X", "Y"]
    k = sb.number_input("Pakai N node pertama (6 = ilustrasi Tabu Search)", 3, len(df), len(df))
    df = df.head(k).reset_index(drop=True); df.insert(0, "City", range(1, len(df) + 1))
sb.divider()
sb.markdown("**Algoritma**")
sb.markdown(chips([("Simulated Annealing", "", "on"), ("Tabu Search", "", "on")]), unsafe_allow_html=True)
use_ils = sb.checkbox("Tambahkan Iterated Local Search (ILS)", False)
sb.caption("SA dan Tabu Search selalu dijalankan karena menjadi perbandingan utama. Pengaturan lanjutan ada di halaman utama.")

# ================= 1. DATA KOTA =================
sec(1, "Data kota")
def render_canvas(cw, ch):
    ss.setdefault("clicks", []); ss.setdefault("cv", 0); step = max(cw, ch) / 50
    st.caption("Klik area kosong untuk menambah kota (dibulatkan ke grid). Pilih 'Hapus kota', lalu klik dekat kota untuk menghapusnya.")
    ca, cb, cc_ = st.columns([2, 1, 1]); act = ca.radio("Aksi klik", ["Tambah kota", "Hapus kota"], horizontal=True, label_visibility="collapsed")
    if cb.button("↩️ Undo"): ss.clicks = ss.clicks[:-1]; ss.cv += 1; st.rerun()
    if cc_.button("🗑️ Reset"): ss.clicks = []; ss.cv += 1; st.rerun()
    gx, gy = np.meshgrid(np.arange(0, cw + 1e-9, cw / 50), np.arange(0, ch + 1e-9, ch / 50))
    fg = go.Figure(go.Scatter(x=gx.ravel(), y=gy.ravel(), mode="markers", marker=dict(size=13, color=C["pink"], opacity=0.05),
                              hovertemplate="x=%{x:.1f}, y=%{y:.1f}<extra></extra>", showlegend=False))
    if ss.clicks:
        cp = np.array(ss.clicks)
        fg.add_trace(go.Scatter(x=cp[:, 0], y=cp[:, 1], mode="markers+text", text=[str(i + 1) for i in range(len(cp))], textposition="top center",
                                marker=dict(size=14, color=C["pink"], line=dict(color=C["deep"], width=2)), hoverinfo="skip", showlegend=False))
    fg.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor=C["cream"], paper_bgcolor=C["bg"], dragmode=False, font=dict(family=FONT, color=C["text"]),
                     xaxis=dict(range=[-cw * .02, cw * 1.02], gridcolor=C["soft"]), yaxis=dict(range=[-ch * .02, ch * 1.02], gridcolor=C["soft"]))
    ev = st.plotly_chart(fg, key=f"cv{ss.cv}", on_select="rerun", selection_mode="points", config={"displayModeBar": False})
    pts = (ev or {}).get("selection", {}).get("points", []) if ev else []
    if pts:
        x, y = float(pts[0]["x"]), float(pts[0]["y"])
        if act == "Tambah kota":
            if not any(abs(x - a) + abs(y - b) < step * .5 for a, b in ss.clicks): ss.clicks.append((round(x, 2), round(y, 2)))
        elif ss.clicks:
            dd = [abs(x - a) + abs(y - b) for a, b in ss.clicks]; k = int(np.argmin(dd))
            if dd[k] < step * 3: ss.clicks.pop(k)
        ss.cv += 1; st.rerun()
    return pd.DataFrame({"City": range(1, len(ss.clicks) + 1), "X": [c[0] for c in ss.clicks], "Y": [c[1] for c in ss.clicks]})

if method == "Klik kanvas X-Y": df = render_canvas(cw, ch)
elif method == "Input manual":
    base = pd.DataFrame({"City": [1, 2, 3, 4, 5, 6], "X": [10., 30, 45, 60, 20, 50], "Y": [25., 15, 40, 10, 50, 45]})
    df = st.data_editor(base, num_rows="dynamic", width="stretch", key="manual")
if df is None: st.info("Lengkapi input data di sidebar (mis. unggah file)."); st.stop()
df = df.dropna(subset=["X", "Y"]).reset_index(drop=True)
err = []
if len(df) < 3: err.append("Jumlah kota minimal 3.")
if len(df) > MAX_CITIES: err.append(f"Jumlah kota maksimal {MAX_CITIES}.")
if df["City"].duplicated().any(): err.append("ID kota duplikat.")
if not (pd.api.types.is_numeric_dtype(df["X"]) and pd.api.types.is_numeric_dtype(df["Y"])): err.append("Koordinat harus numerik.")
if err: st.warning(" ".join(err)); st.stop()
n = len(df); P = df[["X", "Y"]].astype(float).values; names = df["City"].astype(str).tolist()

# ---- helper grafik (palet dipakai bersama)
def new_ax(w=5.5, h=4.5):
    f, a = plt.subplots(figsize=(w, h)); f.patch.set_facecolor(C["bg"]); a.set_facecolor(C["cream"]); a.grid(color=C["soft"], lw=.8)
    for sp in a.spines.values(): sp.set_color(C["soft"])
    a.tick_params(colors=C["text"]); return f, a
def draw_cities(a, r, labels):
    a.scatter(P[:, 0], P[:, 1], s=55, c=C["pink"], edgecolors=C["deep"], zorder=3); a.scatter(*P[r[0]], s=170, marker="*", c=C["gold"], edgecolors=C["deep"], zorder=4)
    if labels:
        for i in range(n): a.annotate(names[i], P[i], fontsize=7, color=C["text"], xytext=(4, 4), textcoords="offset points")
def fig_route(r, title, labels=True):
    f, a = new_ax(); rr = list(r) + [r[0]]; a.plot(P[rr, 0], P[rr, 1], "-", color=C["deep"], lw=1.6, zorder=2); draw_cities(a, r, labels)
    a.set_title(f"{title}\nJarak = {length(r, D):.2f}", color=C["text"]); return f
def edges(r): return {frozenset((r[i], r[(i + 1) % len(r)])) for i in range(len(r))}
def fig_diff(r, other, title, color):
    f, a = new_ax(); sh = edges(r) & edges(other)
    for i in range(n):
        u, v = r[i], r[(i + 1) % n]; same = frozenset((u, v)) in sh
        a.plot(P[[u, v], 0], P[[u, v], 1], color=C["soft"] if same else color, lw=2 if same else 3, zorder=2)
    draw_cities(a, r, False); a.set_title(f"{title}: {length(r, D):.2f}", color=C["text"]); return f, len(sh)
def png(f): b = io.BytesIO(); f.savefig(b, format="png", dpi=150, bbox_inches="tight"); return b.getvalue()
def csv(d): return d.to_csv(index=False).encode()

if method != "Klik kanvas X-Y":
    c1, c2 = st.columns([1, 2]); c1.dataframe(df, height=300, width="stretch", hide_index=True)
    D = np.zeros((n, n))   # sementara; jarak sebenarnya dihitung setelah metric dipilih
    f0, a0 = new_ax(); a0.scatter(P[:, 0], P[:, 1], s=55, c=C["pink"], edgecolors=C["deep"])
    for i in range(n): a0.annotate(names[i], P[i], fontsize=7, color=C["text"], xytext=(4, 4), textcoords="offset points")
    a0.set_title("Sebaran kota", color=C["text"]); c2.pyplot(f0)
else:
    with st.expander(f"Tabel koordinat ({n} kota)"): st.dataframe(df, width="stretch", hide_index=True)

# ================= 2. PENGATURAN LANJUTAN =================
sec(2, "Pengaturan algoritma")
with st.expander("⚙️ Pengaturan lanjutan (klik untuk membuka)", expanded=False):
    tab_names = ["Jarak & Solusi Awal", "Local Search", "Simulated Annealing", "Tabu Search"] + (["ILS"] if use_ils else []) + ["Multi-run & Seed"]
    tt = dict(zip(tab_names, st.tabs(tab_names)))
    with tt["Jarak & Solusi Awal"]:
        a, b = st.columns(2)
        metric = a.selectbox("Distance metric", ["Euclidean", "Manhattan"])
        init_mode = a.radio("Rute awal", ["Otomatis (heuristik)", "Manual (urutan dari user)"])
        start_city = b.number_input("Kota awal (nomor baris, mulai 1)", 1, MAX_CITIES, 1)
        manual_ok = init_mode.startswith("Manual")
        if manual_ok: manual_txt = b.text_input("Rute manual (mis. 1,4,2,5,3,6)", "")
        else: manual_txt = ""; cons = b.selectbox("Construction heuristic", ["NN", "NI", "CI", "FI", "AI"], help="NN nearest neighbor, NI nearest insertion, CI cheapest insertion, FI farthest insertion, AI arbitrary insertion")
        if manual_ok: cons = "NN"
    with tt["Local Search"]:
        a, b = st.columns(2)
        use_2opt = a.checkbox("2-opt pada solusi awal", True); ls_after = a.checkbox("2-opt setelah SA/TS (polish)", False)
        ls_strategy = b.selectbox("Improvement strategy", ["Best Improvement", "First Improvement"])
        st.caption("Operator Swap / 2-opt / 3-opt untuk neighborhood SA dan TS dipilih pada tab masing-masing.")
    with tt["Simulated Annealing"]:
        a, b, c3 = st.columns(3)
        sa_T0 = a.number_input("Initial temperature", 0.1, 1e7, 1000.0); sa_alpha = b.number_input("Alpha (cooling rate)", 0.5, 0.9999, 0.90, 0.01, format="%.4f")
        sa_Tmin = c3.number_input("Minimum temperature", 1e-6, 1e5, 0.1, format="%.4f"); sa_per = a.number_input("Iterasi per temperatur", 1, 10000, 50)
        sa_max = b.number_input("Maximum iteration", 10, 1000000, 3000); sa_op = c3.selectbox("Neighborhood SA", ["2-opt", "Swap", "3-opt"])
    with tt["Tabu Search"]:
        a, b, c3 = st.columns(3)
        ts_max = a.number_input("Maximum iteration (TS)", 5, 100000, 300); ts_size = b.number_input("Tabu size", 1, 1000, 10)
        ts_nbr = c3.number_input("Neighborhood size", 3, 5000, 50); ts_op = a.selectbox("Neighborhood TS", ["2-opt", "Swap", "3-opt"])
        ts_asp = b.checkbox("Aspiration criterion", True)
    ils_it = 50
    if use_ils:
        with tt["ILS"]: ils_it = st.number_input("Iterasi ILS", 5, 5000, 50); st.caption("ILS memakai 2-opt dan perturbasi double-bridge; strategi mengikuti tab Local Search.")
    with tt["Multi-run & Seed"]:
        a, b = st.columns(2)
        runs = a.number_input("Multi-run (jumlah run SA & TS)", 1, 50, 1); seed = b.number_input("Random seed algoritma", 0, 99999, 1)
D = dist_matrix(P, metric)

# ================= 3. RINGKASAN =================
def build_init():
    s = int(start_city) - 1
    if s >= n: raise ValueError(f"Kota awal ({int(start_city)}) melebihi jumlah kota ({n}).")
    if manual_ok:
        try: ids = [x.strip() for x in manual_txt.split(",")]; r = [names.index(i) for i in ids]
        except ValueError: raise ValueError("Rute manual berisi ID kota yang tidak ada atau masih kosong.")
        if sorted(r) != list(range(n)): raise ValueError("Rute manual harus memuat setiap kota tepat satu kali.")
        return r
    g = np.random.default_rng(int(seed))
    return nearest_neighbor(D, s) if cons == "NN" else insertion(D, s, cons, g)
try: init_preview, init_err = build_init(), None
except ValueError as e: init_preview, init_err = None, str(e)

cfg = dict(method=method, n=n, metric=metric, init_mode=init_mode, cons=cons, start_city=int(start_city), manual_txt=manual_txt, seed=int(seed), use_2opt=use_2opt,
           ls_strategy=ls_strategy, ls_after=ls_after, sa=(sa_T0, sa_alpha, sa_Tmin, int(sa_per), int(sa_max), sa_op), ts_size=int(ts_size),
           ts=(int(ts_max), int(ts_size), int(ts_nbr), ts_op, ts_asp), use_ils=use_ils, ils_it=int(ils_it), runs=int(runs))
out = ss.get("out")
ran = out is not None and out["n"] == n and np.array_equal(out["P"], P) and out["metric"] == metric
badge = ('<span class="badge todo">Konfigurasi · belum dijalankan</span>' if not ran else
         '<span class="badge done">Sudah dijalankan · konfigurasi sama</span>' if out["cfg"] == cfg else '<span class="badge stale">Konfigurasi berubah sejak run terakhir</span>')
sec(3, "Ringkasan konfigurasi", badge)
st.caption("Ini hanya ringkasan pengaturan yang akan dijalankan. Hasil optimasi muncul di langkah 4 setelah tombol Run ditekan.")
st.markdown('<div class="ptitle">A · Problem overview</div>', unsafe_allow_html=True)
cs = st.columns(4)
for col, (l, v, s_) in zip(cs, [("Jumlah kota", n, "node"), ("Distance metric", metric, ""), ("Metode input", method.split(" (")[0], ""), ("Solusi awal", "Manual" if manual_ok else cons, "rute dari user" if manual_ok else "construction heuristic")]):
    col.markdown(card(l, v, s_), unsafe_allow_html=True)
on = lambda b: "on" if b else "off"
st.markdown('<div class="panel"><div class="ptitle">B · Optimization configuration</div>'
    + chips([("Simulated Annealing", "", "on"), ("Tabu Search", "", "on"), ("ILS", "", on(use_ils)), ("2-opt awal", "ON" if use_2opt else "OFF", on(use_2opt)),
              ("Polish", "ON" if ls_after else "OFF", on(ls_after)), (ls_strategy, "", ""), ("Multi-run", f"{int(runs)}×", on(runs > 1)), ("Seed", int(seed), "")])
    + '<div class="rl">Simulated Annealing</div>' + chips([("T0", sa_T0, ""), ("α", sa_alpha, ""), ("Tmin", sa_Tmin, ""), ("iter/T", int(sa_per), ""), ("maxIter", int(sa_max), ""), ("operator", sa_op, "")])
    + '<div class="rl">Tabu Search</div>' + chips([("maxIter", int(ts_max), ""), ("tabu size", int(ts_size), ""), ("neighborhood", int(ts_nbr), ""), ("operator", ts_op, ""), ("aspiration", "ON" if ts_asp else "OFF", on(ts_asp))])
    + (('<div class="rl">ILS</div>' + chips([("iterasi", int(ils_it), "")])) if use_ils else "") + "</div>", unsafe_allow_html=True)
if init_err: st.error(init_err)
else:
    di = length(init_preview, D); show = init_preview[:14]
    route_chips = chips([(html.escape(names[i]), "", "") for i in show] + ([(f"… +{n - 14} kota", "", "off")] if n > 14 else []))
    cI = st.columns([1, 1, 3])
    cI[0].markdown(card("Jarak rute awal", f"{di:.2f}", "sebelum optimasi"), unsafe_allow_html=True)
    cI[1].markdown(card("Kota awal", html.escape(names[init_preview[0]]), "titik mulai & akhir"), unsafe_allow_html=True)
    cI[2].markdown(f'<div class="mcard"><div class="ml">C · Urutan kota awal</div>{route_chips}</div>', unsafe_allow_html=True)
    if n > 14:
        with st.expander("Lihat seluruh rute awal"): st.write(" → ".join(names[i] for i in init_preview + [init_preview[0]]))

# ================= RUN =================
st.write("")
rc = st.columns([2, 3])
go_run = rc[0].button("▶  RUN OPTIMIZATION", type="primary", width="stretch", disabled=init_err is not None)
rc[1].caption(f"Akan menjalankan SA dan Tabu Search{' + ILS' if use_ils else ''} pada {n} kota" + (f", {int(runs)} kali (multi-run)." if runs > 1 else "."))
if go_run:
    init = init_preview
    start = two_opt_ls(init, D, ls_strategy) if use_2opt else init
    polish = lambda res: {**res, "route": two_opt_ls(res["route"], D, ls_strategy), "dist": length(two_opt_ls(res["route"], D, ls_strategy), D)} if ls_after else res
    def run_sa(s): return polish(simulated_annealing(start, D, sa_T0, sa_alpha, sa_Tmin, int(sa_per), int(sa_max), sa_op, s))
    def run_ts(s): return polish(tabu_search(start, D, int(ts_max), int(ts_size), int(ts_nbr), ts_op, ts_asp, s, detail=(n <= 12)))
    with st.spinner("Menjalankan optimasi..."):
        R = {"SA": run_sa(int(seed)), "Tabu Search": run_ts(int(seed))}
        if use_ils: R["ILS"] = iterated_local_search(start, D, int(ils_it), ls_strategy, int(seed))
        multi = None
        if runs > 1: multi = {m: [f(int(seed) + i) for i in range(int(runs))] for m, f in [("SA", run_sa), ("Tabu Search", run_ts)]}
    ss["out"] = dict(init=init, start=start, R=R, multi=multi, P=P, D=D, names=names, metric=metric, n=n, cfg=cfg)
    st.rerun()

# ================= 4. HASIL =================
out = ss.get("out")
if out is None or out["n"] != n or not np.array_equal(out["P"], P) or out["metric"] != metric:
    sec(4, "Hasil optimasi", '<span class="badge todo">belum ada hasil</span>')
    st.markdown('<div class="panel draft">Belum ada hasil untuk data ini. Atur konfigurasi, lalu tekan <b>RUN OPTIMIZATION</b>.</div>', unsafe_allow_html=True); st.stop()
rc_ = out["cfg"]; stale = rc_ != cfg
sec(4, "Hasil optimasi", '<span class="badge stale">konfigurasi lama</span>' if stale else '<span class="badge done">sesuai konfigurasi</span>')
if stale: st.warning("Pengaturan sudah berubah sejak run terakhir. Hasil di bawah memakai konfigurasi sebelumnya; tekan RUN untuk memperbarui.")
use_2opt, cons, init_mode, ts_size, runs = rc_["use_2opt"], rc_["cons"], rc_["init_mode"], rc_["ts_size"], rc_["runs"]
D = out["D"]; init, start, R, multi = out["init"], out["start"], out["R"], out["multi"]
d0 = length(init, D); ds = length(start, D)

# ================= HASIL =================
sa, ts = R["SA"], R["Tabu Search"]; best_d = min(r["dist"] for r in R.values())
routes = {"Initial": init, **({"Initial + 2-opt": start} if use_2opt else {}), **{m: r["route"] for m, r in R.items()}}
labels = ["📊 Ringkasan", "🗺️ Rute", "⚔️ SA vs TS", "📈 Konvergensi", "📋 Detail & Log", "🧠 Analisis", "🏁 Kompetisi"] + (["🔍 Tabu Explorer"] if n <= 12 else []) + (["🎲 Multi-run"] if multi else [])
T = dict(zip(labels, st.tabs(labels)))

with T["📊 Ringkasan"]:
    a, b, c = st.columns(3); a.metric("Jarak awal (heuristik)", f"{d0:.2f}"); b.metric("Jarak terbaik", f"{best_d:.2f}")
    c.metric("Peningkatan", f"{100 * (d0 - best_d) / d0:.2f}%")
    rows = [{"Metode": "Initial" + (" + 2-opt" if use_2opt else ""), "Jarak Awal": d0, "Jarak Akhir": ds, "Improvement (%)": 100 * (d0 - ds) / d0, "Runtime (s)": 0.0, "Iterasi": 0}]
    for m, r in R.items(): rows.append({"Metode": m, "Jarak Awal": ds, "Jarak Akhir": r["dist"], "Improvement (%)": 100 * (ds - r["dist"]) / ds, "Runtime (s)": r["time"], "Iterasi": r["iters"]})
    st.dataframe(pd.DataFrame(rows).round(4), width="stretch", hide_index=True)

with T["🗺️ Rute"]:
    lab = st.checkbox("Tampilkan label kota", True); sub = st.tabs(list(routes))
    for t, (name, rt) in zip(sub, routes.items()):
        with t:
            f = fig_route(rt, name, lab); st.pyplot(f); st.download_button("💾 Save PNG", png(f), f"route_{name}.png", "image/png", key="png" + name)

with T["⚔️ SA vs TS"]:
    st.caption("Garis tebal berwarna = edge yang HANYA ada di rute tersebut. Garis tipis pink muda = edge yang sama di kedua rute.")
    c1_, c2_ = st.columns(2); fa, sh = fig_diff(sa["route"], ts["route"], "Simulated Annealing", C["deep"]); fb, _ = fig_diff(ts["route"], sa["route"], "Tabu Search", C["gold"])
    c1_.pyplot(fa); c2_.pyplot(fb)
    st.markdown(f'<div class="card">Edge yang sama: <b>{sh} dari {n}</b> ({100 * sh / n:.0f}%). Selisih jarak: <b>{abs(sa["dist"] - ts["dist"]):.2f}</b>.</div>', unsafe_allow_html=True)
    st.download_button("💾 Save PNG SA", png(fa), "diff_SA.png", "image/png"); st.download_button("💾 Save PNG TS", png(fb), "diff_TS.png", "image/png")

with T["📈 Konvergensi"]:
    f, a = new_ax(8, 3.8)
    for (m, r), col in zip(R.items(), [C["deep"], C["gold"], "#B5838D"]): a.plot(r["hist"], label=m, color=col, lw=2)
    a.set_xlabel("Iterasi"); a.set_ylabel("Best distance"); a.legend(); st.pyplot(f); st.download_button("💾 Save PNG konvergensi", png(f), "convergence.png", "image/png")

with T["📋 Detail & Log"]:
    pick = st.selectbox("Pilih rute", list(routes)); rt = routes[pick]
    det = pd.DataFrame({"Step": range(1, n + 1), "From": [names[rt[i]] for i in range(n)], "To": [names[rt[(i + 1) % n]] for i in range(n)],
                        "Distance": [round(D[rt[i], rt[(i + 1) % n]], 4) for i in range(n)]})
    st.dataframe(det, width="stretch", hide_index=True); st.download_button("⬇️ Export rute CSV", csv(det), f"route_{pick}.csv", "text/csv")
    for m, r in R.items():
        lg = pd.DataFrame(r["log"]).drop(columns=["Candidates", "Current_Before", "Current_After"], errors="ignore")
        with st.expander(f"Iteration log {m}"): st.dataframe(lg, width="stretch")
        st.download_button(f"⬇️ Export iteration log {m}", csv(lg), f"log_{m}.csv", "text/csv", key="log" + m)
    dm = pd.DataFrame(D, index=names, columns=names).round(4)
    with st.expander("Distance matrix"): st.dataframe(dm, width="stretch")
    st.download_button("⬇️ Export distance matrix", dm.to_csv().encode(), "distance_matrix.csv", "text/csv")

with T["🧠 Analisis"]:
    win = "Simulated Annealing" if sa["dist"] < ts["dist"] else "Tabu Search" if ts["dist"] < sa["dist"] else None
    conv = lambda r: next(i for i, v in enumerate(r["hist"]) if v <= r["dist"] + 1e-9)
    L = [f"- Solusi awal ({'manual' if init_mode.startswith('Manual') else cons}) berjarak {d0:.2f}" + (f"; setelah 2-opt menjadi {ds:.2f}." if use_2opt else ".")]
    L.append(f"- SA menghasilkan {sa['dist']:.2f} ({100*(ds-sa['dist'])/ds:.2f}% dari solusi awal), TS menghasilkan {ts['dist']:.2f} ({100*(ds-ts['dist'])/ds:.2f}%).")
    L.append(f"- Pada run ini {win} lebih baik, selisih {abs(sa['dist']-ts['dist']):.2f}." if win else "- SA dan TS menghasilkan jarak yang sama.")
    L.append(f"- Runtime: SA {sa['time']:.2f}s, TS {ts['time']:.2f}s ({'SA' if sa['time']<ts['time'] else 'TS'} lebih cepat).")
    L.append(f"- Best distance terakhir diperbarui pada iterasi {conv(sa)} (SA) dan {conv(ts)} (TS) dari {sa['iters']} / {ts['iters']} iterasi.")
    if conv(sa) >= 0.9 * sa["iters"] or conv(ts) >= 0.9 * ts["iters"]: L.append("- Perbaikan masih terjadi mendekati akhir iterasi; pertimbangkan menambah iterasi.")
    L.append(f"- Edge yang sama antara rute SA dan TS: {sh} dari {n}.")
    if multi:
        sd = {m: np.std([x['dist'] for x in v]) for m, v in multi.items()}; mn = {m: np.mean([x['dist'] for x in v]) for m, v in multi.items()}
        L.append(f"- Multi-run ({int(runs)} run): rata-rata SA {mn['SA']:.2f} (SD {sd['SA']:.2f}), TS {mn['Tabu Search']:.2f} (SD {sd['Tabu Search']:.2f}); lebih konsisten: {'SA' if sd['SA']<sd['Tabu Search'] else 'Tabu Search'}.")
    else: L.append("- Hasil SA/TS bergantung pada seed acak; gunakan Multi-run >1 untuk menilai konsistensi.")
    if "ILS" in R: L.append(f"- ILS menghasilkan {R['ILS']['dist']:.2f} dalam {R['ILS']['time']:.2f}s.")
    st.markdown("\n".join(L))

with T["🏁 Kompetisi"]:
    st.caption("Klik kota satu per satu sesuai urutan kunjunganmu (kembali ke kota awal otomatis). Lalu lihat apakah kamu bisa mengalahkan algoritma!")
    sig = (n, hash(P.tobytes()))
    if ss.get("comp_sig") != sig: ss.comp_sig, ss.mine, ss.mv = sig, [], 0
    mine = ss.mine; done = len(mine) == n
    b1, b2, b3 = st.columns([1, 1, 3]); u = b1.button("↩️ Undo", key="cu"); rs = b2.button("🗑️ Reset", key="cr")
    if u and mine: ss.mine = mine[:-1]; ss.mv += 1; st.rerun()
    if rs: ss.mine = []; ss.mv += 1; st.rerun()
    b3.progress(len(mine) / n, text=f"{len(mine)} dari {n} kota dikunjungi")
    fg = go.Figure(go.Scatter(x=P[:, 0], y=P[:, 1], mode="markers+text", textposition="top center", hovertext=names, hoverinfo="text", showlegend=False,
        text=[str(mine.index(i) + 1) if i in mine else names[i] for i in range(n)],
        marker=dict(size=16, color=[C["pink"] if i in mine else C["gold"] for i in range(n)], line=dict(color=C["deep"], width=2))))
    if len(mine) > 1:
        rr = mine + ([mine[0]] if done else []); fg.add_trace(go.Scatter(x=P[rr, 0], y=P[rr, 1], mode="lines", line=dict(color=C["deep"], width=3), hoverinfo="skip", showlegend=False))
    fg.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor=C["cream"], paper_bgcolor=C["bg"], dragmode=False,
                     xaxis=dict(gridcolor=C["soft"]), yaxis=dict(gridcolor=C["soft"]))
    ev = st.plotly_chart(fg, key=f"comp{ss.mv}", on_select="rerun", selection_mode="points", config={"displayModeBar": False})
    pts = ev["selection"]["points"] if ev else []
    if pts and pts[0].get("curve_number", 0) == 0 and not done:
        k = int(pts[0]["point_index"])
        if k not in mine: ss.mine = mine + [k]
        ss.mv += 1; st.rerun()
    if done:
        ud = length(mine, D); rows = [("Rute kamu", ud)] + [("Initial", d0)] + [(m, r["dist"]) for m, r in R.items()]
        tb = pd.DataFrame(rows, columns=["Rute", "Jarak"]); tb["Selisih vs rute kamu (%)"] = 100 * (tb["Jarak"] - ud) / ud
        st.dataframe(tb.round(3), width="stretch", hide_index=True)
        if ud < best_d - 1e-9: st.success(f"🎉 Kamu mengalahkan semua algoritma! ({ud:.2f} < {best_d:.2f})")
        elif abs(ud - best_d) < 1e-9: st.success("🤝 Seri dengan algoritma terbaik!")
        else: st.info(f"Algoritma terbaik lebih pendek {100 * (ud - best_d) / ud:.1f}% dari rutemu ({best_d:.2f} vs {ud:.2f}). Coba lagi!")

if n <= 12:
    with T["🔍 Tabu Explorer"]:
        lg = ts["log"]; it = st.slider("Iterasi", 1, len(lg), 1); e = lg[it - 1]
        st.write(f"**Current:** {' → '.join(map(str, e['Current_Before']))}  |  **Tabu size:** {ts_size}")
        st.dataframe(pd.DataFrame(e["Candidates"]), width="stretch", hide_index=True)
        st.write(f"**Move dipilih:** {e['Selected_Move']} → rute baru {' → '.join(map(str, e['Current_After']))} (jarak {e['Current_Distance']})")
        st.write(f"**Tabu list setelah iterasi {it}:** {e['Tabu_List']}  |  keluar dari list: {e['Left']}")

if multi:
    with T["🎲 Multi-run"]:
        ms = pd.DataFrame({m: {"Best": min(x["dist"] for x in v), "Worst": max(x["dist"] for x in v), "Mean": np.mean([x["dist"] for x in v]),
            "Std Dev": np.std([x["dist"] for x in v]), "Avg Runtime (s)": np.mean([x["time"] for x in v])} for m, v in multi.items()}).T.round(4)
        st.dataframe(ms, width="stretch"); st.download_button("⬇️ Export multi-run", ms.to_csv().encode(), "multirun.csv", "text/csv")

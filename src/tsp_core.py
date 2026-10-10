"""Core TSP: distance, construction heuristics, local search, SA, Tabu Search, ILS."""
import time
import numpy as np

# ---------- distance ----------
def dist_matrix(P, metric="Euclidean"):
    d = P[:, None, :] - P[None, :, :]
    return np.sqrt((d ** 2).sum(2)) if metric == "Euclidean" else np.abs(d).sum(2)

def length(r, D):
    r = np.asarray(r)
    return float(D[r, np.roll(r, -1)].sum())

# ---------- construction heuristics ----------
def nearest_neighbor(D, s):
    r, left = [s], set(range(len(D))) - {s}
    while left:
        c = min(left, key=lambda j: D[r[-1], j]); r.append(c); left.remove(c)
    return r

def insertion(D, s, kind, rng):
    """kind: NI nearest, CI cheapest, FI farthest, AI arbitrary insertion."""
    left = set(range(len(D))) - {s}
    first = (int(rng.choice(sorted(left))) if kind == "AI" else
             max(left, key=lambda j: D[s, j]) if kind == "FI" else min(left, key=lambda j: D[s, j]))
    r = [s, first]; left.remove(first)
    cost = lambda c, p: D[r[p], c] + D[c, r[(p + 1) % len(r)]] - D[r[p], r[(p + 1) % len(r)]]
    while left:
        if kind == "CI":
            c, p = min(((c, p) for c in left for p in range(len(r))), key=lambda x: cost(*x))
        else:
            if kind == "NI": c = min(left, key=lambda j: min(D[j, k] for k in r))
            elif kind == "FI": c = max(left, key=lambda j: min(D[j, k] for k in r))
            else: c = int(rng.choice(sorted(left)))
            p = min(range(len(r)), key=lambda p: cost(c, p))
        r.insert(p + 1, c); left.remove(c)
    return r

# ---------- neighborhood moves ----------
def apply(r, m):
    op, i, j, k = m
    if op == "Swap":
        r = r[:]; r[i], r[j] = r[j], r[i]; return r
    if op == "2-opt": return r[:i] + r[i:j + 1][::-1] + r[j + 1:]
    return r[:i] + r[j:k] + r[i:j] + r[k:]          # 3-opt (segment exchange)

def rand_move(n, op, rng):
    if op == "3-opt":
        i, j, k = sorted(rng.choice(n + 1, 3, replace=False)); return (op, int(i), int(j), int(k))
    i, j = sorted(rng.choice(n, 2, replace=False)); return (op, int(i), int(j), 0)

def neighborhood(n, op, size, rng):
    if op != "3-opt" and n * (n - 1) // 2 <= size:
        return [(op, i, j, 0) for i in range(n) for j in range(i + 1, n)]
    return [rand_move(n, op, rng) for _ in range(size)]

def move_key(r, m):
    op, i, j, k = m
    cs = [r[i], r[j]] + ([r[k % len(r)]] if op == "3-opt" else [])
    return (op, tuple(sorted(cs)))

# ---------- local search ----------
def two_opt_ls(r, D, strategy="Best Improvement", max_pass=200):
    r = r[:]; n = len(r)
    for _ in range(max_pass):
        best = (-1e-9, None)
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b, c, d = r[i - 1], r[i], r[j], r[(j + 1) % n]
                delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
                if delta < best[0]:
                    best = (delta, (i, j))
                    if strategy == "First Improvement": break
            if strategy == "First Improvement" and best[1]: break
        if best[1] is None: break
        i, j = best[1]; r[i:j + 1] = r[i:j + 1][::-1]
    return r

# ---------- Simulated Annealing ----------
def simulated_annealing(r, D, T0=1000, alpha=0.9, Tmin=0.1, per_T=50, max_iter=5000, op="2-opt", seed=0):
    t0 = time.perf_counter(); rng = np.random.default_rng(seed)
    cur, cd = r[:], length(r, D); best, bd = cur[:], cd
    T, it, hist, log = T0, 0, [bd], []
    while T > Tmin and it < max_iter:
        for _ in range(per_T):
            if it >= max_iter: break
            it += 1
            m = rand_move(len(r), op, rng); c = apply(cur, m); d = length(c, D); dl = d - cd
            acc = dl < 0 or rng.random() < np.exp(-dl / T)
            if acc: cur, cd = c, d
            if cd < bd: best, bd = cur[:], cd
            hist.append(bd)
            log.append(dict(Iteration=it, Temperature=round(T, 4), Move=str(m[:3]), Accepted=bool(acc),
                            Current_Distance=round(cd, 4), Best_Distance=round(bd, 4)))
        T *= alpha
    return dict(route=best, dist=bd, hist=hist, log=log, iters=it, time=time.perf_counter() - t0)

# ---------- Tabu Search ----------
def tabu_search(r, D, max_iter=500, tabu_size=10, nbr_size=50, op="2-opt", aspiration=True, seed=0, detail=False):
    """Tabu list = FIFO of moves (city-pair attributes); size = tabu_size, oldest leaves when full."""
    t0 = time.perf_counter(); rng = np.random.default_rng(seed)
    cur, cd = r[:], length(r, D); best, bd = cur[:], cd
    tl, hist, log = [], [bd], []
    for it in range(1, max_iter + 1):
        cands = []
        for m in neighborhood(len(r), op, nbr_size, rng):
            c = apply(cur, m); cands.append((m, c, length(c, D), move_key(cur, m) in tl))
        ok = [x for x in cands if (not x[3]) or (aspiration and x[2] < bd)]
        m, c, d, _ = min(ok or cands, key=lambda x: x[2])
        k = move_key(cur, m); out = None
        if k in tl: tl.remove(k)
        tl.append(k)
        if len(tl) > tabu_size: out = tl.pop(0)
        prev = cur; cur, cd = c, d
        if cd < bd: best, bd = cur[:], cd
        hist.append(bd)
        row = dict(Iteration=it, Selected_Move=str(tuple(v + 1 for v in k[1])), Current_Distance=round(cd, 4), Best_Distance=round(bd, 4),
                   Tabu_List=str([tuple(v + 1 for v in x[1]) for x in tl]),
                   Left=str(tuple(v + 1 for v in out[1])) if out else "-")
        if detail:
            row["Current_Before"] = [x + 1 for x in prev]; row["Current_After"] = [x + 1 for x in cur]
            row["Candidates"] = [dict(Move=str(tuple(sorted((prev[mm[1]] + 1, prev[mm[2]] + 1)))), Route=str([v + 1 for v in cc]),
                                      Distance=round(dd, 4), Tabu="Ya" if tt else "Tidak", Selected="✓" if mm == m else "")
                                 for mm, cc, dd, tt in cands]
        log.append(row)
    return dict(route=best, dist=bd, hist=hist, log=log, iters=max_iter, time=time.perf_counter() - t0)

# ---------- Iterated Local Search ----------
def double_bridge(r, rng):
    n = len(r)
    if n < 8:
        r = r[:]; i, j = rng.choice(n, 2, replace=False); r[i], r[j] = r[j], r[i]; return r
    a, b, c = sorted(int(x) for x in rng.choice(range(1, n), 3, replace=False))
    return r[:a] + r[b:c] + r[a:b] + r[c:]

def iterated_local_search(r, D, iters=100, strategy="Best Improvement", seed=0):
    t0 = time.perf_counter(); rng = np.random.default_rng(seed)
    cur = two_opt_ls(r, D, strategy); cd = length(cur, D); best, bd = cur[:], cd
    hist, log = [bd], []
    for it in range(1, iters + 1):
        c = two_opt_ls(double_bridge(cur, rng), D, strategy); d = length(c, D)
        if d <= cd: cur, cd = c, d
        if cd < bd: best, bd = cur[:], cd
        hist.append(bd); log.append(dict(Iteration=it, Candidate_Distance=round(d, 4), Current_Distance=round(cd, 4), Best_Distance=round(bd, 4)))
    return dict(route=best, dist=bd, hist=hist, log=log, iters=iters, time=time.perf_counter() - t0)

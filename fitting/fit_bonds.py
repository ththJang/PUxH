#!/usr/bin/env python3
"""
Extract r_eq, the inflection point and p from bond dissociation curves.

Input files : two columns -> distance (Angstrom), energy (kcal/mol)

    r_eq  : E'(r) = 0 with E''(r) > 0   (minimum of a PCHIP spline)
    r_inf : E''(r) = 0 with r > r_eq    (backward-stencil finite difference
            on a quintic smoothing spline evaluated at grid points)

    p = (r_eq * w_eq + w_inf * r_inf) / (w_eq + w_inf),
        w_eq = ln(0.51/0.5), w_inf = 4.6052 = ln(100)

r_eq and r_inf may come from different curves: the *_tot rigid scans carry the
better minimum, the *_Q2 bond-energy curves are what p is defined on.

The Q2 curves are smoothed with a quintic (k=5) smoothing spline before
computing discrete second differences. This suppresses noise from the
non-uniform grid insertion without altering the smooth physical shape.
"""

import argparse

import numpy as np
from scipy.interpolate import PchipInterpolator, UnivariateSpline
from scipy.optimize import brentq

W_EQ = np.log(0.51 / 0.5)
W_INF = 4.6052
CITATION = "J. Chem. Phys. 149, 174502 (2018)"
DEFAULT_EXTERNAL = ["OH=0.9587,1.4428"]

# Smoothing parameter for the quintic spline used on Q2 curves.
SMOOTH_S = 1e-4
SMOOTH_K = 5


def p_value(r_eq, r_inf):
    return (r_eq * W_EQ + W_INF * r_inf) / (W_EQ + W_INF)


def load_curve(path):
    data = np.loadtxt(path, comments=("#", "!"), usecols=(0, 1))
    r, e = data[:, 0], data[:, 1]
    order = np.argsort(r)
    r, e = r[order], e[order]
    if np.any(np.diff(r) == 0.0):
        raise ValueError(f"{path}: duplicate distances")
    return r, e


def spline_minimum(r, e):
    """First minimum of a PCHIP spline through the curve."""
    spl = PchipInterpolator(r, e)
    d1, d2 = spl.derivative(1), spl.derivative(2)
    x = np.linspace(r[0], r[-1], 20000)
    y = np.asarray(d1(x), dtype=float)
    s = np.where(np.sign(y[:-1]) * np.sign(y[1:]) < 0)[0]
    r_eq = float(brentq(d1, x[s[0]], x[s[0] + 1])) if s.size else float(r[np.argmin(e)])
    if d2(r_eq) <= 0:
        r_eq = float(r[np.argmin(e)])
    return r_eq, float(spl(r_eq)), float(d2(r_eq))


def inflection(r, e, r_eq):
    """
    Distance where the curvature changes sign, r > r_eq.

    The energy curve is first smoothed with a quintic (k=5) smoothing spline
    (UnivariateSpline, s=SMOOTH_S) to suppress discrete-grid artefacts.
    The smoothed values are evaluated on the original grid and the standard
    backward finite-difference stencil is applied:

        d2[i] = (E_s[i] - 2 E_s[i-1] + E_s[i-2]) / h^2   placed at r[i]
    """
    h = np.diff(r)
    if not np.allclose(h, h[0]):
        raise ValueError("a uniform distance grid is required")
    h = h[0]

    spl = UnivariateSpline(r, e, k=SMOOTH_K, s=SMOOTH_S)
    e_smooth = spl(r)

    d2 = (e_smooth[2:] - 2 * e_smooth[1:-1] + e_smooth[:-2]) / h ** 2
    pos = r[2:]
    m = pos > r_eq
    pos, d2 = pos[m], d2[m]

    s = np.where(np.sign(d2[:-1]) * np.sign(d2[1:]) < 0)[0]
    if s.size == 0:
        return np.nan
    i = s[0]
    return float(pos[i] + h * d2[i] / (d2[i] - d2[i + 1]))


def analyze(label, req_path, rinf_path=None):
    """r_eq from req_path, r_inf from rinf_path (defaults to req_path)."""
    rinf_path = rinf_path or req_path

    r, e = load_curve(req_path)
    r_eq, e_eq, k_eq = spline_minimum(r, e)

    r2, e2 = (r, e) if rinf_path == req_path else load_curve(rinf_path)
    r_inf = inflection(r2, e2, r_eq)

    return {"bond": label, "req_file": req_path, "rinf_file": rinf_path,
            "r_eq": r_eq, "e_eq": e_eq, "k_eq": k_eq,
            "r_inf": r_inf, "p": p_value(r_eq, r_inf), "external": False}


def external(label, r_eq, p_orig):
    """
    Bond whose r_eq and p come from the literature rather than from a curve.

    The published p_orig follows a different bond-order definition, so it is
    converted back to an inflection point

        r_inf = ln(0.6/0.5) * (-(r_eq - p_orig) / 2.3026) + p_orig

    (2.3026 = ln 10) before the usual weighted average is applied.
    """
    r_inf = np.log(0.6 / 0.5) * (-(r_eq - p_orig) / 2.3026) + p_orig
    return {"bond": label, "req_file": "external", "rinf_file": "external",
            "r_eq": float(r_eq), "e_eq": np.nan, "k_eq": np.nan,
            "r_inf": float(r_inf), "p": p_value(r_eq, r_inf),
            "external": True, "p_orig": float(p_orig)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", default=["NH=NH_tot,NH_Q2", "CO=CO_tot,CO_Q2"],
                    help="LABEL=REQFILE,RINFFILE  (default: NH=NH_tot,NH_Q2 CO=CO_tot,CO_Q2)")
    ap.add_argument("--external", action="append", metavar="NAME=r_eq,p_orig",
                    help="bond from the literature (default: %(default)s)")
    args = ap.parse_args()

    externals = args.external if args.external is not None else DEFAULT_EXTERNAL

    res = []
    for spec in args.files:
        if "=" in spec:
            label, paths = spec.split("=", 1)
            parts = [p for p in paths.split(",") if p]
            res.append(analyze(label, parts[0], parts[1] if len(parts) > 1 else None))
        else:
            res.append(analyze(spec, spec))
    for spec in externals:
        label, vals = spec.split("=", 1)
        rq, po = (float(v) for v in vals.split(","))
        res.append(external(label, rq, po))

    print(f"{'bond':<6}{'r_eq from':<10}{'r_inf from':<10}{'r_eq':>10}{'E_min':>11}"
          f"{'k_eq':>10}{'r_inf':>10}{'p':>10}")
    print("-" * 77)
    for x in res:
        emin = "-" if x["external"] else f"{x['e_eq']:.3f}"
        keq = "-" if x["external"] else f"{x['k_eq']:.1f}"
        mark = " *" if x["external"] else ""
        print(f"{x['bond']:<6}{x['req_file']:<10}{x['rinf_file']:<10}"
              f"{x['r_eq']:10.5f}{emin:>11}{keq:>10}"
              f"{x['r_inf']:10.5f}{x['p']:10.5f}{mark}")

    ext = [x for x in res if x["external"]]
    if ext:
        names = ", ".join(x["bond"] for x in ext)
        print(f"\n * {names}: r_eq and p values are calculated from {CITATION}")
        print("   based on different bond order definition.")


if __name__ == "__main__":
    main()

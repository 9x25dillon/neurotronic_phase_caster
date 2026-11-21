#!/usr/bin/env python3
"""Parametric via-fenced or Si-lens ring-slot. Units here are um and GHz.

preflight: geometry and guard bands without AEDT. api-check: inspect installed
PyAEDT without launching AEDT. build: create, strictly validate, optionally solve.
See CHANGELOG.md for the numerical assumptions and the field export convention.
"""

import argparse
import inspect
import json
import re
from dataclasses import dataclass
from importlib.metadata import version
from math import ceil, cos, pi, radians, sin
from pathlib import Path

import numpy as np
from scipy.constants import c, epsilon_0
from scipy.special import jn_zeros
from shapely import affinity, remove_repeated_points
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

VARIANT = "cavity_via"
R2_LAYOUT = "cpw_gap"
PYAEDT_VERSION = "1.10.0"  # L0 locked/inspected distribution interface
RM, RIN, ROUT = 46.6, 44.6, 48.6
TM, TS, CHOKE = 0.8, 70.7, 95.8
MESA, HM, OVERLAP = 1.1, 0.48, 1.5
GAP_LO, GAP_HI = 0.295, 0.3023  # L1 follows the collector stack, in um
F0BAND = (296.1, 307.7)
BANDS = (("low", 1, 100), ("mid", 100, 450), ("high", 450, 950))
DIE = box(-250, -250, 250, 250)


# L0 numerical layout polygonization; maximum ring chord sag is < 0.001 um.
def disk(radius):
    return Point(0.0, 0.0).buffer(radius, quad_segs=256)


@dataclass(frozen=True)
class Design:
    variant: str = VARIANT
    layout: str = R2_LAYOUT
    rc: float = 71.5
    delta: float | None = None

    def __post_init__(self):
        if self.variant not in ("cavity_via", "si_lens") or self.layout not in (
            "cpw_gap",
            "flank_10deg",
        ):
            raise ValueError("Unknown design variant or R2 layout.")
        if not 0 < self.angle < 360 or self.rc <= 5:
            raise ValueError("Invalid radius or arc separation.")

    @property
    def angle(self):
        if self.delta is not None:
            return self.delta
        return 135.0 if self.variant == "cavity_via" else 95.0

    @property
    def choke_angle(self):
        return self.angle / 2 + 180.0


def cavity_modes(rc):
    if rc <= 0:
        raise ValueError("Fence radius must be positive.")
    return {
        (m, n): chi * c / (2 * pi * rc * 1e-6 * np.sqrt(12.5)) / 1e9
        for m in range(4)
        for n, chi in enumerate(jn_zeros(m, 3), start=1)
    }


def guarded_bands():
    return [(h * F0BAND[0] * 0.97, h * F0BAND[1] * 1.03) for h in (1, 2, 3)]


def check_modes(rc):
    modes = cavity_modes(rc)
    hits = [
        f"TM{m}{n}0={freq:.3f} GHz at h={h}"
        for (m, n), freq in modes.items()
        for h, (lo, hi) in enumerate(guarded_bands(), 1)
        if lo <= freq <= hi
    ]
    if hits:
        raise ValueError(f"L1 R_c={rc:g} um violates 3% guards: " + ", ".join(hits))
    return modes


def rot(g, angle):
    return affinity.rotate(g, angle, origin=(0, 0))


def xyz(x, y, z, angle):
    a = radians(angle)
    return [x * cos(a) - y * sin(a), x * sin(a) + y * cos(a), z]


@dataclass
class Film:
    name: str
    active: Polygon
    tabs: tuple

    @property
    def shape(self):
        return unary_union([self.active, *self.tabs])


def layout(d):
    inner, ground = disk(RIN), DIE.difference(disk(ROUT))
    films = []
    phi = d.choke_angle

    def radial_resistor(name, theta, width):
        nonlocal inner, ground
        half = width / 2
        # Flat 10-um gap over the whole resistor width; taper back outside its edges.
        upper = half + 0.25
        outline = Polygon(
            [
                (RM - 5, -upper),
                (RIN, -upper - 2),
                (ROUT, -upper - 2),
                (RM + 5, -upper),
                (RM + 5, upper),
                (ROUT, upper + 2),
                (RIN, upper + 2),
                (RM - 5, upper),
            ]
        )
        flare = rot(outline, theta)
        inner, ground = inner.difference(flare), ground.difference(flare)
        # A flat inner landing is needed at the extreme corners of a straight film.
        inner_pad = rot(box(RM - 8, -upper, RM - 5, upper), theta)
        outer_pad = rot(box(RM + 5, -upper, RM + 8, upper), theta)
        inner, ground = inner.union(inner_pad), ground.union(outer_pad)

        def strip(a, b):
            return rot(box(a, -half, b, half), theta)

        films.append(
            Film(
                name,
                strip(RM - 5, RM + 5),
                (strip(RM - 5 - OVERLAP, RM - 5), strip(RM + 5, RM + 5 + OVERLAP)),
            )
        )

    radial_resistor("R1", d.angle / 2, 10.7)
    if d.layout == "flank_10deg":
        for sign, name in [(-1, "R2a"), (1, "R2b")]:
            radial_resistor(name, phi + sign * 10, 5.35)

    for theta in (0.0, d.angle):
        inner = inner.union(rot(box(RIN - 1, -MESA / 2, RIN, MESA / 2), theta))
    xvia = RM + CHOKE
    cut = box(RIN - 1, -4.5, xvia + 4.0, 4.5)
    if d.layout == "cpw_gap":
        xc = RM + 2.3
        x0, x1 = xc - 5.35 / 2, xc + 5.35 / 2
        a, b = x0 - 0.25, x1 + 0.25
        cut = cut.union(
            Polygon(
                [
                    (a - 2, -4.5),
                    (a, -11.5),
                    (b, -11.5),
                    (b + 2, -4.5),
                    (b + 2, 4.5),
                    (b, 11.5),
                    (a, 11.5),
                    (a - 2, 4.5),
                ]
            )
        )
        for sign, name in [(-1, "R2a"), (1, "R2b")]:

            def across(y0, y1, sign=sign):
                return rot(box(x0, min(sign * y0, sign * y1), x1, max(sign * y0, sign * y1)), phi)

            ground = ground.union(
                rot(
                    box(a, min(sign * 11.5, sign * 14.5), b + 2, max(sign * 11.5, sign * 14.5)), phi
                )
            )
            films.append(Film(name, across(1.5, 11.5), (across(0, 1.5), across(11.5, 13.0))))
    channel = rot(cut, phi)
    ground = ground.difference(channel)
    inner = inner.union(rot(box(RIN - 1.5, -1.5, xvia + 0.75, 1.5), phi))
    riser = rot(box(xvia - 0.6, -0.6, xvia + 0.6, 0.6), phi)
    plate_start = xvia + 4.5
    plate = rot(box(plate_start, -30, plate_start + 60, 30), phi)
    lead = rot(box(xvia - 0.6, -0.6, plate_start, 0.6), phi)
    bias_lead = rot(box(plate_start + 60, -1, plate_start + 62, 1), phi)
    return dict(
        inner=inner,
        ground=ground,
        films=films,
        channel=channel,
        riser=riser,
        plate=plate,
        lead=lead,
        bias_lead=bias_lead,
        bias_x=plate_start + 62,
    )


def fence_angles(d, intervals=None):
    # Section 4 feed exception: a 30-um exit arc, ordinary gaps at most 20 um.
    total = 2 * pi * d.rc
    if intervals is None:
        intervals = ceil((total - 30) / 20)
    if intervals < 2:
        raise ValueError("Via fence cannot be constructed.")
    gap_deg = np.degrees(30 / d.rc)
    return d.choke_angle + np.linspace(gap_deg / 2, 360 - gap_deg / 2, intervals + 1)


@dataclass
class Solid:
    name: str
    fp: Polygon
    z: float
    h: float
    material: str = "gold"


def device_solids(d, g):
    solids = []

    def block(x0, x1, y0, y1, z, h, angle, name, material="gold"):
        solids.append(Solid(name, rot(box(x0, y0, x1, y1), angle), z, h, material))

    for k, theta in enumerate((0.0, d.angle), start=1):
        x = RIN + 1.25
        block(RIN, x + MESA / 2, -MESA / 2, MESA / 2, 0, 0.25, theta, f"P{k}_SubCollector", "pec")
        block(
            x - MESA / 2,
            x + MESA / 2,
            -MESA / 2,
            MESA / 2,
            0.25,
            GAP_LO - 0.25,
            theta,
            f"P{k}_Collector",
            "pec",
        )
        block(
            x - MESA / 2,
            x + MESA / 2,
            -MESA / 2,
            MESA / 2,
            GAP_HI,
            HM - GAP_HI,
            theta,
            f"P{k}_Emitter",
            "pec",
        )
        zbeam = TM + 1.2
        block(x - 0.55, x + 0.55, -0.55, 0.55, HM, zbeam - HM, theta, f"P{k}_EmitterPost")
        start = x - 0.55
        block(start, start + 6.5, -0.6, 0.6, zbeam, TM, theta, f"P{k}_AirBridge")
        block(start + 5.5, start + 6.5, -0.6, 0.6, TM, zbeam - TM, theta, f"P{k}_GroundPost")
    for y0, y1, label in [(-7, -5.5, "A"), (5.5, 7, "B")]:
        block(57, 59, y0, y1, TM, 1.2, d.choke_angle, "StrapPost" + label)
    block(57, 59, -7, 7, TM + 1.2, TM, d.choke_angle, "CPW_Ground_Strap")
    solids.extend(
        [
            Solid("Choke_to_MIM_Via", g["riser"], TM, 0.1),
            Solid("MIM_SiO2_Dielectric", g["plate"], TM, 0.1, "MIM_oxide"),
            Solid("MIM_Top_Plate", g["plate"], TM + 0.1, TM),
            Solid("MIM_Feed_Tongue", g["lead"], TM + 0.1, TM),
            Solid("MIM_Bias_Tongue", g["bias_lead"], TM + 0.1, TM),
        ]
    )
    return solids


def check_volumes(d, g):
    # Exact section-by-section plan of the final post-Boolean extrusions, including metal.
    solids = device_solids(d, g)
    films = [Solid(f.name, f.shape, 0, 0.5, "CrSi") for f in g["films"]]
    mask = unary_union([f.fp for f in films])
    for key in ("inner", "ground"):
        solids.extend(
            [
                Solid(key + "_base", g[key].difference(mask), 0, 0.5),
                Solid(key + "_cap", g[key], 0.5, TM - 0.5),
            ]
        )
    solids.extend(films)
    for i, a in enumerate(solids):
        for b in solids[:i]:
            dz = min(a.z + a.h, b.z + b.h) - max(a.z, b.z)
            if dz > 1e-8 and a.fp.intersection(b.fp).area > 1e-7:
                raise ValueError(f"Positive-volume intersection: {a.name} / {b.name}")
    for a in solids:
        if a.name.endswith("_GroundPost") and a.fp.difference(g["ground"]).area > 1e-7:
            raise ValueError(f"Ungrounded bridge foot: {a.name}")
    return len(solids)


def r2_estimates(d):
    """L1 TL point samples. k is the local voltage divided by ONE RTD voltage."""
    beta = 2 * pi * 300e9 * np.sqrt(6.8) / c * 1e-6
    half_long_arc = RM * radians(360 - d.angle) / 2

    def loss(offset_um):
        k = sin(beta * offset_um) / sin(beta * half_long_arc)
        pair_S = 2 * k * k / 75.0
        return dict(k_local_over_port=k, pair_mS=1e3 * pair_S, per_port_mS=1e3 * pair_S / 2)

    # ±5 um is the review's field-sampling hypothesis. The physical film averages a 2D field.
    return {
        "cpw_gap_review_5um": loss(5.0),
        "flank_10deg": loss(RM * radians(10)),
        "physical_cpw_transverse_center_um": (1.5 + 11.5) / 2,
    }


def preflight(d, intervals=None):
    mode = check_modes(d.rc) if d.variant == "cavity_via" else {}
    g = layout(d)
    inner, outer = g["inner"], g["ground"]
    if inner.geom_type != "Polygon" or outer.geom_type != "Polygon" or inner.distance(outer) < 1e-5:
        raise ValueError("Disconnected metal or an inner-to-ground short.")
    if not g["channel"].covers(g["riser"]) or not inner.covers(g["riser"]):
        raise ValueError("MIM riser outside the signal/CPW channel.")
    if g["plate"].difference(outer).area > 1e-7:
        raise ValueError("MIM lacks a full bottom electrode.")
    for film in g["films"]:
        if (
            film.active.intersection(inner.union(outer)).area > 1e-7
            or film.tabs[0].difference(inner).area > 1e-7
            or film.tabs[1].difference(outer).area > 1e-7
        ):
            raise ValueError(f"{film.name}: shorted active length or incomplete contact.")
    for i, film in enumerate(g["films"]):
        if any(film.shape.intersection(other.shape).area > 1e-7 for other in g["films"][:i]):
            raise ValueError("Resistor footprints overlap.")
    feet = [rot(box(57, -7, 59, -5.5), d.choke_angle), rot(box(57, 5.5, 59, 7), d.choke_angle)]
    if any(foot.difference(outer).area > 1e-7 for foot in feet):
        raise ValueError("Strap foot is not on intact ground.")
    max_r = max(
        np.hypot(x, y) for film in g["films"] for x, y in film.shape.convex_hull.exterior.coords
    )
    info = dict(
        level="L1 nominal geometry",
        checked_top_volumes=check_volumes(d, g),
        theta2_deg=d.angle,
        theta_R2_deg=d.choke_angle,
        max_film_radius_um=max_r,
        plate_area_um2=g["plate"].area,
        R2_L1=r2_estimates(d),
        CMIM_parallel_plate_pF=epsilon_0 * 3.9 * 60e-6**2 / 100e-9 * 1e12,
        DC_limits_ohm={str(rs): (1 / 0.0182 - rs) / 2 for rs in (3.7, 7.0)},
        L1_DC_margins={str(rs): 1 - 18.75 / ((1 / 0.0182 - rs) / 2) for rs in (3.7, 7.0)},
        raw_film_ohm=[
            2e-5 * 10e-6 / ((10.7 if f.name == "R1" else 5.35) * 1e-6 * 0.5e-6) for f in g["films"]
        ],
    )
    if mode:
        for theta in fence_angles(d, intervals):
            cx, cy, _ = xyz(d.rc, 0, 0, theta)
            if Point(cx, cy).distance(outer.boundary) < 5 or not outer.contains(Point(cx, cy)):
                raise ValueError("Via edge intersects a CPW clearance or leaves its ground land.")
        info.update(
            modes_GHz={f"TM{m}{n}0": float(f) for (m, n), f in mode.items()},
            bands_GHz=guarded_bands(),
            inner_via_edge_um=d.rc - 5,
            nominal_clearance_um=d.rc - 5 - (RM + 5 + OVERLAP),
            corner_clearance_um=d.rc - 5 - max_r,
            via_count=len(fence_angles(d, intervals)),
        )
        if info["corner_clearance_um"] <= 0:
            raise ValueError("Fence overlaps resistor envelope.")
    return info


def checked(result, what):
    if result is None or result is False or (isinstance(result, str) and not result):
        raise RuntimeError(f"AEDT call failed: {what}")
    return result


class AedtCalls:
    """One adapter for each changing API. Exact distribution and keyword contracts."""

    signatures = {
        "assign_lumped_rlc_to_sheet": (
            "assignment",
            "start_direction",
            "name",
            "rlc_type",
            "resistance",
            "inductance",
            "capacitance",
        ),
        "insert_infinite_sphere": (
            "definition",
            "phi_start",
            "phi_stop",
            "phi_step",
            "theta_start",
            "theta_stop",
            "theta_step",
            "units",
            "custom_radiation_faces",
            "custom_coordinate_system",
            "use_slant_polarization",
            "polarization_angle",
            "name",
        ),
        "add": (
            "variable",
            "start_point",
            "end_point",
            "step",
            "variation_type",
            "solution",
            "name",
        ),
    }

    @classmethod
    def verify(cls):
        installed = version("pyaedt")
        print(f"L0 installed pyaedt={installed}; requested contract={PYAEDT_VERSION}", flush=True)
        if installed != PYAEDT_VERSION:
            raise RuntimeError("Unsupported PyAEDT release. Install pyaedt==" + PYAEDT_VERSION)
        from ansys.aedt.core import Hfss
        from ansys.aedt.core.modules.design_xploration import ParametricSetups

        for name, expected in cls.signatures.items():
            owner = ParametricSetups if name == "add" else Hfss
            actual = tuple(inspect.signature(getattr(owner, name)).parameters)[1:]
            if actual != expected:
                raise RuntimeError(f"PyAEDT {name} signature mismatch: {actual}")
        return Hfss

    @staticmethod
    def bias_rl(app, sheet, line):
        return checked(
            app.assign_lumped_rlc_to_sheet(
                assignment=sheet.name,
                start_direction=line,
                name="BiasLine",
                rlc_type="Serial",
                resistance=1.0,
                inductance=1e-9,
            ),
            "bias RL",
        )

    @staticmethod
    def sphere(app, step):
        return checked(
            app.insert_infinite_sphere(
                definition="Theta-Phi",
                phi_start=0,
                phi_stop=360,
                phi_step=step,
                theta_start=0,
                theta_stop=180,
                theta_step=step,
                name="Broadside",
            ),
            "sphere",
        )

    @staticmethod
    def parametric(app, lo, hi, step):
        return checked(
            app.parametrics.add(
                variable="R_c",
                start_point=lo,
                end_point=hi,
                step=step,
                variation_type="LinearStep",
                solution="Driven : LastAdaptive",
                name="GuardedRc",
            ),
            "R_c parametric setup",
        )


def sheet(model, coords, name, material="vacuum"):
    return checked(
        model.create_polyline(
            points=coords, cover_surface=True, close_surface=True, name=name, material=material
        ),
        name,
    )


def prism(model, poly, z, height, name, material):
    """Extrude a polygon; subtract its holes explicitly so material volumes are disjoint."""
    poly = remove_repeated_points(poly, tolerance=1e-7)
    if poly.geom_type != "Polygon" or not poly.is_valid:
        raise ValueError(f"Malformed {name} polygon ({poly.geom_type}).")

    def extrude(ring, label):
        pts = [[float(x), float(y), z] for x, y in list(ring.coords)[:-1]]
        obj = sheet(model, pts, label, material)
        return checked(model.sweep_along_vector(obj, [0, 0, height]), "extrude " + label)

    body = extrude(poly.exterior, name)
    for k, hole in enumerate(poly.interiors):
        tool = extrude(hole, f"{name}_void{k}")
        checked(model.subtract(body, tool, keep_originals=False), "subtract hole " + name)
    return body


def make_materials(app):
    for name, ep, sigma in [
        ("InP_SI", 12.5, 0.0),
        ("Si_lens", 11.7, 0.0),
        ("MIM_oxide", 3.9, 0.0),
        ("CrSi", 1.0, 1 / 2e-5),
    ]:
        material = checked(app.materials.add_material(name), "material " + name)
        material.permittivity = ep
        material.conductivity = sigma


def geom(app, d, bias_port, open_r2, intervals):
    model = app.modeler
    model.model_units = "um"
    make_materials(app)
    g = layout(d)
    sub = checked(
        model.create_box([-250, -250, -TS], [500, 500, TS], name="InP_die", material="InP_SI"),
        "InP",
    )
    outer = prism(model, g["ground"], 0, TM, "Outer_Ground_Plane", "gold")
    inner = prism(model, g["inner"], 0, TM, "Inner_Electrode_CPW", "gold")
    # Film first, Au over the contact tabs. The exposed current length stays exactly 10 um.
    for film in g["films"]:
        parts = film.tabs if open_r2 and film.name.startswith("R2") else (film.shape,)
        for k, fp in enumerate(parts):
            body = prism(model, fp, 0, 0.5, film.name + str(k), "CrSi")
            body.solve_inside = True
            for metal, footprint in [(outer, g["ground"]), (inner, g["inner"])]:
                if fp.intersection(footprint).area > 1e-7:
                    checked(
                        model.subtract(metal, body, keep_originals=True),
                        "trim Au over " + body.name,
                    )
    if d.variant == "cavity_via":
        app["R_c"] = f"{d.rc:.12g}um"
        checked(
            model.create_box(
                [-250, -250, -TS - 0.5], [500, 500, 0.5], name="Back_reflector", material="gold"
            ),
            "reflector",
        )
        vias = []
        for k in range(intervals + 1):
            u = k / intervals
            # Keep the 30-um exit arc fixed even during a native R_c sweep.
            a = f"({d.choke_angle + 360 * u:.15g}deg+({0.5 - u:.15g})*(30um/R_c)*1rad)"
            obj = checked(
                model.create_cylinder(
                    "Z",
                    [f"R_c*cos({a})", f"R_c*sin({a})", -TS],
                    5,
                    TS,
                    name=f"Fence_{k:02d}",
                    material="gold",
                ),
                "via",
            )
            vias.append(obj.name)
        checked(model.subtract(sub, vias, keep_originals=True), "remove via holes from InP")
    else:
        lens = checked(
            model.create_sphere([0, 0, -TS], 3000, name="Si_hemisphere", material="Si_lens"),
            "Si lens",
        )
        tool = checked(
            model.create_box([-3001, -3001, -TS], [6002, 6002, 3001], name="Lens_upper_cut"),
            "hemisphere tool",
        )
        checked(model.subtract(lens, tool, keep_originals=False), "hemisphere lower half")

    for obj in device_solids(d, g):
        prism(model, obj.fp, obj.z, obj.h, obj.name, obj.material)
    for k, theta in enumerate((0.0, d.angle), start=1):
        x = RIN + 1.25
        # A vertical sheet on the side of the active gap; integration from collector to emitter.
        corners = [
            xyz(x + MESA / 2, y, z, theta)
            for y, z in [(-0.55, GAP_LO), (0.55, GAP_LO), (0.55, GAP_HI), (-0.55, GAP_HI)]
        ]
        port = sheet(model, corners, f"P{k}_BarrierSheet")
        line = [xyz(x + MESA / 2, 0, z, theta) for z in (GAP_LO, GAP_HI)]
        checked(
            app.lumped_port(
                assignment=port.name,
                integration_line=line,
                impedance=50,
                name=f"P{k}",
                renormalize=True,
            ),
            "barrier port",
        )

    phi = d.choke_angle
    x = g["bias_x"]
    corners = [xyz(x, y, z, phi) for y, z in [(-1, TM), (1, TM), (1, TM + 0.1), (-1, TM + 0.1)]]
    pad = sheet(model, corners, "BiasPadReference")
    line = [xyz(x, 0, z, phi) for z in (TM, TM + 0.1)]
    if bias_port:
        checked(
            app.lumped_port(
                assignment=pad.name,
                integration_line=line,
                impedance=50,
                name="Bias",
                renormalize=True,
            ),
            "bias access port",
        )
    else:
        AedtCalls.bias_rl(app, pad, line)


def configure(app, d, step, rc_sweep):
    checked(app.create_open_region(frequency="300GHz", boundary="FEBI"), "free-space FE-BI region")
    setup = checked(app.create_setup(name="Driven", setup_type="HFSSDriven"), "driven setup")
    checked(
        setup.enable_adaptive_setup_multifrequency([300, 600, 900], max_delta_s=0.01),
        "adaptive mesh",
    )
    for label, lo, hi in BANDS:
        checked(
            app.create_linear_count_sweep(
                setup="Driven",
                name=label,
                unit="GHz",
                start_frequency=lo,
                stop_frequency=hi,
                num_of_freq_points=ceil((hi - lo) / step) + 1,
                sweep_type="Discrete",
                save_fields=True,
                save_rad_fields=True,
            ),
            "sweep " + label,
        )
    AedtCalls.sphere(app, 1 if d.variant == "si_lens" else 2)
    if rc_sweep:
        lo, hi, st = rc_sweep
        AedtCalls.parametric(app, lo, hi, st)


def trace(app, sweep, expressions, category=None, angles=None, units="dimensionless"):
    nominal = app.available_variations.nominal_variation(dependent_params=False)
    variations = {**nominal, "Freq": ["All"], **(angles or {})}
    kwargs = dict(
        expressions=expressions,
        setup_sweep_name="Driven : " + sweep,
        primary_sweep_variable="Freq",
        variations=variations,
    )
    if category:
        kwargs.update(report_category=category, context="Broadside")
    if category == "Antenna Parameters":
        # In the locked API the generic context setter only binds a sphere for Far Fields.
        report = checked(
            app.post.reports_by_category.antenna_parameters(
                expressions=expressions, setup="Driven : " + sweep, infinite_sphere="Broadside"
            ),
            "antenna parameter report",
        )
        report.primary_sweep = "Freq"
        report.variations = variations
        data = checked(report.get_solution_data(), "antenna power data")
    else:
        data = checked(app.post.get_solution_data(**kwargs), f"{sweep} {category} {expressions}")
    fscale = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9, "THz": 1e12}[data.units_sweeps["Freq"]]
    scales = {
        "dimensionless": {"": 1.0, "1": 1.0},
        "power": {"W": 1.0, "mW": 1e-3, "uW": 1e-6, "nW": 1e-9, "pW": 1e-12, "fW": 1e-15},
        "voltage": {"V": 1.0, "mV": 1e-3, "uV": 1e-6, "nV": 1e-9},
    }
    freq = None
    values = {}
    for expr in expressions:
        x, re0 = data.get_expression_data(expression=expr, formula="real")
        x, re0 = np.asarray(x, dtype=float) * fscale, np.asarray(re0, dtype=float)
        if x.size < 2 or len(x) != len(re0) or not np.all(np.isfinite(re0)):
            raise RuntimeError(f"Incomplete HFSS report: {expr} in {sweep}")
        scale = scales[units][data.units_data[expr] or ""]
        if units == "power":
            val = re0
        else:
            xi, im = data.get_expression_data(expression=expr, formula="imag")
            if len(im) != len(x) or not np.array_equal(np.asarray(xi) * fscale, x):
                raise RuntimeError(f"Missing complex field/S phase: {expr}")
            val = re0 + 1j * np.asarray(im)
        ix = np.argsort(x)
        if np.any(np.diff(x[ix]) <= 0) or not np.all(np.isfinite(val)):
            raise RuntimeError(f"Invalid report grid or values: {expr}")
        if freq is not None and not np.array_equal(freq, x[ix]):
            raise RuntimeError("Inconsistent frequency vectors in one report.")
        freq = x[ix]
        values[expr] = val[ix] * scale
    return freq, values


def canonical_touchstone(app, label, path, ports):
    """Validate HFSS's raw export against explicitly named traces, then order by port name."""
    import itertools

    import skrf

    checked(
        app.export_touchstone(
            setup="Driven", sweep=label, output_file=str(path), renormalization=True, impedance=50
        ),
        "export " + path.name,
    )
    if not path.is_file() or not path.stat().st_size:
        raise RuntimeError(f"HFSS did not write {path}")
    raw = skrf.Network(str(path))
    expr = [f"S({a},{b})" for a in ports for b in ports]
    f, tr = trace(app, label, expr)
    n = len(ports)
    s = np.stack([tr[e] for e in expr], axis=-1).reshape(-1, n, n)
    if raw.nports != n or raw.f.shape != f.shape or not np.allclose(raw.f, f, rtol=1e-9, atol=1):
        raise RuntimeError("Named HFSS report and Touchstone grids disagree.")
    if not any(
        np.allclose(raw.s[:, p, :][:, :, p], s, rtol=1e-4, atol=2e-6)
        for p in itertools.permutations(range(n))
    ):
        raise RuntimeError(
            "Port names, reference planes or normalization inconsistent with raw HFSS S."
        )
    ordered = skrf.Network(frequency=skrf.Frequency.from_f(f, unit="hz"), s=s, z0=50)
    ordered.port_names = ports
    ordered.write_touchstone(filename=str(path), form="ri", write_z0=False)
    return ordered


def radiation_rows(app, label, network, d):
    from sim_300ghz_rtd_cosim import axial_ratio

    fields = {}
    power = {}
    accepted = {}
    theta = "0deg" if d.variant == "cavity_via" else "180deg"
    # For modal sources the magnitude is incident power in watts, phase in degrees.
    drives = {
        "one": (1, 0, 0),
        "two": (0, 1, 0),
        "plus": (1, 1, 0),
        "quad": (1, 1, 90),
        "odd": (1, 1, 180),
    }
    sources = app.get_all_sources()
    source_map = {s.split(":")[0]: s for s in sources}
    if set(source_map) != {"P1", "P2"}:
        raise RuntimeError(f"Unexpected HFSS source labels: {sources}")

    def same_grid(f):
        if f.shape != network.f.shape or not np.allclose(f, network.f, rtol=1e-9, atol=2.0):
            raise RuntimeError(f"Far-field report does not cover {label} network frequencies.")

    for name, (p1, p2, phase) in drives.items():
        # GetAllSources may omit the one-mode suffix; EditSources requires modal names.
        def modal(p):
            src = source_map[p]
            return src if ":" in src else src + ":1"

        checked(
            app.edit_sources(
                {modal("P1"): (f"{p1}W", "0deg"), modal("P2"): (f"{p2}W", f"{phase}deg")},
                include_port_post_processing=True,
            ),
            "source " + name,
        )
        f, p = trace(
            app, label, ["RadiatedPower", "AcceptedPower"], "Antenna Parameters", units="power"
        )
        same_grid(f)
        power[name], accepted[name] = p["RadiatedPower"], p["AcceptedPower"]
        a = np.array([p1**0.5, p2**0.5 * np.exp(1j * np.deg2rad(phase))])
        expected = np.vdot(a, a).real - np.sum(abs(network.s @ a) ** 2, axis=1)
        if not np.allclose(expected, accepted[name], rtol=0.01, atol=5e-9):
            raise RuntimeError("HFSS accepted power disagrees with incident minus reflected power.")
        if name in ("one", "two", "odd"):
            f, field = trace(
                app,
                label,
                ["rETheta", "rEPhi"],
                "Far Fields",
                {"Theta": [theta], "Phi": ["0deg"]},
                "voltage",
            )
            same_grid(f)
            fields[name] = np.column_stack((field["rETheta"], field["rEPhi"]))
    base = power["one"] + power["two"]
    cross = (power["plus"] - base + 1j * (base - power["quad"])) / 2
    if not np.allclose(fields["one"] - fields["two"], fields["odd"], rtol=0.005, atol=1e-10):
        raise RuntimeError("Broadside Jones basis does not reproduce the direct odd excitation.")
    if not np.allclose(base - 2 * cross.real, power["odd"], rtol=0.01, atol=1e-12):
        raise RuntimeError("Independent odd radiation power disagrees with the basis.")
    rows = []
    for k, freq in enumerate(network.f):
        po, pa = power["odd"][k], accepted["odd"][k]
        if not 0 <= po <= pa or pa <= 0:
            raise RuntimeError(f"Invalid odd radiated/accepted power at {freq / 1e9:g} GHz [L1].")
        ar = axial_ratio(fields["odd"][k])
        row = dict(
            level="L1 HFSS",
            f_Hz=freq,
            eta_odd=po / pa,
            radiated_odd_W=po,
            accepted_odd_W=pa,
            AR_odd_linear=ar,
            AR_odd_dB=20 * np.log10(ar),
            rad11=power["one"][k],
            rad22=power["two"][k],
            rad12_re=cross[k].real,
            rad12_im=cross[k].imag,
        )
        for j, pol in enumerate(("theta", "phi")):
            for i, drive in enumerate(("one", "two"), 1):
                z = fields[drive][k, j]
                row[f"{pol}{i}_re"], row[f"{pol}{i}_im"] = z.real, z.imag
        rows.append(row)
    return rows


def run_design(Hfss, d, folder, job, args):
    from sim_300ghz_rtd_cosim import Radiation, save_rows

    bias_port = job == "bias_access"
    open_r2 = job == "R2_open"
    path = folder / (job + ".aedt")
    if path.exists():
        raise FileExistsError(f"Choose a fresh output directory to preserve {path}.")
    app = Hfss(
        project=str(path),
        design="RingSlot",
        solution_type="Modal",
        version=args.aedt_version,
        non_graphical=not args.gui,
        new_desktop=True,
        close_on_exit=True,
    )
    try:
        print(f"L0 AEDT {app.aedt_version_id}; pyaedt {version('pyaedt')}", flush=True)
        geom(app, d, bias_port, open_r2, args.intervals)
        configure(app, d, args.step_ghz, args.rc_sweep)
        checked(
            app.change_validation_settings(
                entity_check_level="Strict", ignore_unclassified=False, skip_intersections=False
            ),
            "strict intersection checks",
        )
        log, ok = app.validate_full_design(output_dir=str(folder), ports=3 if bias_port else 2)
        bad = [
            s
            for s in log
            if re.search(r"\b(intersects?|intersecting|overlap(?:ping|s)?)\b", s, re.I)
            and not re.search(
                r"(no |zero |checking |check for )[^.]*"
                r"(intersect|overlap)",
                s,
                re.I,
            )
        ]
        if not ok or bad:
            raise RuntimeError("HFSS geometry/port validation failed:\n" + "\n".join(log))
        checked(app.save_project(), "save project")
        if not args.solve:
            return
        checked(app.analyze_setup("Driven", cores=args.cores, blocking=True), "HFSS solve")
        ports = ["P1", "P2"] + (["Bias"] if bias_port else [])
        all_rows = []
        for label, lo, _hi in BANDS:
            net = canonical_touchstone(app, label, folder / f"{label}.s{len(ports)}p", ports)
            if job == "loaded":
                rows = radiation_rows(app, label, net, d)
                if all_rows:
                    # Separate solves on the shared mesh have independent endpoint estimates.
                    last, first = all_rows[-1], rows[0]
                    for key in ("radiated_odd_W", "accepted_odd_W"):
                        if not np.isclose(last[key], first[key], rtol=0.01, atol=1e-12):
                            raise ValueError(f"Radiation discontinuity at {lo} GHz for {key} [L1].")
                    rows = rows[1:]
                all_rows.extend(rows)
        if all_rows:
            save_rows(folder / "radiation.csv", all_rows)
            Radiation.read(folder / "radiation.csv")
        if args.rc_sweep:
            checked(
                app.analyze_setup("GuardedRc", cores=args.cores, blocking=True),
                "guarded radius sweep",
            )
        checked(app.save_project(), "save solved project")
    finally:
        app.release_desktop(close_projects=True, close_desktop=True)


def compare_r2(folder):
    from sim_300ghz_rtd_cosim import GHZ, ODD, save_rows, stitch

    on = stitch(folder, 2)
    off = stitch(folder / "open", 2)
    f = np.unique(np.r_[on.f, off.f])
    delta = (ODD @ (on.at(f) - off.at(f)) @ ODD).real
    save_rows(
        folder / "R2_increment.csv",
        [
            dict(f_Hz=x, added_Re_Yodd_S=y, pair_equivalent_S=2 * y, level="L1 HFSS")
            for x, y in zip(f, delta, strict=True)
        ],
    )
    nominal = float((ODD @ (on.at(300 * GHZ) - off.at(300 * GHZ)) @ ODD).real)
    print(
        f"L1 HFSS ΔRe(Yodd) = q† ReH(Yon−Yopen) q = {1e3 * nominal:.6g} mS "
        f"at 300 GHz, {folder.name}",
        flush=True,
    )
    return nominal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["preflight", "api-check", "build"])
    parser.add_argument("--variant", choices=["cavity_via", "si_lens"], default=VARIANT)
    parser.add_argument("--r2-layout", choices=["cpw_gap", "flank_10deg"], default=R2_LAYOUT)
    parser.add_argument("--rc", type=float, default=71.5)
    parser.add_argument("--delta", type=float)
    parser.add_argument("--rc-sweep", nargs=3, type=float, metavar=("START", "STOP", "STEP"))
    parser.add_argument(
        "--compare-r2",
        action="store_true",
        help="Solve both R2 layouts and their geometry-matched open controls.",
    )
    parser.add_argument("--solve", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument(
        "--step-ghz", type=float, default=1.0, help="Maximum discrete sample spacing"
    )
    parser.add_argument("--aedt-version", default="2026.1")
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    if args.action == "api-check":
        AedtCalls.verify()
        return
    ds = [
        Design(args.variant, layout_name, args.rc, args.delta)
        for layout_name in (("cpw_gap", "flank_10deg") if args.compare_r2 else (args.r2_layout,))
    ]
    if args.rc_sweep and args.variant != "cavity_via":
        parser.error("R_c has no physical body in the Si lens design.")
    radii = [args.rc]
    if args.rc_sweep:
        lo, hi, step = args.rc_sweep
        if step <= 0 or hi <= lo or not np.isclose((hi - lo) / step, round((hi - lo) / step)):
            parser.error("Use an ascending radius grid including both endpoints.")
        radii.extend(np.linspace(lo, hi, round((hi - lo) / step) + 1))
    args.intervals = ceil((2 * pi * max(radii) - 30) / 20)
    for d in ds:
        for rc in radii:
            info = preflight(Design(d.variant, d.layout, rc, d.delta), args.intervals)
            if rc == d.rc:
                print(json.dumps(info, indent=2))
    if args.action == "preflight":
        if args.variant == "cavity_via":
            clear = []
            for rc in np.arange(70.0, 95.01, 0.5):
                try:
                    check_modes(rc)
                except ValueError:
                    continue
                clear.append(float(rc))
            print("L1 guard-clear radii (um) on the specified 0.5-um test grid:", clear)
        return
    if not args.out or args.step_ghz <= 0:
        parser.error("build needs --out and a positive frequency spacing.")
    Hfss = AedtCalls.verify()
    summary = []
    for d in ds:
        folder = args.out.resolve() / d.variant / d.layout
        folder.mkdir(parents=True, exist_ok=True)
        run_design(Hfss, d, folder, "loaded", args)
        run_design(Hfss, d, folder, "bias_access", args)
        if args.compare_r2:
            (folder / "open").mkdir(exist_ok=True)
            run_design(Hfss, d, folder / "open", "R2_open", args)
            if args.solve:
                summary.append(
                    dict(
                        layout=d.layout,
                        f_GHz=300.0,
                        level="L1 HFSS",
                        delta_Re_Yodd_S=compare_r2(folder),
                    )
                )
                summary[-1]["pair_equivalent_S"] = 2 * summary[-1]["delta_Re_Yodd_S"]
        if args.solve:
            meta = dict(
                schema="ringslot.barrier.v1",
                level="L1 EM",
                variant=d.variant,
                layout=d.layout,
                ports=["P1", "P2", "Bias"],
                bias_ohm_H=[1.0, 1e-9],
                bridge_in_em=True,
                partition=["R1", "R2", "CPW", "MIM"],
                rc_um=d.rc if d.variant == "cavity_via" else None,
                delta_deg=d.angle,
                empty_gap_F=epsilon_0 * (MESA * 1e-6) ** 2 / ((GAP_HI - GAP_LO) * 1e-6),
            )
            (folder / "model.json").write_text(json.dumps(meta, indent=2))
    if summary:
        from sim_300ghz_rtd_cosim import save_rows

        save_rows(args.out / ds[0].variant / "R2_comparison_at_300GHz.csv", summary)


if __name__ == "__main__":
    main()

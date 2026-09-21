#!/usr/bin/env python3
# Renders px4-updates/models/sentinel_vision_slung/model.sdf from model.sdf.template.
#
# Usage: SLUNG_MASS_KG=0.25 ./generate-slung-load-model.py
#    or: ./generate-slung-load-model.py 0.25
#
# Run this on the host, from the repo root, BEFORE update-px4-files.sh -- that script is what
# actually copies px4-updates/models/ into the live SITL build tree inside the container.
# Regenerating model.sdf here has zero effect on the running sim until that copy happens.
#
# SLUNG_MASS_KG=0 (or unset) intentionally generates nothing -- spawn-sim-env.sh falls back to
# the plain sentinel_vision model (no load) in that case.

import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
MODEL_DIR = REPO_ROOT / "px4-updates" / "models" / "sentinel_vision_slung"
TEMPLATE_PATH = MODEL_DIR / "model.sdf.template"
OUTPUT_PATH = MODEL_DIR / "model.sdf"

# Fixed across every mass sweep run -- only payload mass (and its derived inertia) changes.
ROD_LENGTH_M = 0.15
ROD_RADIUS_M = 0.004
ROD_MASS_KG = 0.02  # ~8% of PAYLOAD_MASS_KG=0.25 -- was 0.001 (a ~250:1 ratio to the payload
# and ~1300:1 to the drone), flagged early on as an untested numerical-solver-instability
# risk: a near-massless body between two ball joints in series is a known way to get
# spurious stiffness/energy injection from DART's iterative constraint solver
PAYLOAD_RADIUS_M = 0.035
PAYLOAD_HEIGHT_M = 0.20

# Real aerodynamic quadratic drag via the gz-sim Hydrodynamics system plugin (confirmed installed
# on this host: /usr/lib/x86_64-linux-gnu/libgz-sim8-hydrodynamics-system.so, parameter names
# verified against /usr/share/gz/gz-sim8/worlds/auv_controls.sdf). velocity_decay was tried first
# and confirmed to be a dead no-op in gz-sim/gz-physics as of Harmonic (gz-physics#635) -- this
# replaces it, not supplements it.
#
# AIR_DENSITY_KG_M3 is deliberately inflated well past real air (1.225 kg/m^3) -- real
# aerodynamic drag on an object this small takes many tens of oscillations to decay a
# pendulum, but the real water bottle settles in ~2 back-and-forths. That fast a decay is
# almost certainly dominated by liquid sloshing inside the bottle (internal viscous/
# turbulent dissipation), not exterior air resistance -- a mechanism this solid-rigid-body
# model has no way to represent directly. Rather than add a separate fake "sloshing" term,
# this constant is used as a single empirical damping knob and intentionally no longer
# means literal air density once it's this large -- it's sized to reproduce the observed
# decay rate, not to be physically accurate. Retune by feel; there's no principled way to
# derive the "right" value here for an unmodeled mechanism.
AIR_DENSITY_KG_M3 = 1.225*4
CD_CROSSFLOW = 1.1  # cylinder broadside to the flow (its long axis perpendicular to velocity)
CD_AXIAL = 0.9       # flat-ended cylinder moving along its own long axis

# Openly pragmatic, NOT first-principles like CD_CROSSFLOW/CD_AXIAL above -- the raw
# strip-theory rotational coefficients come out ~1e-6 (they scale with length^4), and reusing
# the tumbling-drag formula for nRabsR (spin about the cylinder's own long axis) is physically
# the wrong mechanism for that axis anyway (skin friction, not crossflow form drag) since it
# wasn't derived separately. Bump this if payload spin visibly persists; the spin marker visual
# (see model.sdf.template) is there specifically to tell real self-spin from mere orbital
# revolution around the pivot.
ROTATIONAL_DRAG_MULTIPLIER = 200.0

# Purely visual (no collision/inertial effect) -- a stripe on one side of the payload cylinder so
# self-spin is visible in the GUI instead of ambiguous with orbital motion around the pivot.
STRIPE_THICKNESS_M = 0.006
STRIPE_WIDTH_M = 0.012

# Attach point on base_link -- center of one landing gear's horizontal foot pad
# (base_link_collision_3 in sentinel_base/model.sdf: pose (0, -0.132, -0.2195), one arm of the
# inverted-T leg). The legs aren't separate links in this model, just collision boxes on
# base_link, so this is only a different offset on the same rigid body. Flip ATTACH_Y_M to
# +0.132 for the other (right) landing gear.
ATTACH_X_M = 0.0
ATTACH_Y_M = -0.132
ATTACH_Z_M = -0.2195


def solid_cylinder_inertia(mass: float, radius: float, length: float) -> tuple[float, float]:
    """(ixx == iyy, izz) for a solid cylinder with its long axis along local z, about its OWN center."""
    izz = 0.5 * mass * radius ** 2
    ixx = (1.0 / 12.0) * mass * (3.0 * radius ** 2 + length ** 2)
    return ixx, izz


def quadratic_drag_coeffs(radius: float, length: float) -> tuple[float, float, float, float, float, float]:
    """Fossen-model quadratic drag coefficients (xUabsU, yVabsV, zWabsW, kPabsP, mQabsQ, nRabsR)
    for a solid cylinder (long axis along local z) in air, from real drag-equation geometry --
    not a tuned/guessed numerical damping knob like velocity_decay was.

    Translational (surge/sway/heave): force = -0.5 * rho * Cd * A * v*|v|, A = crossflow or
    end-cap area depending on axis. Rotational (roll/pitch/yaw): strip-theory approximation,
    integrating crossflow drag over the cylinder's length, about the cylinder's OWN center --
    an order-of-magnitude estimate, not as rigorously derived as the translational terms;
    retune if payload spin looks wrong.
    """
    diameter = 2.0 * radius
    crossflow_area = diameter * length
    endcap_area = math.pi * radius ** 2

    x_uabsu = y_vabsv = -0.5 * AIR_DENSITY_KG_M3 * CD_CROSSFLOW * crossflow_area
    z_wabsw = -0.5 * AIR_DENSITY_KG_M3 * CD_AXIAL * endcap_area
    # torque = 0.5*rho*Cd*diameter*omega*|omega| * 2*integral[0, L/2] of r^3 dr
    #        = 0.5*rho*Cd*diameter*omega*|omega| * (L^4 / 32)
    rot_coeff = -0.5 * AIR_DENSITY_KG_M3 * CD_CROSSFLOW * diameter * (length ** 4) / 32.0 \
        * ROTATIONAL_DRAG_MULTIPLIER
    k_pabsp = m_qabsq = n_rabsr = rot_coeff
    return x_uabsu, y_vabsv, z_wabsw, k_pabsp, m_qabsq, n_rabsr


def main() -> None:
    mass_kg = float(sys.argv[1]) if len(sys.argv) > 1 else float(os.environ.get("SLUNG_MASS_KG", "0"))

    if mass_kg <= 0:
        print(f"SLUNG_MASS_KG={mass_kg} -- nothing to generate (0/unset means 'no load'; "
              f"spawn-sim-env.sh will use the plain sentinel_vision model instead).")
        return

    # Single rigid body (rod + payload welded together on ONE ball joint at base_link), not
    # two bodies on two ball joints in series. A real string is inextensible and has no
    # bending stiffness, but it also has no independent second degree of freedom the way two
    # frictionless ball joints in series do -- a taut string behaves as a SIMPLE pendulum
    # (all mass swinging together), not a double pendulum. The two-joint version was letting
    # the rod and payload rotate independently, which is almost certainly why the payload was
    # swinging out to ~90 degrees: that's a real double-pendulum energy-exchange mode a
    # taut string physically can't exhibit, not just "insufficient damping."
    rod_ixx_own, rod_izz_own = solid_cylinder_inertia(ROD_MASS_KG, ROD_RADIUS_M, ROD_LENGTH_M)
    payload_ixx_own, payload_izz_own = solid_cylinder_inertia(mass_kg, PAYLOAD_RADIUS_M, PAYLOAD_HEIGHT_M)

    rod_center_z = ATTACH_Z_M - ROD_LENGTH_M / 2.0
    rod_bottom_z = ATTACH_Z_M - ROD_LENGTH_M
    payload_center_z = rod_bottom_z - PAYLOAD_HEIGHT_M / 2.0

    total_mass = ROD_MASS_KG + mass_kg
    combined_cg_z = (ROD_MASS_KG * rod_center_z + mass_kg * payload_center_z) / total_mass

    # Parallel-axis theorem about the combined CG. Both parts share the same (x, y) as the
    # combined CG (everything hangs straight down the same vertical line), so the offset is
    # purely along z: that offset contributes to Ixx/Iyy (axes perpendicular to z) but NOT to
    # Izz (axis parallel to z, zero perpendicular offset in the xy-plane).
    rod_dz = rod_center_z - combined_cg_z
    payload_dz = payload_center_z - combined_cg_z
    combined_ixx = (rod_ixx_own + ROD_MASS_KG * rod_dz ** 2) + (payload_ixx_own + mass_kg * payload_dz ** 2)
    combined_izz = rod_izz_own + payload_izz_own

    # Drag coefficients: both surfaces move at the same velocity now (one rigid body), so
    # translational drag force contributions add directly (two independent surfaces, same v).
    # Rotational terms are summed the same way as a pragmatic approximation -- strictly the
    # combined body's rotational drag about the NEW (offset) pivot isn't just the sum of each
    # part's drag about its own center, but ROTATIONAL_DRAG_MULTIPLIER is already an
    # order-of-magnitude fudge factor, not a rigorously derived term, so this is consistent
    # with the existing precision level rather than a new source of error.
    rod_coeffs = quadratic_drag_coeffs(ROD_RADIUS_M, ROD_LENGTH_M)
    payload_coeffs = quadratic_drag_coeffs(PAYLOAD_RADIUS_M, PAYLOAD_HEIGHT_M)
    x_uabsu, y_vabsv, z_wabsw, k_pabsp, m_qabsq, n_rabsr = (
        r + p for r, p in zip(rod_coeffs, payload_coeffs)
    )

    stripe_offset_m = PAYLOAD_RADIUS_M + STRIPE_THICKNESS_M / 2.0
    total_hang_depth_below_base_link_m = -ATTACH_Z_M + ROD_LENGTH_M + PAYLOAD_HEIGHT_M

    rendered = TEMPLATE_PATH.read_text().format(
        ATTACH_X_M=ATTACH_X_M,
        ATTACH_Y_M=ATTACH_Y_M,
        ATTACH_Z_M=ATTACH_Z_M,
        COMBINED_MASS_KG=total_mass,
        COMBINED_CG_Z=combined_cg_z,
        COMBINED_IXX=combined_ixx,
        COMBINED_IZZ=combined_izz,
        ROD_LOCAL_Z=rod_center_z - combined_cg_z,
        ROD_RADIUS_M=ROD_RADIUS_M,
        ROD_LENGTH_M=ROD_LENGTH_M,
        PAYLOAD_LOCAL_Z=payload_center_z - combined_cg_z,
        PAYLOAD_RADIUS_M=PAYLOAD_RADIUS_M,
        PAYLOAD_HEIGHT_M=PAYLOAD_HEIGHT_M,
        STRIPE_OFFSET_M=stripe_offset_m,
        STRIPE_THICKNESS_M=STRIPE_THICKNESS_M,
        STRIPE_WIDTH_M=STRIPE_WIDTH_M,
        X_UABSU=x_uabsu,
        Y_VABSV=y_vabsv,
        Z_WABSW=z_wabsw,
        K_PABSP=k_pabsp,
        M_QABSQ=m_qabsq,
        N_RABSR=n_rabsr,
    )
    OUTPUT_PATH.write_text(rendered)
    print(f"Wrote {OUTPUT_PATH} -- payload={mass_kg}kg, rod={ROD_LENGTH_M}m (combined single-body "
          f"pendulum, mass={total_mass:.3f}kg, CG {-combined_cg_z:.3f}m below base_link), attach="
          f"(base_link leg pad at {ATTACH_X_M},{ATTACH_Y_M},{ATTACH_Z_M}), total hang depth below "
          f"base_link={total_hang_depth_below_base_link_m:.3f}m.")
    print(f"Hydrodynamics drag coeffs (combined body) -- xUabsU/yVabsV={x_uabsu:.6g}, "
          f"zWabsW={z_wabsw:.6g} (AIR_DENSITY_KG_M3={AIR_DENSITY_KG_M3}, inflated well past real "
          f"air -- see comment in source).")
    print(f"No collision on the payload link, so a low/ground-level spawn is visually odd (payload "
          f"may render below the ground plane at the bottom of its swing) but not a physics "
          f"problem -- no contact solver reaction, nothing to generate contact with.")


if __name__ == "__main__":
    main()

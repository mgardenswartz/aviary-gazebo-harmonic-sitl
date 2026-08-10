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
ROD_LENGTH_M = 0.3
ROD_RADIUS_M = 0.004
ROD_MASS_KG = 0.001  # negligible-but-nonzero stand-in for "massless" (see template comment)
PAYLOAD_RADIUS_M = 0.035
PAYLOAD_HEIGHT_M = 0.20

# Attach point on base_link -- center of one landing gear's horizontal foot pad
# (base_link_collision_3 in sentinel_base/model.sdf: pose (0, -0.132, -0.2195), one arm of the
# inverted-T leg). The legs aren't separate links in this model, just collision boxes on
# base_link, so this is only a different offset on the same rigid body. Flip ATTACH_Y_M to
# +0.132 for the other (right) landing gear.
ATTACH_X_M = 0.0
ATTACH_Y_M = -0.132
ATTACH_Z_M = -0.2195

# Real aerodynamic quadratic drag via the gz-sim Hydrodynamics system plugin (confirmed installed
# on this host: /usr/lib/x86_64-linux-gnu/libgz-sim8-hydrodynamics-system.so, parameter names
# verified against /usr/share/gz/gz-sim8/worlds/auv_controls.sdf). velocity_decay was tried first
# and confirmed to be a dead no-op in gz-sim/gz-physics as of Harmonic (gz-physics#635) -- this
# replaces it, not supplements it.
AIR_DENSITY_KG_M3 = 1.225
CD_CROSSFLOW = 1.1  # cylinder broadside to the flow (its long axis perpendicular to velocity)
CD_AXIAL = 0.9       # flat-ended cylinder moving along its own long axis


def solid_cylinder_inertia(mass: float, radius: float, length: float) -> tuple[float, float]:
    """(ixx == iyy, izz) for a solid cylinder with its long axis along local z."""
    izz = 0.5 * mass * radius ** 2
    ixx = (1.0 / 12.0) * mass * (3.0 * radius ** 2 + length ** 2)
    return ixx, izz


def quadratic_drag_coeffs(radius: float, length: float) -> tuple[float, float, float, float, float, float]:
    """Fossen-model quadratic drag coefficients (xUabsU, yVabsV, zWabsW, kPabsP, mQabsQ, nRabsR)
    for a solid cylinder (long axis along local z) in air, from real drag-equation geometry --
    not a tuned/guessed numerical damping knob like velocity_decay was.

    Translational (surge/sway/heave): force = -0.5 * rho * Cd * A * v*|v|, A = crossflow or
    end-cap area depending on axis. Rotational (roll/pitch/yaw): strip-theory approximation,
    integrating crossflow drag over the cylinder's length -- an order-of-magnitude estimate, not
    as rigorously derived as the translational terms; retune if payload spin looks wrong.
    """
    diameter = 2.0 * radius
    crossflow_area = diameter * length
    endcap_area = math.pi * radius ** 2

    x_uabsu = y_vabsv = -0.5 * AIR_DENSITY_KG_M3 * CD_CROSSFLOW * crossflow_area
    z_wabsw = -0.5 * AIR_DENSITY_KG_M3 * CD_AXIAL * endcap_area
    # torque = 0.5*rho*Cd*diameter*omega*|omega| * 2*integral[0, L/2] of r^3 dr
    #        = 0.5*rho*Cd*diameter*omega*|omega| * (L^4 / 32)
    rot_coeff = -0.5 * AIR_DENSITY_KG_M3 * CD_CROSSFLOW * diameter * (length ** 4) / 32.0
    k_pabsp = m_qabsq = n_rabsr = rot_coeff
    return x_uabsu, y_vabsv, z_wabsw, k_pabsp, m_qabsq, n_rabsr


def main() -> None:
    mass_kg = float(sys.argv[1]) if len(sys.argv) > 1 else float(os.environ.get("SLUNG_MASS_KG", "0"))

    if mass_kg <= 0:
        print(f"SLUNG_MASS_KG={mass_kg} -- nothing to generate (0/unset means 'no load'; "
              f"spawn-sim-env.sh will use the plain sentinel_vision model instead).")
        return

    rod_ixx, rod_izz = solid_cylinder_inertia(ROD_MASS_KG, ROD_RADIUS_M, ROD_LENGTH_M)
    payload_ixx, payload_izz = solid_cylinder_inertia(mass_kg, PAYLOAD_RADIUS_M, PAYLOAD_HEIGHT_M)

    rod_x_uabsu, rod_y_vabsv, rod_z_wabsw, rod_k_pabsp, rod_m_qabsq, rod_n_rabsr = \
        quadratic_drag_coeffs(ROD_RADIUS_M, ROD_LENGTH_M)
    payload_x_uabsu, payload_y_vabsv, payload_z_wabsw, payload_k_pabsp, payload_m_qabsq, payload_n_rabsr = \
        quadratic_drag_coeffs(PAYLOAD_RADIUS_M, PAYLOAD_HEIGHT_M)

    rod_midpoint_z = ATTACH_Z_M - ROD_LENGTH_M / 2.0
    rod_bottom_z = ATTACH_Z_M - ROD_LENGTH_M
    payload_center_z = rod_bottom_z - PAYLOAD_HEIGHT_M / 2.0
    total_hang_depth_below_base_link_m = -ATTACH_Z_M + ROD_LENGTH_M + PAYLOAD_HEIGHT_M

    rendered = TEMPLATE_PATH.read_text().format(
        ATTACH_X_M=ATTACH_X_M,
        ATTACH_Y_M=ATTACH_Y_M,
        ATTACH_Z_M=ATTACH_Z_M,
        ROD_MIDPOINT_Z=rod_midpoint_z,
        ROD_BOTTOM_Z=rod_bottom_z,
        ROD_MASS_KG=ROD_MASS_KG,
        ROD_IXX=rod_ixx,
        ROD_IZZ=rod_izz,
        ROD_RADIUS_M=ROD_RADIUS_M,
        ROD_LENGTH_M=ROD_LENGTH_M,
        ROD_X_UABSU=rod_x_uabsu,
        ROD_Y_VABSV=rod_y_vabsv,
        ROD_Z_WABSW=rod_z_wabsw,
        ROD_K_PABSP=rod_k_pabsp,
        ROD_M_QABSQ=rod_m_qabsq,
        ROD_N_RABSR=rod_n_rabsr,
        PAYLOAD_CENTER_Z=payload_center_z,
        PAYLOAD_MASS_KG=mass_kg,
        PAYLOAD_IXX=payload_ixx,
        PAYLOAD_IZZ=payload_izz,
        PAYLOAD_RADIUS_M=PAYLOAD_RADIUS_M,
        PAYLOAD_HEIGHT_M=PAYLOAD_HEIGHT_M,
        PAYLOAD_X_UABSU=payload_x_uabsu,
        PAYLOAD_Y_VABSV=payload_y_vabsv,
        PAYLOAD_Z_WABSW=payload_z_wabsw,
        PAYLOAD_K_PABSP=payload_k_pabsp,
        PAYLOAD_M_QABSQ=payload_m_qabsq,
        PAYLOAD_N_RABSR=payload_n_rabsr,
    )
    OUTPUT_PATH.write_text(rendered)
    print(f"Wrote {OUTPUT_PATH} -- payload={mass_kg}kg, rod={ROD_LENGTH_M}m, attach=(base_link "
          f"leg pad at {ATTACH_X_M},{ATTACH_Y_M},{ATTACH_Z_M}), total hang depth below "
          f"base_link={total_hang_depth_below_base_link_m:.3f}m.")
    print(f"Hydrodynamics drag coeffs -- rod xUabsU/yVabsV={rod_x_uabsu:.6g}, zWabsW="
          f"{rod_z_wabsw:.6g} | payload xUabsU/yVabsV={payload_x_uabsu:.6g}, zWabsW="
          f"{payload_z_wabsw:.6g}.")
    print(f"No collision on either link, so a low/ground-level spawn is visually odd (payload "
          f"may render below the ground plane at the bottom of its swing) but not a physics "
          f"problem -- no contact solver reaction, nothing to generate contact with.")


if __name__ == "__main__":
    main()

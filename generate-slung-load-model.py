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


def solid_cylinder_inertia(mass: float, radius: float, length: float) -> tuple[float, float]:
    """(ixx == iyy, izz) for a solid cylinder with its long axis along local z."""
    izz = 0.5 * mass * radius ** 2
    ixx = (1.0 / 12.0) * mass * (3.0 * radius ** 2 + length ** 2)
    return ixx, izz


def main() -> None:
    mass_kg = float(sys.argv[1]) if len(sys.argv) > 1 else float(os.environ.get("SLUNG_MASS_KG", "0"))

    if mass_kg <= 0:
        print(f"SLUNG_MASS_KG={mass_kg} -- nothing to generate (0/unset means 'no load'; "
              f"spawn-sim-env.sh will use the plain sentinel_vision model instead).")
        return

    rod_ixx, rod_izz = solid_cylinder_inertia(ROD_MASS_KG, ROD_RADIUS_M, ROD_LENGTH_M)
    payload_ixx, payload_izz = solid_cylinder_inertia(mass_kg, PAYLOAD_RADIUS_M, PAYLOAD_HEIGHT_M)

    rod_midpoint_z = -ROD_LENGTH_M / 2.0
    rod_bottom_z = -ROD_LENGTH_M
    payload_center_z = rod_bottom_z - PAYLOAD_HEIGHT_M / 2.0
    total_hang_depth_m = ROD_LENGTH_M + PAYLOAD_HEIGHT_M

    rendered = TEMPLATE_PATH.read_text().format(
        ROD_MIDPOINT_Z=rod_midpoint_z,
        ROD_BOTTOM_Z=rod_bottom_z,
        ROD_MASS_KG=ROD_MASS_KG,
        ROD_IXX=rod_ixx,
        ROD_IZZ=rod_izz,
        ROD_RADIUS_M=ROD_RADIUS_M,
        ROD_LENGTH_M=ROD_LENGTH_M,
        PAYLOAD_CENTER_Z=payload_center_z,
        PAYLOAD_MASS_KG=mass_kg,
        PAYLOAD_IXX=payload_ixx,
        PAYLOAD_IZZ=payload_izz,
        PAYLOAD_RADIUS_M=PAYLOAD_RADIUS_M,
        PAYLOAD_HEIGHT_M=PAYLOAD_HEIGHT_M,
    )
    OUTPUT_PATH.write_text(rendered)
    print(f"Wrote {OUTPUT_PATH} -- payload={mass_kg}kg, rod={ROD_LENGTH_M}m, "
          f"total hang depth below base_link={total_hang_depth_m:.3f}m.")
    print(f"Reminder: base_link sits ~0.02m above ground at a ground-level spawn -- a "
          f"{total_hang_depth_m:.3f}m hang depth WILL clip the ground at spawn unless you spawn "
          f"higher (e.g. QUAD1_LOCATION=\"0.0,0.0,-0.8\" in spawn-locations.env -- NED, so "
          f"negative z is up).")


if __name__ == "__main__":
    main()

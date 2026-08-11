import argparse
import os
import re

import optuna

# --- Argparse Setup ---
parser = argparse.ArgumentParser(description="Load an Optuna study from a specific .db file.")
parser.add_argument(
    "db_file",
    type=str,
    help="The path to the Optuna .db file (e.g., output/traj1/stage_1B.db)"
)
args = parser.parse_args()

# --- Dynamic Configuration ---
# Ensure the file exists (optional, but good practice)
if not os.path.exists(args.db_file):
    parser.error(f"The file '{args.db_file}' does not exist.")

# Construct the DB URL and extract the study name from the filename
db_url = f"sqlite:///{args.db_file}"
study_name = os.path.splitext(os.path.basename(args.db_file))[0]

# --- Verification (Optional) ---
print(f"Loading study '{study_name}' from {db_url}")
try:
    # Inspect the database to see what studies actually exist inside it
    study_summaries = optuna.get_all_study_summaries(storage=db_url)

    if not study_summaries:
        print(f"[!] The database at {db_url} contains no studies.")
        exit()

    # Warn if there's more than one, but default to the first one found
    if len(study_summaries) > 1:
        print(f"⚠️  [Warning]: Multiple studies found in this DB ({len(study_summaries)} total).")
        print(f"    Loading the first one: '{study_summaries[0].study_name}'")
    else:
        print(f"Loading study: '{study_summaries[0].study_name}'")

    # Target the actual name stored in the DB
    study_name = study_summaries[0].study_name
    study = optuna.load_study(study_name=study_name, storage=db_url)

except Exception as e:
    print(f"[!] Could not load study from {db_url}. Error: {e}")
    exit()


def detect_stage(name: str):
    """stage_{X}_study / stage_{X} -> X (e.g. '1B', '2A', '3'), else None."""
    m = re.search(r"stage_([0-9A-Za-z]+)", name)
    return m.group(1) if m else None


STAGE_CONTROLLER_TYPE = {
    '1A': 'baseline',
    '1B': 'baseline',
    '2A': 'resnet',
    '2B': 'integrated_resnet',
    '2C': 'resnet',
    '2D': 'integrated_resnet',
    '3': 'supertwisting',
    '4': 'pid',
}

# Stage 2A/2B are always seeded from stage 1B (hanging mass); 2C/2D are
# always seeded from stage 1A (no hanging mass). See
# scripts/unified_orchestrator_tpe.py's `stages` list in main().
STAGE2_BASE_STAGE = {
    '2A': '1B',
    '2B': '1B',
    '2C': '1A',
    '2D': '1A',
}


def load_base_gains(db_file: str, stage: str):
    """For stage_2A/2B/2C/2D: their k_1/k_2/k_3/k_rise are seeded from a fixed
    RISE stage (see STAGE2_BASE_STAGE above) -- those values are injected into
    param_dict directly, never suggested via trial.suggest_*, so they never
    appear in stage_2*.db's own trial.params. Have to go load the sibling
    stage_{base}.db to recover them.
    Returns (gains_dict_or_None, base_stage_or_None)."""
    base_stage = STAGE2_BASE_STAGE.get(stage)
    if base_stage is None:
        return None, None

    base_db = os.path.join(os.path.dirname(os.path.abspath(db_file)), f"stage_{base_stage}.db")
    if not os.path.exists(base_db):
        return None, base_stage
    try:
        base_study = optuna.load_study(
            study_name=f"stage_{base_stage}_study", storage=f"sqlite:///{base_db}"
        )
        return base_study.best_params, base_stage
    except Exception:
        return None, base_stage


stage = detect_stage(study_name)
controller_type = STAGE_CONTROLLER_TYPE.get(stage)

# 1. Get the absolute best trial
print("\n" + "="*40)
print("🏆 ABSOLUTE BEST TRIAL 🏆")
print("="*40)
best_trial = study.best_trial
print(f"Trial Number : {best_trial.number}")
print(f"Cost         : {best_trial.value:.4f}")
print("Parameters   :")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")
print("User Attrs   :")
for key, value in best_trial.user_attrs.items():
    print(f"    {key}: {value}")

# 2. Sort all completed, valid trials by value
completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
completed_trials.sort(key=lambda t: t.value)

# e_RMS/u_RMS user attrs are only ever set by unified_orchestrator_tpe.py's
# evaluate_single() -- check the first completed trial.
has_rms = bool(completed_trials) and (
    "e_RMS" in completed_trials[0].user_attrs or "u_RMS" in completed_trials[0].user_attrs
)

print("\n" + "="*95)

if controller_type == 'baseline':
    # RISE baseline: rows carry the raw search-space params directly.
    header_str = (
        f"{'Rank':<6} | {'Trial':<7} | {'Cost':<9} | {'k_1':<8} | {'k_2':<8} | {'k_3':<8} | {'k_rise':<9}"
    )
    if has_rms:
        header_str += f" | {'e_RMS':<9} | {'u_RMS':<9}"
    print(header_str)
    print("-" * len(header_str))

    for i, trial in enumerate(completed_trials[:15]):
        k_1 = trial.params.get("k_1", 0)
        k_2 = trial.params.get("k_2", 0)
        k_3 = trial.params.get("k_3", 0)
        k_rise = trial.params.get("k_rise", 0)
        row_str = (
            f"{i+1:<6} | {trial.number:<7} | {trial.value:<9.3g} | {k_1:<8.3g} | {k_2:<8.3g} | "
            f"{k_3:<8.3g} | {k_rise:<9.3g}"
        )
        if has_rms:
            row_str += f" | {trial.user_attrs.get('e_RMS', 0):<9.3g} | {trial.user_attrs.get('u_RMS', 0):<9.3g}"
        print(row_str)

elif controller_type in ('resnet', 'integrated_resnet'):
    # ResNet / Integrated_ResNet: k_1/k_2/k_3/k_rise are constant for every
    # trial in this stage (seeded once from the base RISE stage, never
    # varied here) -- printed once above the table instead of per-row.
    base_gains, base_stage = load_base_gains(args.db_file, stage)
    print(f"Base RISE gains (seeded from stage_{base_stage}, fixed for every trial in this stage):")
    if base_gains:
        k_1 = base_gains.get("k_1", 0)
        k_2 = base_gains.get("k_2", 0)
        k_3 = base_gains.get("k_3", 0)
        k_rise = base_gains.get("k_rise", 0)
        print(f"    k_1={k_1:.4g}  k_2={k_2:.4g}  k_3={k_3:.4g}  k_rise={k_rise:.4g}")
    else:
        print(f"    [!] Could not load base gains (checked '{args.config}' and the sibling stage_{base_stage}.db)")
    print()

    header_str = f"{'Rank':<6} | {'Trial':<7} | {'Cost':<9} | {'gamma':<9} | {'sigma_mod':<10}"
    if has_rms:
        header_str += f" | {'e_RMS':<9} | {'u_RMS':<9}"
    print(header_str)
    print("-" * len(header_str))

    for i, trial in enumerate(completed_trials[:15]):
        gamma = trial.params.get("gamma", 0)
        sigma_mod = trial.params.get("sigma_mod", 0)
        row_str = f"{i+1:<6} | {trial.number:<7} | {trial.value:<9.3g} | {gamma:<9.3g} | {sigma_mod:<10.3g}"
        if has_rms:
            row_str += f" | {trial.user_attrs.get('e_RMS', 0):<9.3g} | {trial.user_attrs.get('u_RMS', 0):<9.3g}"
        print(row_str)

elif controller_type == 'pid':
    # PID: an independent baseline, its K_P/K_I/K_D are directly the
    # search-space params.
    header_str = f"{'Rank':<6} | {'Trial':<7} | {'Cost':<9} | {'K_P':<9} | {'K_I':<9} | {'K_D':<9}"
    if has_rms:
        header_str += f" | {'e_RMS':<9} | {'u_RMS':<9}"
    print(header_str)
    print("-" * len(header_str))

    for i, trial in enumerate(completed_trials[:15]):
        K_P = trial.params.get("K_P", 0)
        K_I = trial.params.get("K_I", 0)
        K_D = trial.params.get("K_D", 0)
        row_str = f"{i+1:<6} | {trial.number:<7} | {trial.value:<9.3g} | {K_P:<9.3g} | {K_I:<9.3g} | {K_D:<9.3g}"
        if has_rms:
            row_str += f" | {trial.user_attrs.get('e_RMS', 0):<9.3g} | {trial.user_attrs.get('u_RMS', 0):<9.3g}"
        print(row_str)

elif controller_type == 'supertwisting':
    # Supertwisting: search-space params are k_st_1/2/3.
    header_str = f"{'Rank':<6} | {'Trial':<7} | {'Cost':<9} | {'k_st_1':<9} | {'k_st_2':<9} | {'k_st_3':<9}"
    if has_rms:
        header_str += f" | {'e_RMS':<9} | {'u_RMS':<9}"
    print(header_str)
    print("-" * len(header_str))

    for i, trial in enumerate(completed_trials[:15]):
        k_1 = trial.params.get("k_st_1", 0)
        k_2 = trial.params.get("k_st_2", 0)
        k_3 = trial.params.get("k_st_3", 0)
        row_str = f"{i+1:<6} | {trial.number:<7} | {trial.value:<9.3g} | {k_1:<9.3g} | {k_2:<9.3g} | {k_3:<9.3g}"
        if has_rms:
            row_str += f" | {trial.user_attrs.get('e_RMS', 0):<9.3g} | {trial.user_attrs.get('u_RMS', 0):<9.3g}"
        print(row_str)

else:
    # Unrecognized study/file naming (not stage_{X}_study) -- fall back to
    # a generic param-sniffing layout so this script still works on arbitrary
    # Optuna dbs instead of hard failing.
    print(f"[!] Could not determine stage from study name '{study_name}' -- using generic layout.")
    use_nn_layout = bool(completed_trials) and (
        "num_blocks" in completed_trials[0].params or "hidden_width" in completed_trials[0].params
    )

    if use_nn_layout:
        header_str = f"{'Rank':<6} | {'Trial':<7} | {'Cost':<9} | {'num_blocks':<10} | {'hidden_width':<12} | {'k_0':<6} | {'k_i':<6} | {'gamma':<7} | {'sigma_mod':<9} | {'W_s':<6}"
    else:
        header_str = f"{'Rank':<6} | {'Trial':<7} | {'Cost':<9} | {'k_1':<7} | {'k_2':<7} | {'k_3':<7} | {'k_rise':<9}"
    if has_rms:
        header_str += f" | {'e_RMS':<9} | {'u_RMS':<9}"
    print(header_str)
    print("-" * len(header_str))

    for i, trial in enumerate(completed_trials[:15]):
        if use_nn_layout:
            nb = trial.params.get("num_blocks", 0)
            hw = trial.params.get("hidden_width", 0)
            k0 = trial.params.get("k_0", 0)
            ki = trial.params.get("k_i", 0)
            g = trial.params.get("gamma", 0)
            sm = trial.params.get("sigma_mod", 0)
            ws = trial.params.get("initial_weight_scale_factor", 0)
            row_str = f"{i+1:<6} | {trial.number:<7} | {trial.value:<9.3g} | {nb:<10} | " \
                      f"{hw:<12} | {k0:<6.3g} | {ki:<6.3g} | {g:<7.3g} | {sm:<9.3g} | {ws:<6.3g}"
        else:
            k_1 = trial.params.get("k_1", 0)
            k_2 = trial.params.get("k_2", 0)
            k_3 = trial.params.get("k_3", 0)
            krise = trial.params.get("k_rise", 0)
            row_str = f"{i+1:<6} | {trial.number:<7} | {trial.value:<9.3g} | {k_1:<7.3g} | " \
                      f"{k_2:<7.3g} | {k_3:<7.3g} | {krise:<9.3g}"
        if has_rms:
            row_str += f" | {trial.user_attrs.get('e_RMS', 0):<9.3g} | {trial.user_attrs.get('u_RMS', 0):<9.3g}"
        print(row_str)

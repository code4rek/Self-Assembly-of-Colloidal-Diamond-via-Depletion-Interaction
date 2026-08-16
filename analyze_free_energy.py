import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.integrate import quad


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
PARENT_FOLDER = Path("/home/akash/hpc_hoomd-files/free_energy")

# True allows a preliminary estimate from energies.csv before a job finishes.
# Completed summary.json results are always preferred when available.
ALLOW_INCOMPLETE_RUNS = True

KT = 1.0
EXPECTED_SEEDS_PER_S = 5
EXPECTED_S_VALUES = 10


STATE_PATTERN = re.compile(
    r"^(diamond|mtab|rhombo)_"
    r"sz_([0-9.eE+-]+)_"
    r"cr_([0-9.eE+-]+)_"
    r"rl_([0-9.eE+-]+)_"
    r"phi_([0-9.eE+-]+)_"
    r"sigmap_([0-9.eE+-]+)_"
    r"lambda_([0-9.eE+-]+)_"
    r"N_([0-9]+)$"
)


def parse_output_path(energy_path):
    """Read state parameters from the simulation directory hierarchy."""
    attempt_dir = energy_path.parent
    seed_dir = attempt_dir.parent
    s_dir = seed_dir.parent
    state_dir = s_dir.parent

    match = STATE_PATTERN.match(state_dir.name)
    if match is None:
        return None

    if not seed_dir.name.startswith("seed_") or not s_dir.name.startswith("s_"):
        return None

    crystal, sz, cr, rl, phi, sigmap, lambda_max, n_tlpps = match.groups()
    return {
        "crystal": crystal,
        "sz": float(sz),
        "cr": float(cr),
        "rl": float(rl),
        "phi": float(phi),
        "sigmap": float(sigmap),
        "lambda_max": float(lambda_max),
        "n_tlpps": int(n_tlpps),
        "s": float(s_dir.name.removeprefix("s_")),
        "seed": int(seed_dir.name.removeprefix("seed_")),
        "attempt_dir": attempt_dir,
    }


def mean_from_energy_csv(energy_path):
    """Return the mean g/N and number of recorded production samples."""
    values = []
    with energy_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            value = row.get("g_per_tlpp", "")
            if value:
                values.append(float(value))

    if not values:
        return None
    return float(np.mean(values)), len(values)


def load_attempt(energy_path):
    """Load one attempt, preferring its completed summary.json."""
    parsed = parse_output_path(energy_path)
    if parsed is None:
        return None

    summary_path = energy_path.parent / "summary.json"
    if summary_path.exists():
        try:
            with summary_path.open() as stream:
                summary = json.load(stream)
        except (OSError, json.JSONDecodeError):
            summary = None

        if summary is not None and summary.get("status") == "complete":
            parsed.update(
                {
                    "crystal": summary["crystal"],
                    "sz": float(summary["sz"]),
                    "cr": float(summary["cr"]),
                    "rl": float(summary["rl"]),
                    "lambda_max": float(summary["lambda_max"]),
                    "n_tlpps": int(summary["n_tlpps"]),
                    "s": float(summary["s"]),
                    "seed": int(summary["seed"]),
                    "g_per_tlpp": float(summary["g_mean_per_tlpp"]),
                    "samples": int(summary["n_energy_samples"]),
                    "complete": True,
                    "modified_time": summary_path.stat().st_mtime,
                }
            )
            return parsed

    if not ALLOW_INCOMPLETE_RUNS:
        return None

    provisional = mean_from_energy_csv(energy_path)
    if provisional is None:
        return None

    parsed.update(
        {
            "g_per_tlpp": provisional[0],
            "samples": provisional[1],
            "complete": False,
            "modified_time": energy_path.stat().st_mtime,
        }
    )
    return parsed


def discover_runs():
    """Return one preferred attempt for every state, s value, and seed."""
    preferred = {}

    for energy_path in PARENT_FOLDER.rglob("energies.csv"):
        run = load_attempt(energy_path)
        if run is None:
            continue

        key = (
            run["crystal"],
            run["sz"],
            run["cr"],
            run["rl"],
            run["lambda_max"],
            run["n_tlpps"],
            round(run["s"], 12),
            run["seed"],
        )

        old = preferred.get(key)
        if old is None:
            preferred[key] = run
        elif run["complete"] and not old["complete"]:
            preferred[key] = run
        elif run["complete"] == old["complete"]:
            if run["modified_time"] > old["modified_time"]:
                preferred[key] = run

    return list(preferred.values())


def einstein_free_energy_per_tlpp(n_tlpps, rl, cr, lambda_max, kT=1.0):
    """Configurational Einstein free energy with one fixed TLPP."""
    beta = 1.0 / kT
    b = 2.0 * rl * cr

    translation = 1.5 * np.log(5.0 * beta * lambda_max / (2.0 * np.pi))

    def rotational_integrand(theta):
        half_sine_squared = np.sin(theta / 2.0) ** 2
        return (
            2.0
            / np.pi
            * half_sine_squared
            * np.exp(-2.0 * beta * lambda_max * b**2 * half_sine_squared)
        )

    z_rotation, _ = quad(
        rotational_integrand,
        0.0,
        np.pi,
        epsabs=1e-14,
        epsrel=1e-12,
        limit=300,
    )
    rotation = -np.log(z_rotation)

    return (n_tlpps - 1.0) / n_tlpps * (translation + rotation)


def average_seeds(runs):
    """Average independent seed means at every s value."""
    groups = defaultdict(list)
    for run in runs:
        state_key = (
            run["crystal"],
            run["sz"],
            run["cr"],
            run["rl"],
            run["lambda_max"],
            run["n_tlpps"],
        )
        groups[(state_key, round(run["s"], 12))].append(run)

    averaged = []
    for (state_key, _), seed_runs in groups.items():
        values = np.asarray([run["g_per_tlpp"] for run in seed_runs])
        seed_sem = (
            float(np.std(values, ddof=1) / np.sqrt(len(values)))
            if len(values) > 1
            else float("nan")
        )

        crystal, sz, cr, rl, lambda_max, n_tlpps = state_key
        averaged.append(
            {
                "crystal": crystal,
                "sz": sz,
                "cr": cr,
                "rl": rl,
                "lambda_max": lambda_max,
                "n_tlpps": n_tlpps,
                "s": float(np.mean([run["s"] for run in seed_runs])),
                "g_per_tlpp_mean": float(np.mean(values)),
                "g_per_tlpp_seed_sem": seed_sem,
                "number_of_seeds": len(seed_runs),
                "minimum_samples_per_seed": min(run["samples"] for run in seed_runs),
                "all_runs_complete": all(run["complete"] for run in seed_runs),
            }
        )

    return averaged


def integrate_state(points):
    """Integrate ten mapped Gauss-Legendre points for one crystal state."""
    points = sorted(points, key=lambda point: point["s"])
    if len(points) != EXPECTED_S_VALUES:
        return None, (
            f"found {len(points)} s values; expected {EXPECTED_S_VALUES}"
        )

    gauss_nodes, gauss_weights = np.polynomial.legendre.leggauss(
        EXPECTED_S_VALUES
    )
    expected_s = 0.5 * (gauss_nodes + 1.0)
    mapped_weights = 0.5 * gauss_weights
    measured_s = np.asarray([point["s"] for point in points])

    largest_node_difference = float(np.max(np.abs(measured_s - expected_s)))
    if largest_node_difference > 1.0e-3:
        return None, (
            "the s values do not match the ten Gauss-Legendre nodes; "
            f"largest difference={largest_node_difference:.6g}"
        )

    g_values = np.asarray([point["g_per_tlpp_mean"] for point in points])
    delta_f_per_tlpp = float(np.sum(mapped_weights * g_values))

    sem_values = np.asarray(
        [point["g_per_tlpp_seed_sem"] for point in points]
    )
    if np.all(np.isfinite(sem_values)):
        integration_sem = float(
            np.sqrt(np.sum((mapped_weights * sem_values) ** 2))
        )
    else:
        integration_sem = float("nan")

    first = points[0]
    f_einstein_per_tlpp = einstein_free_energy_per_tlpp(
        first["n_tlpps"],
        first["rl"],
        first["cr"],
        first["lambda_max"],
        KT,
    )

    result = {
        "crystal": first["crystal"],
        "sz": first["sz"],
        "cr": first["cr"],
        "rl": first["rl"],
        "lambda_max": first["lambda_max"],
        "n_tlpps": first["n_tlpps"],
        "f_einstein_per_tlpp": f_einstein_per_tlpp,
        "delta_f_per_tlpp": delta_f_per_tlpp,
        "delta_f_sem": integration_sem,
        "f_crystal_per_tlpp": f_einstein_per_tlpp + delta_f_per_tlpp,
        "all_runs_complete": all(
            point["all_runs_complete"] for point in points
        ),
        "all_s_have_expected_seeds": all(
            point["number_of_seeds"] == EXPECTED_SEEDS_PER_S
            for point in points
        ),
    }
    return result, None


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not PARENT_FOLDER.exists():
        raise FileNotFoundError(f"Folder not found: {PARENT_FOLDER}")

    runs = discover_runs()
    if not runs:
        raise RuntimeError(
            f"No usable energies.csv files found below {PARENT_FOLDER}"
        )

    averaged = average_seeds(runs)
    averaged.sort(key=lambda row: (row["crystal"], row["s"]))

    point_output = PARENT_FOLDER / "free_energy_g_vs_s.csv"
    point_fields = [
        "crystal",
        "sz",
        "cr",
        "rl",
        "lambda_max",
        "n_tlpps",
        "s",
        "g_per_tlpp_mean",
        "g_per_tlpp_seed_sem",
        "number_of_seeds",
        "minimum_samples_per_seed",
        "all_runs_complete",
    ]
    write_csv(point_output, averaged, point_fields)

    state_groups = defaultdict(list)
    for point in averaged:
        key = (
            point["crystal"],
            point["sz"],
            point["cr"],
            point["rl"],
            point["lambda_max"],
            point["n_tlpps"],
        )
        state_groups[key].append(point)

    results = []
    for state_key, points in sorted(state_groups.items()):
        result, error = integrate_state(points)
        if error is not None:
            print(f"NOT INTEGRATED {state_key}: {error}")
            continue
        results.append(result)

    result_output = PARENT_FOLDER / "free_energy_results.csv"
    result_fields = [
        "crystal",
        "sz",
        "cr",
        "rl",
        "lambda_max",
        "n_tlpps",
        "f_einstein_per_tlpp",
        "delta_f_per_tlpp",
        "delta_f_sem",
        "f_crystal_per_tlpp",
        "all_runs_complete",
        "all_s_have_expected_seeds",
    ]
    write_csv(result_output, results, result_fields)

    print(f"\nRead {len(runs)} independent simulation windows.")
    print(f"Saved s-dependent averages to: {point_output}")
    print(f"Saved integrated free energies to: {result_output}\n")

    if results:
        print("Free energy per TLPP (kT units because kT=1):")
        for result in sorted(results, key=lambda row: row["f_crystal_per_tlpp"]):
            label = (
                "FINAL"
                if result["all_runs_complete"]
                and result["all_s_have_expected_seeds"]
                else "PROVISIONAL"
            )
            print(
                f"  {result['crystal']:8s} "
                f"F_E/N={result['f_einstein_per_tlpp']:.8f}  "
                f"DeltaF/N={result['delta_f_per_tlpp']:.8f}  "
                f"F/N={result['f_crystal_per_tlpp']:.8f}  "
                f"SEM={result['delta_f_sem']:.3g}  [{label}]"
            )


if __name__ == "__main__":
    main()

"""Forward-only necessary amplitude-compatibility bounds; no mask synthesis."""

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.data import ImageBank
from src.models.transport import UnitaryFresnelTransport
from src.physics.fresnel import FresnelPropagator
from src.protocol import PEAK_PROTOCOL, prepare_target


FREQUENCIES = ((1, 0), (0, 1), (1, 1), (1, -1), (2, 0), (0, 2))
PAIRS = tuple((first, second) for first in range(4) for second in range(first + 1, 4))


@dataclass
class Config:
    size: int = 128
    train_quartets: int = 128
    selection_per_group: int = 64
    data_seed: int = 42
    train_seed: int = 50042
    random_seed: int = 20261108
    random_fields: int = 32
    batch_size: int = 16
    violation_tolerance: float = 1e-7
    output: Path = Path("outputs/physics_audit_night")


def commutator_bounds(input_phase, frequencies=FREQUENCIES):
    rows = []
    for first, second in PAIRS:
        difference = input_phase[second] - input_phase[first]
        values = []
        for kx, ky in frequencies:
            shifted = torch.roll(difference, shifts=(-ky, -kx), dims=(-2, -1))
            values.append(torch.expm1(1j * (shifted - difference)).abs().amax())
        rows.append(torch.stack(values))
    return torch.stack(rows)


def intensity_moments(amplitude, frequencies=FREQUENCIES):
    intensity = amplitude.double().square()
    energy = intensity.sum((-2, -1), keepdim=True)
    valid = energy[..., 0, 0] > 0
    probability = intensity / energy.clamp_min(torch.finfo(intensity.dtype).tiny)
    spectrum = torch.fft.fft2(probability)
    moments = torch.stack([spectrum[..., ky % amplitude.shape[-2], kx % amplitude.shape[-1]] for kx, ky in frequencies], dim=-1)
    return moments, valid


def dense_operator_check():
    size = 8
    physics = FresnelPropagator(size=size)
    bounds = commutator_bounds(physics.input_phase)
    basis = torch.eye(size * size, dtype=torch.complex128).reshape(size * size, size, size)
    transformed = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(basis, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))
    fourier = transformed.reshape(size * size, size * size).T
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    errors = []
    for pair_index, (first, second) in enumerate(PAIRS):
        difference = physics.input_phase[second] - physics.input_phase[first]
        transport = fourier @ torch.diag(torch.exp(1j * difference.flatten())) @ fourier.conj().T
        for frequency_index, (kx, ky) in enumerate(FREQUENCIES):
            measurement = torch.diag(torch.exp(-2j * math.pi * (kx * xx + ky * yy).double().flatten() / size))
            commutator = transport.conj().T @ measurement @ transport - measurement
            norm = torch.linalg.matrix_norm(commutator, ord=2)
            errors.append(float((norm - bounds[pair_index, frequency_index]).abs()))
    return {"size": size, "matrices": len(errors), "max_formula_vs_dense_svd_error": max(errors)}


def random_field_check(cfg, bounds):
    rng = torch.Generator().manual_seed(cfg.random_seed)
    source = torch.complex(torch.randn(cfg.random_fields, 1, cfg.size, cfg.size, generator=rng, dtype=torch.float64),
                           torch.randn(cfg.random_fields, 1, cfg.size, cfg.size, generator=rng, dtype=torch.float64))
    operator = UnitaryFresnelTransport(size=cfg.size).double()
    fields = operator(source.expand(-1, 4, -1, -1))
    moments, valid = intensity_moments(fields.abs())
    differences = torch.stack([(moments[:, second] - moments[:, first]).abs() for first, second in PAIRS], dim=1)
    excess = differences - bounds[None]
    return {"fields": cfg.random_fields, "seed": cfg.random_seed, "dtype": "complex128",
            "all_nonzero": bool(valid.all()), "max_observed_moment_difference_minus_bound": float(excess.max()),
            "violations_at_1e_minus_10": int((excess > 1e-10).sum())}


def summarize_group(bank, records, bounds, cfg):
    all_moments, all_valid = [], []
    for start in range(0, len(records), cfg.batch_size):
        target = prepare_target(bank.fetch(records[start:start + cfg.batch_size]), PEAK_PROTOCOL)
        moments, valid = intensity_moments(target)
        all_moments.append(moments)
        all_valid.append(valid)
    moments = torch.cat(all_moments)
    valid = torch.cat(all_valid)
    differences = torch.stack([(moments[:, second] - moments[:, first]).abs() for first, second in PAIRS], dim=1)
    valid_pairs = torch.stack([valid[:, first] & valid[:, second] for first, second in PAIRS], dim=1)
    excess = differences - bounds[None]
    positive = torch.where(valid_pairs[..., None], excess.clamp_min(0), torch.zeros_like(excess))
    violations = valid_pairs[..., None] & (excess > cfg.violation_tolerance)
    results = {"quartets": len(records), "zero_target_images": int((~valid).sum()),
               "quartets_with_any_violation": int(violations.any(dim=(1, 2)).sum()),
               "fraction_with_any_violation": float(violations.any(dim=(1, 2)).double().mean()),
               "mean_max_pair_intensity_l1_error_sum_lower_bound": float(positive.amax((1, 2)).mean()),
               "pairs": []}
    for pair_index, (first, second) in enumerate(PAIRS):
        count = int(valid_pairs[:, pair_index].sum())
        row = {"from_plane": first + 1, "to_plane": second + 1, "valid_quartets": count, "frequencies": []}
        for frequency_index, frequency in enumerate(FREQUENCIES):
            flags = violations[:, pair_index, frequency_index]
            measured = positive[:, pair_index, frequency_index]
            row["frequencies"].append({"frequency_kx_ky": list(frequency), "commutator_bound": float(bounds[pair_index, frequency_index]),
                                       "violation_count": int(flags.sum()), "violation_fraction": float(flags.sum() / max(count, 1)),
                                       "mean_intensity_l1_error_sum_lower_bound": float(measured.sum() / max(count, 1)),
                                       "max_violation": float(measured.max())})
        results["pairs"].append(row)
    flat_index = int(positive.reshape(-1).argmax())
    example, pair, frequency = np.unravel_index(flat_index, positive.shape)
    first, second = PAIRS[pair]
    encode = lambda value: [float(value.real), float(value.imag)]
    results["largest_violation_example"] = {
        "quartet_index": int(example), "source_identities_dataset_index": records[example].tolist(),
        "planes_one_based": [first + 1, second + 1], "frequency_kx_ky": list(FREQUENCIES[frequency]),
        "first_moment_real_imag": encode(moments[example, first, frequency]),
        "second_moment_real_imag": encode(moments[example, second, frequency]),
        "moment_difference": float(differences[example, pair, frequency]),
        "bound": float(bounds[pair, frequency]), "violation": float(positive[example, pair, frequency]),
    }
    return results


def main():
    cfg = Config()
    torch.set_num_threads(4)
    started = time.perf_counter()
    physics = FresnelPropagator(size=cfg.size)
    bounds = commutator_bounds(physics.input_phase)
    checks = {"dense_commutator": dense_operator_check(), "random_complex_fields": random_field_check(cfg, bounds)}
    if checks["dense_commutator"]["max_formula_vs_dense_svd_error"] > 1e-11 or checks["random_complex_fields"]["violations_at_1e_minus_10"]:
        raise RuntimeError("Necessary-bound validation failed")
    bank = ImageBank(seed=cfg.data_seed)
    train_records = bank.sample_train(cfg.train_quartets, np.random.default_rng(cfg.train_seed), "mixed")
    selection = bank.validation_manifest(cfg.selection_per_group)
    for records, pools in [(train_records, bank.train)] + [(records, bank.val) for records in selection.values()]:
        for dataset in range(3):
            selected = records[:, :, 1][records[:, :, 0] == dataset]
            if not np.isin(selected, pools[dataset]).all():
                raise RuntimeError("Manifest split membership is invalid")
    manifest = {"TRAIN": train_records.tolist(), "selection": {key: records.tolist() for key, records in selection.items()}}
    manifest_text = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    sources = [Path(__file__), Path("src/physics/fresnel.py"), Path("src/models/transport.py"), Path("src/data.py"), Path("src/protocol.py")]
    result = {
        "diagnostic_only": True, "purpose": "Necessary normalized-intensity compatibility bound; no synthesis or mask optimization.",
        "bound": "abs(mu_j(k)-mu_i(k)) <= max_n abs(exp(i*(delta_phase[n+k]-delta_phase[n]))-1)",
        "l1_bound": "For any realizable intensity probabilities p, ||q_i-p_i||_1+||q_j-p_j||_1 >= positive target-moment violation.",
        "not_a_bound_on": ["SSIM", "Pearson", "amplitude MSE"],
        "size": cfg.size, "wavelength_m": physics.wavelength, "source_pitch_m": physics.pitch,
        "distances_m": physics.distances.tolist(), "frequencies_kx_ky": [list(value) for value in FREQUENCIES],
        "checks": checks, "provenance": {
            "train_quartets": cfg.train_quartets, "train_sampler": "half homogeneous, half independent; original mixed sampler",
            "selection_per_group": cfg.selection_per_group, "data_seed": cfg.data_seed, "train_seed": cfg.train_seed,
            "train_homogeneous_quartets": int(sum(len(set(row[:, 0])) == 1 for row in train_records)),
            "protocol": PEAK_PROTOCOL, "moment_arithmetic": "float64; targets fetched and peak-normalized as normal float32 evaluator inputs",
            "moment_origin": "array index zero, unnormalized FFT of q=A^2/sum(A^2); pair-difference magnitude is independent of common origin",
            "violation_tolerance": cfg.violation_tolerance, "test_used": False, "reserve_used": False,
            "manifest_sha256": hashlib.sha256(manifest_text.encode()).hexdigest(),
            "source_sha256": {str(path.relative_to(Path.cwd()) if path.is_absolute() else path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        },
        "groups": {"TRAIN": summarize_group(bank, train_records, bounds, cfg)},
    }
    for name, records in selection.items():
        result["groups"]["selection_" + name] = summarize_group(bank, records, bounds, cfg)
    result["seconds"] = time.perf_counter() - started
    cfg.output.mkdir(parents=True, exist_ok=True)
    (cfg.output / "moment_compatibility_manifest.json").write_text(manifest_text, encoding="utf-8")
    (cfg.output / "moment_compatibility_diagnostic.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"checks": checks, "seconds": result["seconds"], "groups": {name: {key: values[key] for key in ("quartets", "quartets_with_any_violation", "fraction_with_any_violation", "mean_max_pair_intensity_l1_error_sum_lower_bound", "largest_violation_example")} for name, values in result["groups"].items()}}, indent=2))


if __name__ == "__main__":
    main()

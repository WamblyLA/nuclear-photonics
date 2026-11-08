"""Forward-only analysis of exact discrete inter-plane unitary operators.

This script does not optimize or synthesize any mask. Random phase-only fields
are data-independent probes, not hologram labels or benchmark solutions.
Run from the repository root: python -m scripts.diagnose_plane_coupling
"""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from src.models.transport import UnitaryFresnelTransport
from src.physics.fresnel import FresnelPropagator


@dataclass
class Config:
    size: int = 128
    samples: int = 32
    seed: int = 20261108
    output: Path = Path("outputs/physics_audit_night/plane_coupling_diagnostic.json")


def correlation(first, second):
    first = first.flatten(1)
    second = second.flatten(1)
    first = first - first.mean(1, keepdim=True)
    second = second - second.mean(1, keepdim=True)
    return (first * second).sum(1) / (first.norm(dim=1) * second.norm(dim=1)).clamp_min(1e-12)


def summarize(values):
    return {"mean": float(values.mean()), "std_across_random_fields": float(values.std(unbiased=False))}


def impulse_summary(energy, radius, pixel_pitch):
    size = energy.shape[-1]
    middle = size // 2
    order = radius.flatten().argsort()
    cumulative = energy.flatten()[order].cumsum(0)
    result = {"total_energy": float(energy.sum()), "central_energy": float(energy[middle, middle])}
    for fraction in (0.5, 0.9, 0.95, 0.99):
        index = int(torch.searchsorted(cumulative, fraction))
        distance = float(radius.flatten()[order[min(index, size * size - 1)]])
        result[f"radius_{round(100 * fraction)}_pixels"] = distance
        result[f"radius_{round(100 * fraction)}_micrometers"] = distance * pixel_pitch * 1e6
    for window in (3, 7, 15, 31):
        lower = middle - window // 2
        result[f"energy_in_{window}x{window}"] = float(energy[lower:lower + window, lower:lower + window].sum())
    result["rms_radius_pixels"] = float((energy * radius.square()).sum().sqrt())
    return result


def main():
    cfg = Config()
    started = time.perf_counter()
    torch.set_num_threads(4)
    operator = UnitaryFresnelTransport(size=cfg.size).double()
    physics = FresnelPropagator(size=cfg.size).double()
    coordinates = torch.arange(cfg.size, dtype=torch.float64) - cfg.size // 2
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square())
    impulse = torch.zeros(1, 4, cfg.size, cfg.size, dtype=torch.complex128)
    impulse[..., cfg.size // 2, cfg.size // 2] = 1
    source_impulses = operator.adjoint(impulse)
    shifted_impulse = torch.roll(impulse, shifts=(-32, 29), dims=(-2, -1))
    shifted_sources = operator.adjoint(shifted_impulse)
    rng = torch.Generator().manual_seed(cfg.seed)
    phases = 2 * math.pi * torch.rand(cfg.samples, 1, cfg.size, cfg.size, generator=rng, dtype=torch.float64) - math.pi
    fields = operator(torch.exp(1j * phases).expand(-1, 4, -1, -1))
    result = {
        "purpose": "Data-independent forward operator diagnostic; no mask synthesis, training or input fitting.",
        "size": cfg.size, "samples": cfg.samples, "seed": cfg.seed, "device": "cpu",
        "distances_mm": (physics.distances * 1e3).tolist(),
        "output_pitch_um": (physics.output_pitch * 1e6).tolist(),
        "operator": "Q_ji = P_j P_i*, where P uses both Fresnel chirps and orthonormal centered FFT; positive prefactors omitted.",
        "random_probe": "Uniform independent phase in [-pi,pi] with unit amplitude in the DOE plane, same field propagated to every plane.",
        "coarse_statistic": "Pearson of valid stride1 local mean intensity, no peak normalization; coordinate indices correspond to different physical pitches at different planes.",
        "pairs": [],
    }
    with torch.no_grad():
        for source in range(4):
            responses = operator(source_impulses[:, source:source + 1].expand(-1, 4, -1, -1))
            shifted = operator(shifted_sources[:, source:source + 1].expand(-1, 4, -1, -1))
            for destination in range(source + 1, 4):
                energy = responses[0, destination].abs().square()
                record = {"from_plane": source + 1, "to_plane": destination + 1}
                record.update(impulse_summary(energy, radius, float(physics.output_pitch[destination])))
                translated = torch.roll(energy, shifts=(-32, 29), dims=(-2, -1))
                record["impulse_energy_translation_max_error"] = float((translated - shifted[0, destination].abs().square()).abs().max())
                record["random_amplitude_pearson"] = summarize(correlation(fields[:, source].abs(), fields[:, destination].abs()))
                first = fields[:, source:source + 1].abs().square()
                second = fields[:, destination:destination + 1].abs().square()
                for window in (1, 7, 15, 31):
                    record[f"random_local_intensity_pearson_{window}x{window}"] = summarize(correlation(F.avg_pool2d(first, window, stride=1), F.avg_pool2d(second, window, stride=1)))
                result["pairs"].append(record)
    result["seconds"] = time.perf_counter() - started
    result["limitations"] = [
        "The relation constrains complex fields; desired target magnitudes leave their phases free.",
        "The impulse radius is not a support bound: coherent phase interference and sidelobes can redistribute power beyond it.",
        "Random-field correlations are not a bound on learned holograms or on attainable SSIM/Pearson.",
        "Independent per-image peak normalization changes displayed brightness and does not remove physical field coupling.",
        "Finite periodic discrete grids and z-dependent pitches are those of the task implementation; this is not an independent continuous-optics convergence study.",
    ]
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(cfg.output), "seconds": result["seconds"], "pairs": [{key: pair[key] for key in ("from_plane", "to_plane", "central_energy", "radius_90_pixels", "radius_95_pixels", "energy_in_15x15", "random_amplitude_pearson", "random_local_intensity_pearson_31x31")} for pair in result["pairs"]]}, indent=2))


if __name__ == "__main__":
    main()

"""Representative factor matrices for layout review (not a full combinatorial grid)."""

from dataclasses import dataclass


PLATE_POOL = ('MS', 'MS+Cd', 'MS+NaCl', 'MS+PEG')
EXTRA_POOL = ('Run A', 'Run B')
# Invented labels only — never reuse donor Experiment / genotype names.
GENO_POOL = (
    'Genotype A',
    'Genotype B',
    'Genotype C',
    'Genotype D',
    'Genotype E',
    'Genotype F',
)


@dataclass(frozen=True)
class Scenario:
    """One remixed factor layout to generate figures for."""

    id: str
    n_genotypes: int
    n_plates: int
    n_extras: int
    blurb: str

    @property
    def genotypes(self):
        return GENO_POOL[: self.n_genotypes]

    @property
    def plates(self):
        return PLATE_POOL[: self.n_plates]

    @property
    def extras(self):
        return EXTRA_POOL[: self.n_extras]


# Cover 1–6 genotypes, 1–4 plates, 1–2 extras without exploding the gallery.
SCENARIOS = (
    Scenario('g1_p1_e1', 1, 1, 1, 'Minimal: single genotype (plate/extra modes skip).'),
    Scenario('g2_p2_e1', 2, 2, 1, 'Two genotypes × two plates.'),
    Scenario('g4_p2_e2', 4, 2, 2, 'Four genotypes, 2 plates, 2 runs.'),
    Scenario('g4_p4_e2', 4, 4, 2, 'Four plate facets (wide genotype_by_plate).'),
    Scenario('g6_p3_e2', 6, 3, 2, 'Six genotypes × three plates (dense facets).'),
    Scenario('g6_p4_e2', 6, 4, 2, 'Max layout stress: 6×4×2.'),
)

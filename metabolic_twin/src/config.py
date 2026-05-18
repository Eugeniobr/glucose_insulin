"""Configuration for the metabolic twin MVP."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TwinConfig:
    data_csv: Path = Path("metabolic_twin/data/sample.csv")
    output_dir: Path = Path("metabolic_twin/artifacts")

    freq_minutes: int = 5
    lookback_hours: int = 4
    horizons_minutes: list[int] = field(default_factory=lambda: [30, 60, 120])

    model_type: str = "gru"  # "gru" or "lstm"
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1

    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 40
    patience: int = 8
    seed: int = 42

    split_strategy: str = "patient"  # "patient" or "time"
    val_frac: float = 0.15
    test_frac: float = 0.15

    @property
    def lookback_steps(self) -> int:
        return int((self.lookback_hours * 60) // self.freq_minutes)

    @property
    def horizon_steps(self) -> list[int]:
        return [int(h // self.freq_minutes) for h in self.horizons_minutes]

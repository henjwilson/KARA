from dataclasses import dataclass
# ---------------------------------------------------------------------------
# block class Variable
# ---------------------------------------------------------------------------
@dataclass
class Block:
    id: int
    color: str
    x: float
    y: float
    angle: float
    confidence: float
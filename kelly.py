"""Kelly fraccional y calculo de edge sobre probabilidad implicita."""
from __future__ import annotations


def implied_probability(decimal_odds: float) -> float:
    return 1.0 / decimal_odds


def kelly_fraction(prob: float, decimal_odds: float) -> float:
    """f* = (b*p - q) / b, con b = cuota decimal - 1, q = 1 - p."""
    b = decimal_odds - 1.0
    q = 1.0 - prob
    f = (b * prob - q) / b
    return max(0.0, f)


def fractional_kelly_stake(prob: float, decimal_odds: float,
                           bankroll: float, kelly_fraction_param: float = 0.25,
                           max_stake_fraction: float = 0.02) -> dict:
    """Devuelve stake recomendado (Kelly fraccional acotado) y edge."""
    imp = implied_probability(decimal_odds)
    edge = prob - imp
    full_kelly = kelly_fraction(prob, decimal_odds)
    stake = min(bankroll * kelly_fraction_param * full_kelly,
                bankroll * max_stake_fraction)
    return {
        "model_prob": round(prob, 4),
        "implied_prob": round(imp, 4),
        "edge": round(edge, 4),
        "full_kelly": round(full_kelly, 4),
        "stake_units": round(max(0.0, stake), 4),
        "ev_per_unit": round(prob * (decimal_odds - 1.0) - (1.0 - prob), 4),
    }

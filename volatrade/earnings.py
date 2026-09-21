"""Dates de publication des resultats trimestriels (API publique Nasdaq).

Pourquoi une deuxieme source : l'API de cours utilisee par `data.py` ne donne
plus le calendrier (les endpoints `quoteSummary` et `v7/finance/quote` de Yahoo
repondent desormais 401 « Invalid Crumb »). Nasdaq expose la meme information
sans cle ni compte.

Deux niveaux, du plus fiable au moins fiable :

1. l'annonce publiee par Nasdaq (`/api/analyst/<ticker>/earnings-date`) ;
2. a defaut, une projection a partir des dates deja publiees
   (`/api/company/<ticker>/earnings-surprise`), en prolongeant la cadence
   trimestrielle observee.

Dans les deux cas le resultat est une ESTIMATION : Nasdaq precise lui-meme que
la date provient d'un algorithme tant que l'entreprise n'a rien confirme. Une
date saisie a la main dans le portefeuille prime donc toujours.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import data
from .data import DataError, cache_path, read_cache, write_cache

ANNOUNCE_URL = "https://api.nasdaq.com/api/analyst/{ticker}/earnings-date"
HISTORY_URL = "https://api.nasdaq.com/api/company/{ticker}/earnings-surprise"

SOURCE_ANNOUNCE = "annonce Nasdaq"
SOURCE_PROJECTED = "cadence trimestrielle"

#: Duree par defaut entre deux publications, faute de mieux.
DEFAULT_QUARTER_DAYS = 91
#: Au-dela, deux publications consecutives ne forment plus une cadence credible.
MAX_QUARTER_DAYS = 120
MIN_QUARTER_DAYS = 60


@dataclass(frozen=True)
class EarningsDate:
    """Date de publication estimee pour un titre."""

    ticker: str
    date: str
    source: str

    @property
    def is_projected(self) -> bool:
        return self.source == SOURCE_PROJECTED

    def describe(self) -> str:
        return f"{self.date} ({self.source}, estimation)"


def _today(value: pd.Timestamp | None) -> pd.Timestamp:
    """Date du jour normalisee, sans fuseau, pour comparer a des dates simples."""
    stamp = pd.Timestamp(value) if value is not None else pd.Timestamp.now(tz="UTC")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    return stamp.normalize()


def _parse(text: str, formats: tuple[str, ...]) -> pd.Timestamp | None:
    """Lit une date dans l'un des formats donnes, sans deviner."""
    cleaned = " ".join(str(text).split())
    if not cleaned:
        return None
    for fmt in formats:
        try:
            parsed = pd.to_datetime(cleaned, format=fmt)
        except (ValueError, TypeError):
            continue
        # Une chaine vide ou hors format peut ressortir en NaT sans lever.
        if not pd.isna(parsed):
            return parsed.normalize()
    return None


def parse_announcement(payload: dict) -> pd.Timestamp | None:
    """Extrait la date de « Earnings announcement* for AMD: Nov 3, 2026 »."""
    data = (payload or {}).get("data") or {}
    announcement = str(data.get("announcement") or "")
    match = re.search(r":\s*(.+)$", announcement)
    if not match:
        return None
    candidate = match.group(1).strip()
    if not candidate or candidate.upper() in {"TBA", "N/A"}:
        return None
    return _parse(candidate, ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d"))


def parse_reported_dates(payload: dict) -> list[pd.Timestamp]:
    """Dates deja publiees, de la plus ancienne a la plus recente."""
    data = (payload or {}).get("data") or {}
    table = data.get("earningsSurpriseTable") or {}
    dates = []
    for row in table.get("rows") or []:
        parsed = _parse(row.get("dateReported", ""), ("%m/%d/%Y", "%Y-%m-%d"))
        if parsed is not None:
            dates.append(parsed)
    return sorted(set(dates))


def project_next(reported: list[pd.Timestamp], today: pd.Timestamp) -> pd.Timestamp | None:
    """Prolonge la cadence trimestrielle observee jusqu'a depasser `today`.

    La cadence retenue est la mediane des ecarts entre publications
    consecutives, bornee : un ecart aberrant (rattrapage comptable, changement
    d'exercice) ne doit pas produire une projection absurde.
    """
    if not reported:
        return None
    gaps = [
        (later - earlier).days
        for earlier, later in zip(reported, reported[1:])
        if MIN_QUARTER_DAYS <= (later - earlier).days <= MAX_QUARTER_DAYS
    ]
    cadence = int(pd.Series(gaps).median()) if gaps else DEFAULT_QUARTER_DAYS
    candidate = reported[-1]
    # Au plus huit trimestres : au-dela, l'historique est trop vieux pour servir.
    for _ in range(8):
        candidate = candidate + pd.Timedelta(days=cadence)
        if candidate > today:
            return candidate.normalize()
    return None


def fetch_earnings_date(
    ticker: str,
    *,
    today: pd.Timestamp | None = None,
    cache_dir: Path | str = ".volatrade_cache",
    cache_ttl: float = 24 * 3600,
    timeout: float = 20.0,
) -> EarningsDate | None:
    """Date de la prochaine publication, ou None si elle reste introuvable.

    Le cache est volontairement plus long que celui des cours (24 h par
    defaut) : une date de publication ne bouge pas d'une heure a l'autre.
    """
    today = _today(today)
    path = cache_path(cache_dir, ticker, "earnings")
    payloads = read_cache(path, cache_ttl)
    if payloads is None:
        payloads = {}
        for key, url in (
            ("announce", ANNOUNCE_URL),
            ("history", HISTORY_URL),
        ):
            try:
                payloads[key] = data.http_get_json(
                    url.format(ticker=ticker.upper()), timeout=timeout, attempts=2
                )
            except DataError:
                payloads[key] = {}
        write_cache(path, payloads)

    announced = parse_announcement(payloads.get("announce") or {})
    if announced is not None and announced >= today:
        return EarningsDate(ticker.upper(), str(announced.date()), SOURCE_ANNOUNCE)

    projected = project_next(parse_reported_dates(payloads.get("history") or {}), today)
    if projected is not None:
        return EarningsDate(ticker.upper(), str(projected.date()), SOURCE_PROJECTED)
    return None


def complete_positions(
    positions: list,
    *,
    today: pd.Timestamp | None = None,
    cache_dir: Path | str = ".volatrade_cache",
    cache_ttl: float = 24 * 3600,
    timeout: float = 20.0,
) -> list[str]:
    """Complete les `date_resultats` manquantes et renvoie les messages a afficher.

    Une date saisie a la main n'est JAMAIS ecrasee : en cas de desaccord avec
    Nasdaq, la votre est conservee et l'ecart est signale.
    """
    messages: list[str] = []
    for position in positions:
        found = fetch_earnings_date(
            position.ticker,
            today=today,
            cache_dir=cache_dir,
            cache_ttl=cache_ttl,
            timeout=timeout,
        )
        if position.date_resultats:
            if found is not None and found.date != position.date_resultats:
                messages.append(
                    f"{position.ticker} : votre date {position.date_resultats} est conservee, "
                    f"Nasdaq annonce {found.describe()}"
                )
            continue
        if found is None:
            messages.append(
                f"{position.ticker} : aucune date de publication trouvee, "
                "la regle resultats restera inactive"
            )
            continue
        position.date_resultats = found.date
        messages.append(f"{position.ticker} : date recuperee automatiquement, {found.describe()}")
    return messages

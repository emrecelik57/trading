"""Exemple : brancher votre propre API de donnees.

Copiez ce fichier, adaptez `get_history` a votre fournisseur, puis pointez la
configuration dessus :

    data:
      provider: custom
      custom: "examples.mon_provider:MonProvider"
      provider_args:
        api_key: "votre-cle"

Le fichier doit etre importable (memes repertoire que celui d'ou vous lancez
la commande, ou present sur le PYTHONPATH).

Le seul contrat a respecter :
  - renvoyer un DataFrame indexe par date, ou avec une colonne de date ;
  - fournir au minimum une colonne de cloture ;
  - livrer des prix AJUSTES des splits et dividendes.

Ce dernier point est le piege classique : avec des prix bruts, un split 4:1
ressemble a une chute de 75 % et empoisonne tous les indicateurs.
"""

from __future__ import annotations

import pandas as pd

from quantfolio.data import DataProvider


class MonProvider(DataProvider):
    """Squelette a adapter a votre API."""

    name = "mon-api"

    def __init__(self, api_key: str | None = None, base_url: str = "https://api.exemple.com"):
        self.api_key = api_key
        self.base_url = base_url
        # Ouvrez ici votre session HTTP, votre client SDK, votre connexion...

    def get_history(
        self, ticker: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        """Historique journalier de `ticker`, entre `start` et `end` inclus."""
        # --- a remplacer par votre appel reel ----------------------------
        # response = requests.get(
        #     f"{self.base_url}/v1/daily/{ticker}",
        #     params={
        #         "from": start.strftime("%Y-%m-%d"),
        #         "to": end.strftime("%Y-%m-%d"),
        #         "adjusted": "true",
        #     },
        #     headers={"Authorization": f"Bearer {self.api_key}"},
        #     timeout=20,
        # )
        # response.raise_for_status()
        # rows = response.json()["results"]
        rows: list[dict] = []
        # -----------------------------------------------------------------

        if not rows:
            return pd.DataFrame()

        frame = pd.DataFrame(rows)
        # Renommez vos colonnes vers date/open/high/low/close/volume.
        # Les variantes usuelles (Date, Adj Close, o/h/l/c/v...) sont deja
        # reconnues automatiquement, ce mapping n'est utile que pour des noms
        # vraiment specifiques a votre API.
        return frame.rename(
            columns={
                "t": "date",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
            }
        )

    def get_many(
        self, tickers: list[str], start: pd.Timestamp, end: pd.Timestamp
    ) -> dict[str, pd.DataFrame]:
        """Surchargez cette methode si votre API sait repondre en lot.

        L'implementation par defaut boucle sur `get_history` et normalise le
        resultat ; si vous la surchargez, pensez a appeler `normalize_ohlcv`
        sur chaque DataFrame renvoye.
        """
        return super().get_many(tickers, start, end)


class ProviderDepuisUnDataFrame(DataProvider):
    """Variante utile : vos donnees sont deja en memoire (base, parquet...).

        import pandas as pd
        from examples.mon_provider import ProviderDepuisUnDataFrame

        donnees = pd.read_parquet("prix.parquet")  # colonnes: date, ticker, close
        provider = ProviderDepuisUnDataFrame(donnees)
    """

    name = "dataframe"

    def __init__(self, frame: pd.DataFrame, ticker_column: str = "ticker"):
        self.frame = frame
        self.ticker_column = ticker_column

    def get_history(
        self, ticker: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        subset = self.frame[self.frame[self.ticker_column] == ticker]
        return subset.drop(columns=[self.ticker_column])

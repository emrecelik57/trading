# volatrade

Outil en ligne de commande qui **sélectionne une dizaine d'actions très volatiles**,
**mesure leur risque**, puis produit pour chacune un **plan d'achat chiffré** et les
**règles de vente** associées : où entrer, où placer le stop, quels objectifs viser,
et à quelles conditions solder.

> **Avertissement.** Outil quantitatif à but éducatif, ce n'est pas du conseil en
> investissement. Les titres sélectionnés perdent régulièrement 50 à 80 % de leur
> valeur en quelques mois — c'est même le critère de sélection. Les tailles de
> position calculées ne protègent qu'à la condition que les stops soient réellement
> respectés, et un gap à l'ouverture peut faire perdre davantage que le stop prévu.

## Installation

```bash
pip install -e .          # ou : pip install -r requirements.txt
```

Python ≥ 3.10, `numpy` et `pandas`. Les cours viennent de l'API publique de Yahoo
Finance via la bibliothèque standard : aucune clé d'API, aucun compte. Les réponses
sont mises en cache 6 heures dans `.volatrade_cache/`.

## Démarrage rapide

```bash
volatrade selection                    # les 10 titres les plus volatils du vivier
volatrade risque                       # leurs mesures de risque + corrélations
volatrade plan --capital 20000         # plans d'achat + règles de vente (défaut)
volatrade suivi -p portefeuille.json   # que faire des positions déjà ouvertes
volatrade backtest                     # ce que ces règles auraient donné
volatrade analyse TSLA NVDA            # fiche détaillée de titres précis
```

Sans sous-commande, `volatrade` exécute `plan`. Toutes les options globales
(`--capital`, `--risque`, `--top`, `--univers`, `--historique`, `--json`,
`--sans-cache`, `--config`) s'écrivent indifféremment avant ou après la
sous-commande. `python -m volatrade` fonctionne aussi.

## Ce que fait le code, étape par étape

### 1. Sélection de l'univers (`universe.py`)

Le vivier de départ (`CANDIDATE_POOL`) regroupe 53 valeurs américaines réputées
nerveuses, classées par thématique : tech nerveuse, logiciel de croissance,
proxies crypto, fintech, véhicules électriques, énergie nouvelle, spatial/défense,
quantique/IA, consommation volatile.

Le classement **n'est pas figé** : à chaque exécution, la volatilité réalisée est
recalculée et le top 10 en est déduit.

- **Score de volatilité** = 0,6 × volatilité annualisée 3 mois + 0,4 × volatilité 6 mois.
  Le mélange évite qu'un choc isolé propulse un titre en tête, tout en restant
  assez réactif pour repérer un titre qui se réveille.
- **Filtres d'éligibilité** : ≥ 200 séances d'historique, cours ≥ 3 $,
  volume médian ≥ 30 M$/jour (sinon le plan ne serait pas exécutable), et
  volatilité ≤ 250 % (au-delà, aucun stop ne pilote plus le risque).

`--univers TSLA,NVDA,AMD` remplace complètement le vivier par votre propre liste.

### 2. Mesures de risque (`metrics.py`)

| Mesure | Ce qu'elle dit |
|---|---|
| Volatilité annualisée 1 m / 3 m / 1 an | Amplitude des variations sur trois horizons |
| Volatilité EWMA (λ = 0,94) | Même chose, mais pondérée : réagit vite au changement de régime |
| Régime de volatilité | Vol. 1 mois ÷ vol. 6 mois. > 1,3 = le risque s'emballe |
| ATR(14) en % | Amplitude typique d'une séance — c'est l'unité qui calibre le stop |
| Bêta et corrélation vs S&P 500 | Sensibilité au marché (ces titres sont souvent à 3×) |
| Perte maximale (12 mois) | Pire chute pic-creux déjà vécue |
| Écart au plus haut | Où en est le titre dans sa propre chute |
| Indice d'ulcère | Pénalise les baisses profondes **et longues** |
| VaR 95 % à 1 j / 10 j | Perte dépassée dans les 5 % de séances les pires (historique) |
| CVaR 95 % | Perte moyenne le jour où la VaR est dépassée : le vrai coût des queues |
| Déviation baissière | Volatilité de la seule jambe baissière |
| Sharpe / Sortino / Calmar | Rendement par unité de risque total / baissier / de perte max |
| Asymétrie, kurtosis | Forme de la distribution : queues épaisses, sauts |
| Volume médian en devise | Liquidité réelle |

Toutes les mesures sont annualisées sur 252 séances, calculées sur rendements
logarithmiques (volatilité) ou arithmétiques (VaR, pertes en devise).

### 3. Score d'achat (`signals.py`)

Un score de 0 à 100, moyenne pondérée de quatre blocs. La thèse assumée est le
**suivi de tendance sur repli** : on achète un titre haussier qui respire, jamais
un couteau qui tombe ni une bougie verticale.

| Bloc | Poids | Contenu |
|---|---|---|
| Tendance | 35 % | Position vs MM50 et MM200, croisement MM50/MM200, pente de la MM50 |
| Momentum | 25 % | Performances 3 mois, 6 mois et 12-1 mois |
| Entrée | 20 % | RSI(14) dans la plage 45-65, repli de 2 à 12 % sous le plus haut 20 j, extension en ATR au-dessus de la MM20 |
| Risque | 20 % | Sortino, Calmar, régime de volatilité, écart au plus haut |

Traduction en action : **≥ 68 ACHAT**, **≥ 55 ACHAT PARTIEL** (demi-position),
**≥ 42 SURVEILLER**, **< 42 ÉVITER**.

Quatre garde-fous priment sur le score :

- sous la MM200 **et** momentum 6 mois < −20 % → ÉVITER, quel que soit le score ;
- sous la MM200 → demi-position au maximum ;
- régime de volatilité > 1,8 (choc en cours) → demi-position au maximum ;
- momentum trop faible (< 45/100) → demi-position : une position pleine se justifie
  par une dynamique, pas par la seule absence de mauvaise nouvelle.

### 4. Taille de position (`risk.py`)

Trois contraintes s'appliquent à chaque ligne, **la plus sévère l'emporte** :

1. **Risque par trade** — si le stop saute, la perte ne dépasse pas 1 % du capital
   (paramétrable). Quantité = (capital × 1 %) ÷ distance du stop.
2. **Cible de volatilité** — poids inversement proportionnel à la volatilité du
   titre : viser 45 % de volatilité annuelle par ligne, un titre à 90 % pèse deux
   fois moins qu'un titre à 45 %.
3. **Poids maximum** — 15 % du capital par ligne.

La distance de stop vaut 2,5 × ATR(14), avec un plancher à 3 % du cours : sur un
titre dont l'amplitude quotidienne dépasse 6 %, un stop serré serait touché par le
seul bruit intraday.

Puis quatre limites de portefeuille :

- **chaleur totale** ≤ 6 % du capital (somme des risques ouverts) ;
- **exposition totale** ≤ 100 % du capital ;
- **2 lignes maximum par thématique** — trois proxies crypto ne font pas trois paris ;
- **décote de corrélation** — jusqu'à −40 % de taille pour un titre dont la
  corrélation moyenne au reste du panier dépasse 0,5.

Quand une ligne ressort à zéro titre, le rapport dit pourquoi : limite de thème,
limite de portefeuille, ou capital insuffisant pour acheter ne serait-ce qu'un
titre en respectant le risque par trade (le montant nécessaire est alors indiqué).

### 5. Plan de vente (`plan.py`)

L'unité de compte est **R**, la distance entre le prix d'entrée et le stop initial.
Chaque plan comprend sept règles de sortie, décidées **avant** l'entrée :

| Règle | Déclencheur | Action |
|---|---|---|
| Stop initial | Cours sous entrée − 2,5 ATR | Vendre 100 %, sans discussion |
| Objectif 1 | +1,5 R | Vendre 33 %, remonter le stop au point mort |
| Objectif 2 | +3 R | Vendre 33 %, laisser courir le solde |
| Stop suiveur | Plus haut 22 séances − 3 ATR | Vendre le solde s'il est touché |
| Rupture de tendance | 2 clôtures à plus de 1 ATR sous la MM50 | Vendre la totalité du solde |
| Stop temporel | 25 séances sans atteindre +0,5 R | Solder : le capital travaille mieux ailleurs |
| Choc de volatilité | Vol. 1 mois > 1,8 × vol. 6 mois | Réduire de moitié, même sans signal de prix |

L'horizon affiché est estimé par marche aléatoire : le temps moyen pour parcourir
3 R vaut (distance ÷ volatilité quotidienne)² séances. C'est un ordre de grandeur,
pas une prévision.

### 6. Suivi des positions ouvertes (`positions.py`)

`volatrade suivi` applique ces mêmes règles à un portefeuille réel et rend un
verdict par ligne : **VENDRE**, **ALLÉGER**, **REMONTER LE STOP** ou **CONSERVER**,
avec le gain latent en devise, en pourcentage et en R.

```json
[
  {"ticker": "MU", "shares": 3, "entry_price": 700.0, "entry_date": "2026-06-02",
   "stop_price": 640.0, "target1": 790.0, "target2": 880.0, "trimmed": []}
]
```

Seuls `ticker`, `shares` et `entry_price` sont obligatoires : un stop ou un objectif
absent est reconstruit à partir de l'ATR courant. `trimmed` liste les objectifs déjà
encaissés (`"objectif1"`, `"objectif2"`) pour qu'ils ne soient pas reproposés et
pour que le stop ne redescende jamais sous le point mort. Voir
`examples/portefeuille.example.json`.

## Backtest : ce que valent ces règles

`volatrade backtest` rejoue exactement la logique de production — score calculé sur
les seules données disponibles à la date de décision, entrée à l'ouverture de la
séance suivante, sorties selon les sept règles ci-dessus.

Mesuré le 8 septembre 2026 sur le top 10 du vivier, 2 ans d'historique, décision
tous les 5 jours :

```
Trades               133
Taux de réussite     39 %
Gain moyen           +0.13 R (médiane -0.09 R)
Facteur de profit    1.43
Résultat cumulé      +17.1 R
Pire série           -9.6 R
Durée moyenne        9 séances
Performance capital  17 %  (en risquant 1 % à chaque trade)
Acheter-conserver    85 %  en moyenne sur les mêmes titres
```

**Lecture honnête** : sur cette période franchement haussière, acheter et conserver
ces dix titres aurait rapporté cinq fois plus. Le jeu de règles gagne peu souvent
(39 %) mais gagne gros quand il gagne (facteur de profit 1,43), et surtout il borne
chaque perte à environ 1 R — là où un achat-conservation encaisse les −60 % à −80 %
que ces titres traversent régulièrement. C'est un arbitrage rendement contre
drawdown, pas une machine à surperformer.

Trois biais à garder en tête : l'univers est sélectionné sur la volatilité
**d'aujourd'hui** (les titres testés sont ceux qui ont survécu jusqu'ici), la
période testée est un marché haussier, et ni frais ni glissement ne sont modélisés.

La marge d'un ATR sous la MM50 avant de déclarer la tendance cassée vient de ce
backtest : sans elle, le bruit quotidien déclenchait une sortie toutes les trois
séances, pour un gain moyen de +0,01 R au lieu de +0,13 R.

## Configuration

Toutes les valeurs par défaut sont dans `config/config.example.json` :

```bash
volatrade plan --config config/config.example.json
```

| Clé | Défaut | Rôle |
|---|---|---|
| `capital` | 10000 | Capital total du portefeuille |
| `devise` | `"$"` | Symbole affiché |
| `risque_par_trade` | 0.01 | Perte maximale par ligne, en fraction du capital |
| `multiple_atr_stop` | 2.5 | Distance du stop, en ATR(14) |
| `poids_max` | 0.15 | Poids maximum d'une ligne |
| `volatilite_cible` | 0.45 | Volatilité annuelle visée par ligne |
| `chaleur_max` | 0.06 | Somme maximale des risques ouverts |
| `exposition_max` | 1.0 | Part maximale du capital investie |
| `max_positions_par_theme` | 2 | Lignes maximum par thématique |
| `nombre_titres` | 10 | Taille de la sélection |
| `volume_min` | 30000000 | Volume médian minimum, en devise par jour |
| `prix_min` | 3.0 | Cours plancher |
| `volatilite_max` | 2.5 | Volatilité au-delà de laquelle un titre est écarté |
| `univers` | vivier interne | Liste de tickers |
| `indice_reference` | `"SPY"` | Référence pour le bêta et la corrélation |
| `historique` | `"2y"` | Profondeur téléchargée |
| `taux_sans_risque` | 0.03 | Taux annuel pour Sharpe et Sortino |
| `cache_ttl_heures` | 6 | Durée de vie du cache de cours |
| `dossier_cache` | `.volatrade_cache` | Emplacement du cache |

Une clé inconnue ou une valeur hors bornes est refusée avec un message explicite.

## Structure du code

```
volatrade/
  data.py        téléchargement des cours, ajustement splits/dividendes, cache
  universe.py    vivier de candidats et filtrage par volatilité/liquidité
  metrics.py     volatilité, VaR/CVaR, drawdowns, Sharpe/Sortino/Calmar, ATR, RSI
  signals.py     score d'achat en quatre blocs et garde-fous
  risk.py        dimensionnement des positions et limites de portefeuille
  plan.py        plan de trade : entrée, stop, objectifs, règles de sortie
  positions.py   suivi des positions ouvertes et verdicts de vente
  backtest.py    rejeu historique des règles, sans look-ahead
  engine.py      enchaînement complet données -> plans
  report.py      tableaux et fiches pour le terminal
  config.py      configuration et validation
  cli.py         interface en ligne de commande
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

122 tests, aucun accès réseau : les cours sont synthétiques ou injectés, y compris
pour les tests de bout en bout de la ligne de commande. Ils couvrent notamment la
calibration des mesures de risque sur des séries aux propriétés connues,
l'application des filtres de sélection, les garde-fous du score, la primauté de la
contrainte la plus sévère dans le dimensionnement, chaque règle de vente, et
l'absence de biais de look-ahead dans le backtest.

## Limites connues

- Actions américaines uniquement (l'API utilisée accepte d'autres places, mais le
  vivier et les filtres de liquidité sont calibrés pour les États-Unis).
- Données de fin de séance : les plans se lisent avant l'ouverture suivante, pas en
  intraday.
- Aucune donnée fondamentale, aucun calendrier de résultats — or une publication
  trimestrielle est la première cause de gap au-delà du stop sur ces titres.
- Les paramètres (2,5 ATR, +1,5 R / +3 R, 25 séances) sont des valeurs de bon sens
  vérifiées par backtest, pas des optima : les optimiser sur deux ans d'historique
  reviendrait surtout à mémoriser le passé.

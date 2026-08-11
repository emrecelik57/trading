# quantfolio

Outil de gestion de portefeuille actions : chaque jour, il classe les titres de
votre univers, en déduit un portefeuille cible, et vous dit **quoi acheter, quoi
vendre et en quelle quantité**.

```
$ quantfolio daily

SEANCE DU 2026-08-10
====================
  Modele : gbm, 40200 observations x 19 indicateurs du 2018-07-30 au 2026-07-29

PORTEFEUILLE CIBLE
==================
  Exposition visee 98.0% | liquidites 2.0% | vol estimee 12.0% vs cible 15.0%
  ticker  score  vol ann.  poids
  ------  -----  --------  -----
  UNH     +1.00  26.0%     17.8%
  JNJ     +0.90  25.0%     17.0%
  XOM     +0.70  29.2%     12.1%
  GOOGL   +0.80  38.0%     10.3%
  ...

CLASSEMENT DU JOUR
------------------
  Bas de classement (a eviter / vendre) :
    COST  -0.700
    MSFT  -0.800
    AVGO  -0.900

ORDRES A PASSER
===============
  sens   ticker  qte  cours   montant  poids act.  poids cible  score  motif
  -----  ------  ---  ------  -------  ----------  -----------  -----  ---------
  ACHAT  UNH     43   408.74  17,576   0.0%        17.8%        +1.00  ouverture
  ACHAT  JNJ     65   261.81  17,018   0.0%        17.0%        +0.90  ouverture
  ACHAT  XOM     75   159.79  11,984   0.0%        12.1%        +0.70  ouverture

  10 achat(s) pour 96,614 USD | frais estimes 48 USD
  Capital 100,000 USD | liquidites 100,000 -> 3,338 USD | rotation 96.6%
```

Le modèle et la gestion du risque sont fournis ; **la source de données est un
point de branchement** : vous mettez votre API derrière une classe de dix lignes.

---

## Installation

```bash
git clone <ce-depot> && cd trading
pip install -e .          # ou : pip install -r requirements.txt
```

Python 3.10+. Dépendances : numpy, pandas, scikit-learn, PyYAML.

## Démarrage rapide

```bash
quantfolio init                              # crée config/config.yaml
$EDITOR config/config.yaml                   # votre univers, votre capital
quantfolio backtest                          # la stratégie tient-elle la route ?
quantfolio init-portfolio --cash 50000       # déclarez votre portefeuille
quantfolio daily                             # les ordres du jour
```

Pour essayer sans aucune donnée ni réseau, mettez `data.provider: synthetic` :
l'outil génère des historiques simulés et tout le pipeline tourne.

Le flux quotidien ensuite :

```bash
quantfolio daily                                    # affiche + écrit output/orders_AAAA-MM-JJ.json
# ... vous passez les ordres chez votre courtier ...
quantfolio apply --file output/orders_2025-11-14.json   # enregistre l'exécution
quantfolio portfolio                                # état, P&L, valorisation
```

Si vos prix d'exécution réels diffèrent des prix estimés, éditez le JSON avant
`apply` : ce sont eux qui font foi pour votre prix de revient.

---

## Brancher votre API

Une classe, une méthode :

```python
# mon_provider.py
import pandas as pd
from quantfolio.data import DataProvider

class MonProvider(DataProvider):
    name = "mon-api"

    def __init__(self, api_key=None):
        self.api_key = api_key

    def get_history(self, ticker, start, end) -> pd.DataFrame:
        rows = mon_client.daily(ticker, start, end, adjusted=True)
        return pd.DataFrame(rows)   # colonnes date/open/high/low/close/volume
```

```yaml
data:
  provider: custom
  custom: "mon_provider:MonProvider"
  provider_args:
    api_key: "..."
```

L'outil se charge du reste : normalisation des en-têtes (`Date`, `Adj Close`,
`o/h/l/c/v`... sont reconnus), tri, déduplication, cache disque, alignement des
calendriers entre titres, gestion des IPO et des retraits de cote.

> **Le seul vrai piège : donnez des prix ajustés des splits et dividendes.**
> Avec des prix bruts, un split 4:1 ressemble à une chute de 75 % et fausse tous
> les indicateurs. Si votre API renvoie `close` et `adj_close`, prenez le second.

Voir `examples/mon_provider.py` pour un squelette commenté, y compris la
variante « mes données sont déjà dans un DataFrame ».

Trois fournisseurs sont livrés d'origine : `csv` (fichiers `<TICKER>.csv`),
`yahoo` (API publique, sans dépendance, pratique pour démarrer) et `synthetic`.

---

## Comment ça marche

### 1. Le modèle classe, il ne prédit pas le marché

Prévoir si le marché va monter est très difficile. Prévoir que **NVDA fera mieux
que XOM le mois prochain** l'est nettement moins. quantfolio ne fait donc que du
**classement transversal** : chaque jour, il ordonne les titres entre eux.

Concrètement, dix-neuf indicateurs sont calculés par titre (momentum 1/3/6/12
mois, momentum 12-1, volatilité réalisée et *downside*, RSI, MACD, ATR, distance
aux moyennes mobiles 50 et 200, croisement de moyennes, Bollinger, retournement
court terme, drawdown, asymétrie, ratio de volume). Chacun est ensuite remplacé
par **son rang parmi les titres cotés ce jour-là**, ramené dans `[-1, +1]`.

Cette normalisation est le cœur du dispositif : elle rend les indicateurs
comparables entre titres et stables dans le temps. Une volatilité de 40 % ne veut
rien dire en soi ; « la plus volatile de l'univers » veut dire quelque chose dans
tous les régimes de marché.

La cible apprise est le **rang du rendement futur en excès de la moyenne de
l'univers** à un horizon de 5 séances. On retire ainsi la composante marché : le
modèle apprend le classement relatif, pas la direction générale.

Trois modèles au choix (`model.kind`) :

| valeur  | ce que c'est                         | quand l'utiliser                            |
|---------|--------------------------------------|---------------------------------------------|
| `gbm`   | gradient boosting (défaut)           | historique fourni, univers large            |
| `ridge` | linéaire régularisé                  | petit univers, ou pour un modèle lisible    |
| `rules` | multi-facteur, aucun apprentissage   | démarrage, ou si vous vous méfiez du ML     |

Le score final mélange le modèle et un score multi-facteur classique
(momentum + tendance + faible volatilité + retournement), pondéré par
`model.rules_weight`. Ce mélange sert d'a priori : il donne un système
exploitable dès le premier jour et amortit les décisions du modèle quand
celui-ci est peu sûr. Monter `rules_weight` vers 1.0 rend le système plus
prudent et plus stable ; le descendre vers 0 laisse le modèle décider seul.

### 2. Du score au portefeuille

- **Sélection** : les `max_positions` meilleurs scores au-dessus de
  `score_threshold`.
- **Pondération** : proportionnelle à la conviction **divisée par la
  volatilité**. À conviction égale, un titre deux fois plus volatil reçoit une
  position deux fois plus petite, pour que chaque ligne pèse autant en risque.
- **Plafonds** : `max_weight` par ligne, l'excédent est redistribué ; les
  positions sous `min_weight` sont supprimées.
- **Ciblage de volatilité** : les poids sont mis à l'échelle pour viser
  `vol_target` de volatilité annualisée, estimée sur la covariance récente
  (contractée vers sa diagonale pour rester stable).
- **Filtre de régime** : quand le marché passe sous sa moyenne mobile 200 jours,
  l'exposition est réduite (`regime_risk_off`). C'est là que les stratégies de
  momentum souffrent le plus.

### 3. Du portefeuille aux ordres

Deux garde-fous évitent que les frais mangent le signal :

- **Bande de non-négociation** (`no_trade_band`) : aucun ordre tant que l'écart
  de poids reste sous le seuil. Les entrées et sorties complètes ne sont jamais
  filtrées.
- **Hystérésis** (`exit_threshold`, `hold_bonus`) : un titre détenu n'est vendu
  que si son score tombe nettement plus bas que le seuil d'achat, et il n'est pas
  remplacé par un candidat à peine mieux classé. **C'est le levier principal sur
  la rotation** : sans lui, le portefeuille se reconstruit à chaque séance.

Les ventes sont calculées avant les achats, pour que le produit des cessions
finance les acquisitions du même jour. Les achats ne dépassent jamais les
liquidités disponibles, et aucune vente à découvert n'est générée.

---

## Lire les résultats

### Le score

Un rang dans `[-1, +1]` : `+1` désigne le titre le mieux classé du jour, `-1` le
moins bien classé. C'est une **conviction relative, pas une prévision de
rendement** : un score de +0.9 ne dit pas « ce titre va monter de 9 % », il dit
« parmi cet univers, aujourd'hui, c'est celui sur lequel le modèle est le plus
confiant ».

### L'IC (information coefficient)

C'est la mesure de référence d'un modèle de classement : la corrélation de rang,
jour par jour, entre le score prédit et le rendement réellement observé.

```bash
quantfolio evaluate
```

| IC moyen      | interprétation                                             |
|---------------|------------------------------------------------------------|
| ~0.00         | le modèle n'apporte rien de plus que le hasard             |
| 0.02 – 0.05   | signal faible mais réel et exploitable, si t-stat > 2      |
| > 0.15        | **suspectez une fuite de données**, pas un bon modèle      |

Le t-stat compte autant que la moyenne : un IC de 0.04 stable sur 500 dates vaut
bien mieux qu'un IC de 0.10 obtenu sur 30 dates.

### Le backtest

`quantfolio backtest` déroule la stratégie séance par séance, avec des règles
strictes :

- le modèle n'est **jamais** entraîné sur des données postérieures à la décision ;
- les observations dont le label chevauche la date de décision sont **purgées**
  (horizon + `embargo_days`), ce qui évite la fuite la plus courante ;
- un ordre décidé le jour J est **exécuté au jour J+1**, aux prix de J+1 ;
- frais et slippage sont prélevés sur chaque transaction.

Le résultat reste **une simulation optimiste**, et il faut le lire comme tel :

- **Biais du survivant** — si votre univers est la liste des 20 plus grosses
  capitalisations *d'aujourd'hui*, vous testez la stratégie sur les gagnants
  connus d'avance. C'est de loin le biais le plus trompeur, et il vient de votre
  univers, pas du code. Utilisez l'univers tel qu'il était à l'époque si vous
  le pouvez.
- **Liquidité** — le backtest suppose que vos ordres sont exécutés en totalité au
  prix affiché.
- **Sur-ajustement** — si vous réglez les paramètres jusqu'à ce que le backtest
  soit beau, vous avez ajusté le passé, pas trouvé un signal. Décidez de vos
  paramètres avant de regarder, et vérifiez le résultat sur une période que vous
  n'avez pas utilisée pour les choisir.

Comparez toujours au benchmark affiché : une stratégie qui fait 12 % quand
l'indice fait 15 % n'est pas une bonne stratégie, même si 12 % fait plaisir.

### Ce que donne la configuration par défaut

Voici ce que produit la config livrée, telle quelle, sur les 20 grandes
capitalisations américaines de l'exemple, de 2018 à 2026 :

| `score_smoothing` | CAGR | Vol | Sharpe | Max DD | Rotation/an | IC |
|---|---|---|---|---|---|---|
| 1 (aucun lissage) | 10.4 % | 13.8 % | 0.79 | −22.5 % | 2977 % | +0.021 |
| **5 (défaut)** | 10.2 % | 13.9 % | 0.77 | −21.6 % | 2411 % | +0.013 |
| 10 | 10.9 % | 14.0 % | 0.81 | −22.3 % | 1892 % | +0.012 |

Sur la même période, **le S&P 500 fait 14.8 % par an**, avec 19.1 % de
volatilité et un drawdown maximal de −33.7 %.

Lisez ce tableau honnêtement, il dit trois choses :

1. **La stratégie ne bat pas l'indice en absolu**, et l'écart est net. Elle prend
   en revanche nettement moins de risque : beta 0.54, drawdown réduit d'un tiers.
   Le rapport des rendements (0.70) est quasiment celui des volatilités (0.72) —
   autrement dit, sur cet échantillon, **le modèle n'a pas apporté d'avantage
   mesurable au-delà du fait de prendre moins de risque** (Sharpe 0.77 contre
   0.82 pour l'indice). Si vous voulez une exposition comparable à celle du
   marché, montez `vol_target` ; ne vous attendez pas pour autant à de l'alpha.
2. **Le lissage réduit franchement la rotation** (−36 % entre 1 et 10) pour un
   effet sur la performance qui reste dans le bruit. Le défaut est fixé à 5, soit
   l'horizon de prévision — choisi *avant* de regarder ce tableau. La valeur 10 y
   fait un peu mieux, mais retenir un paramètre parce qu'il gagne sur un seul
   échantillon, c'est exactement le sur-ajustement décrit plus haut.
3. **L'IC baisse quand on lisse** (0.021 → 0.012) : le lissage efface une part du
   signal court terme en même temps que le bruit. Il se paie, simplement moins
   cher qu'il ne rapporte en frais évités.

Un IC de 0.02 avec un t-stat de 1.5 sur 430 dates, c'est un signal ténu, à la
limite du significatif. Ce n'est pas un défaut de l'outil : c'est ce à quoi il
faut s'attendre sur 20 méga-capitalisations très suivies. Un univers plus large
et moins efficient laisse davantage de place au classement.

---

## Commandes

| commande | rôle |
|---|---|
| `quantfolio init` | crée `config/config.yaml` |
| `quantfolio fetch` | télécharge et met en cache les données, signale les titres à l'historique trop court |
| `quantfolio train` | entraîne le modèle et le sauvegarde (`--importance` pour voir les indicateurs qui comptent) |
| `quantfolio evaluate` | mesure l'IC hors échantillon |
| `quantfolio backtest` | simule la stratégie, écrit courbe / trades / poids dans `output/` |
| `quantfolio daily` | recommandations du jour (`--date` pour rejouer une séance passée) |
| `quantfolio apply --file ...` | enregistre l'exécution des ordres |
| `quantfolio portfolio` | positions, P&L, valorisation (`--history N`) |
| `quantfolio init-portfolio --cash X` | crée le portefeuille |
| `quantfolio set-position TICKER --shares N --price P` | déclare une position déjà détenue |

Aussi utilisable comme bibliothèque :

```python
from quantfolio import Config, Engine

cfg = Config.from_yaml("config/config.yaml")
engine = Engine(cfg)
engine.prepare()
ranker = engine.fit()

date = engine.last_date()
scores = engine.score_at(date, ranker)
target = engine.target_at(date, ranker, scores=scores)
print(target.holdings)
```

---

## Réglages qui comptent

Tout est dans `config/config.yaml`, commenté. Les leviers réellement utiles :

**Si la rotation est trop forte** (les frais mangent la performance)
→ augmentez `no_trade_band`, baissez `exit_threshold`, augmentez `hold_bonus`,
ou espacez les rebalancements (`rebalance_days`).

**Si le portefeuille est trop concentré**
→ baissez `max_weight`, montez `max_positions`.

**Si ça bouge trop violemment**
→ baissez `vol_target` ; c'est le réglage qui pilote directement l'amplitude des
variations de votre capital.

**Si vous ne faites pas confiance au modèle**
→ montez `model.rules_weight` (1.0 = uniquement le multi-facteur, sans ML), ou
passez `model.kind: rules`.

**Univers** : en dessous de ~15 titres, un classement transversal a peu de sens —
il n'y a pas grand-chose à classer. Visez au moins 20 à 30 titres.

---

## Développement

```bash
pip install -e ".[dev]"
pytest                    # 134 tests
pytest -m "not slow"      # sans les tests d'apprentissage
```

La suite couvre en priorité ce qui peut vous coûter de l'argent :

- **`test_leakage.py`** — le plus important. Vérifie notamment que le score
  calculé à une date est *identique* que l'historique soit tronqué à cette date
  ou complet. Si un indicateur regardait vers l'avant, ce test échouerait.
- **`test_model.py`** — vérifie que le modèle atteint un IC franchement positif
  sur un univers où un signal existe par construction, et reste à zéro sur un
  univers sans structure. Un modèle qui « marche » sur du bruit est un modèle qui
  triche.
- Le reste couvre les contraintes de portefeuille, les ordres (jamais plus que
  les liquidités, jamais de vente à découvert), la comptabilité du portefeuille,
  la cohérence du backtest et la validation de la configuration.

Structure :

```
quantfolio/
├── data/          sources de données, cache, alignement des calendriers
├── features.py    indicateurs + normalisation transversale
├── labels.py      cible d'apprentissage (rendement futur en excès)
├── rules.py       score multi-facteur sans apprentissage
├── model.py       modèle de classement (scikit-learn)
├── engine.py      pipeline complet, purge temporelle
├── risk.py        volatilité, covariance, filtre de régime
├── portfolio.py   sélection, pondération, exposition cible
├── orders.py      portefeuille cible → ordres concrets
├── state.py       positions détenues, PRU, P&L
├── backtest.py    simulation walk-forward
└── cli.py         ligne de commande
```

---

## Avertissement

Cet outil est une aide à la décision. Les scores sont des estimations
statistiques, jamais des certitudes, et un backtest flatteur ne garantit rien sur
l'avenir. Aucun ordre n'est transmis à un courtier : vous restez seul décideur de
ce que vous exécutez. Investir comporte un risque de perte en capital.

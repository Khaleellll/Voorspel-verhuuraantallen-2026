# Voorspel verhuuraantallen 2026 — Portfolio-opdracht 2

Machine Learning, De Haagse Hogeschool, Data Science & AI (2026-2027)

CityBike wil per uur voorspellen hoeveel fietsen er worden verhuurd (`cnt`). We trainen modellen
op de data van 1-1-2011 t/m 30-11-2012 en voorspellen 1 t/m 19 december 2012 voor de
Kaggle-competitie **Voorspel verhuuraantallen 2026**.

**Eis:** RMSE op Kaggle lager dan **80**.

---

## Team

| Naam | GitHub | Kaggle-gebruikersnaam | Branch |
|---|---|---|---|
| Khaleel Al-Aqel | [@Khaleellll](https://github.com/Khaleellll) | khaleelalaqel | `khaleel` |
| Ahmed Bashabeeb | [@AhmedBashabeeb](https://github.com/AhmedBashabeeb) | ahmedbashabeeb | `ahmed` |
| Mehmet Kilinc | [@Mehmet-hhs](https://github.com/Mehmet-hhs) | mehmetkilinc123 | `mehmet` |

**Kaggle-teamnaam:** *[invullen]*

---

## Plan van aanpak

### Taakverdeling

| Onderdeel | Wie | Status | Inhoud |
|---|---|---|---|
| **Organisatie** | Khaleel | ✅ klaar | Kaggle-team aangemaakt · GitHub-repository aangemaakt · taken verdeeld · werkstructuur vastgelegd (branches, mappen, afspraken in deze README) |
| **Opdracht 1: EDA** | Khaleel | ✅ klaar | Data inladen (`DataLoader`) · datatypes en basisstatistieken · datakwaliteit (missende waarden, ontbrekende uren, onjuiste waarden) · tijdreekselementen toevoegen · visualisaties van patronen (dag, week, jaar, trend) · relaties tussen de variabelen en `cnt` · data opschonen · belangrijkste bevindingen · data in `data/` |
| **Opdracht 2: Time series feature engineering** | Ahmed | | Evaluatiemethode (`TimeSeriesSplit`, RMSE) · onderzoek naar trend · seizoenspatronen met Fourier-analyse (periodogram) · cycli en autocorrelatie (ACF/PACF) · overige tijdsfeatures · `FeatureEngineer`-klasse · experiment: invloed van de features · onderbouwde keuze van de features |
| **Opdracht 3: Modelleren** | Mehmet | | `ModelTrainer`-klasse (GridSearchCV + cross-validatie) · lineaire regressie met regularisatie (wiskundige uitleg: loss function, Ridge/Lasso) · ensemble-modellen (Random Forest, XGBoost, met wiskundige uitleg) · minstens 5 modellen in totaal · SARIMAX + een tweede tijdreeksmodel · hybride model (residuen) · hyperparameter tuning per model · submission-bestand per model |
| **Bevindingen** | Mehmet | | Overzicht van de resultaten · beste model en parameters · Kaggle-scores en verbeteringen |
| **Kaggle** | Mehmet | | Alle modellen uploaden · scores bijhouden in de tabel hieronder |
| **Opdracht 4: Conclusie en aanbevelingen** | Ahmed | | Belangrijkste features · aansluiting op de EDA · advies aan CityBike · modeladvies (en andere overwegingen dan alleen de score) |
| **Bronnenlijst** | Ahmed | | Alle bronnen samenvoegen in APA-stijl, inclusief GenAI-prompts (zie afspraken) |
| **Eindversie en inleveren** | Ahmed (samenvoegen en inleveren), Mehmet controleert | | Alles samenvoegen in één notebook · notebook draait van begin tot eind · PEP8-controle · inleveren |

### Khaleel: wat er klaar is

**Organisatie**

- Kaggle-team aangemaakt voor de competitie *Voorspel verhuuraantallen 2026*.
- GitHub-repository aangemaakt, met een eigen branch per teamlid.
- Taken verdeeld (tabel hierboven) en de planning gemaakt.
- Werkstructuur vastgelegd: mappenstructuur, werkwijze met branches en pull requests, en afspraken
  over code, bronnen en GenAI (zie hieronder).

**Opdracht 1: EDA** (alles staat in `main`)

- **`notebooks/opdracht1_eda.ipynb`**: de volledige EDA (paragraaf 1.1 t/m 1.10), met bronnen en de
  link naar het GenAI-gesprek (prompt 1 t/m 4).
- **`data/`**: `train.csv`, `test.csv` en `sample_submission.csv`.
- **Klasse `DataLoader`** (paragraaf 1.2): kopieer deze naar je eigen notebook. De opgeschoonde data
  krijg je zo:

  ```python
  loader = DataLoader(os.path.join('..', 'data'))
  train_raw = loader.load('train.csv')
  train = loader.clean(train_raw, ['2012-10-29', '2012-10-30'])
  ```

- **Opschoning:** `hum == 0` (22 meetfouten) vervangen door interpolatie, 29–30 oktober 2012
  (uitzonderlijke dagen) verwijderd. In de trainingsdata ontbreken 163 uren.
- **Belangrijkste bevindingen voor Opdracht 2 en 3** (zie 1.10): het uur is de grootste bron van
  variatie (η = 0,71), en uur en type dag moeten **samen** gebruikt worden (interactie: twee
  spitsen op werkdagen, één middagpiek in het weekend). Verder: trend van 2011 naar 2012,
  jaarseizoen, en het weer (`temp`, `hum`, `weathersit`). `cnt` is scheef verdeeld, dus een
  log-schaal of Poisson-loss kan helpen.

### Afhankelijkheden

De opdrachten bouwen op elkaar voort. Om niet op elkaar te wachten:

1. ✅ **Khaleel** heeft `DataLoader`, de opgeschoonde data en de EDA in `main` gezet. Ahmed en
   Mehmet gebruiken die.
2. **Ahmed** zet zo snel mogelijk een eerste versie van `FeatureEngineer` in `main`. Daarna kan hij
   de features verder verbeteren.
3. **Mehmet** kan meteen beginnen met `ModelTrainer` en de modeluitleg, en test eerst met eenvoudige
   features (uur, weekdag, weer).

### Planning

| Week | Datum | Doel |
|---|---|---|
| 1 | 12/11 | Repo en Kaggle-team klaar · `DataLoader` + opgeschoonde data in `main` · eerste Kaggle-submission (simpel model) |
| 2 | 16/11 | Opdracht 1 af · eerste versie features in `main` · eerste modellen + submissions |
| 3 | 17/11 | Opdracht 2 en 3 af · alle modellen op Kaggle · bevindingen en conclusie samen schrijven |
| 4 | 20/11 | Alles samengevoegd · eindcontrole · inleveren (**deadline: [invullen]**) · voorbereiding assessment |

---

## Mappenstructuur

```
├── README.md
├── data/                          # train.csv, test.csv, sample_submission.csv
├── notebooks/
│   ├── opdracht1_eda.ipynb        # Khaleel
│   ├── opdracht2_features.ipynb   # Ahmed
│   └── opdracht3_modellen.ipynb   # Mehmet
├── submissions/                   # csv-bestanden voor Kaggle
└── Portfolio2_CityBike.ipynb      # eindversie (samengevoegd)
```

Iedereen werkt in **een eigen notebook**. Zo zijn er geen merge-conflicten (twee mensen die
hetzelfde `.ipynb`-bestand aanpassen geeft bijna altijd conflicten). Aan het eind voegt Ahmed de
delen samen in `Portfolio2_CityBike.ipynb`.

---

## Werkwijze met GitHub

### Eenmalig: repo binnenhalen en eigen branch maken

```bash
git clone https://github.com/Khaleellll/Voorspel-verhuuraantallen-2026.git
cd Voorspel-verhuuraantallen-2026
git checkout -b ahmed          # gebruik je eigen naam: khaleel / ahmed / mehmet
git push -u origin ahmed
```

### Elke keer dat je gaat werken

```bash
git checkout ahmed             # naar je eigen branch
git pull origin main           # eerst de nieuwste versie van main ophalen
# ... werken in je notebook ...
git add notebooks/opdracht2_features.ipynb
git commit -m "Opdracht 2: Fourier-analyse toegevoegd"
git push
```

### Werk inleveren in `main`

1. Ga op GitHub naar de repo → **Pull requests** → **New pull request**.
2. Kies `base: main` ← `compare: <jouw branch>` en schrijf kort wat je hebt gedaan.
3. Het andere teamlid (Ahmed of Mehmet) controleert de pull request en klikt op **Merge**.
4. Daarna doet iedereen `git pull origin main` op de eigen branch.

### Afspraken

- **Nooit direct op `main` werken**: alleen via een pull request.
- **Kleine commits** met een duidelijke boodschap (`Opdracht 1: boxplot weersituatie`).
- **Outputs wissen voor een commit** (*Kernel → Restart & Clear Output*), behalve in de eindversie.
  Dit houdt de bestanden klein en de verschillen leesbaar.
- **Code:** PEP8, functies of klassen (OOP), comments in het Nederlands, dezelfde naamgeving als de
  rest (Engelse variabelenamen, bijv. `train_data`).
- **Bronnen:** elke bron in APA-stijl. Gebruik je GenAI, zet dan in de code-cel
  `# Prompt <nr>: <titel>` en in de markdown/bronnenlijst de naam en versie van het model, de datum,
  het promptnummer, de titel van de chat en de link. Voeg ook een niet-GenAI bron toe die de
  informatie bevestigt. Maximaal 25% van de code mag letterlijk uit bronnen komen.
- **Iedereen begrijpt alle code** en kan die uitleggen bij het assessment. Leg je eigen deel uit aan
  de anderen voordat het in `main` komt.
- **Kaggle:** elke submission noteren in de tabel hieronder.

---

## Installatie

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost statsmodels prophet
```

Zet `train.csv`, `test.csv` en `sample_submission.csv` (van Kaggle) in de map `data/`.

---

## Kaggle-scores

| # | Datum | Wie | Model / verbetering | Bestand | CV-RMSE | Kaggle public |
|---|---|---|---|---|---:|---:|
| 1 | 12-11-2026 | Khaleel | XGBoost (ronde 1) | `submission_xgb.csv` | — | 49,28172 |
| 2 | 12-11-2026 | Khaleel | Beslisboom (ronde 1) | `submission_tree.csv` | — | 62,11089 |
| 3 | 12-11-2026 | Khaleel | SVR (ronde 1) | `submission_svr.csv` | — | 63,40589 |
| 4 | 12-11-2026 | Khaleel | SARIMAX (ronde 1) | `submission_sarimax.csv` | — | 54,84523 |
| 5 | 12-11-2026 | Khaleel | Random Forest (ronde 1) | `submission_rf.csv` | — | 66,13760 |
| 6 | 12-11-2026 | Khaleel | Prophet (ronde 1) | `submission_prophet.csv` | — | 73,82706 |
| 7 | 12-11-2026 | Khaleel | Lineaire regressie (ronde 1) | `submission_linear.csv` | — | 56,14528 |
| 8 | 12-11-2026 | Khaleel | Hybride model (ronde 1) | `submission_hybrid.csv` | — | 49,42684 |
| 9 | 12-11-2026 | Khaleel | Gemiddelde van XGBoost en het hybride model (ronde 1) | `submission_average.csv` | — | 47,44858 |
| 10 | 27-11-2026 | Mehmet | Hybride model met Poisson-loss | `submission_hybrid_poisson.csv` | 42,8 | **40,99780** |
| 11 | 27-11-2026 | Mehmet | Gradient Boosting | `submission_gb.csv` | 50,37 | 41,16671 |
| 12 | 27-11-2026 | Mehmet | Seizoensbaseline | `submission_baseline.csv` | 70,32 | 49,77962 |

Onze beste openbare Kaggle-score is **40,99780** met het hybride Poisson-model.
Dat is **6,45 RMSE-punten lager** dan onze beste score uit ronde 1 (47,44858).
Daarmee halen we de eis van een Kaggle-RMSE onder 80.

---

## Checklist inleveren

- [ ] Eén net notebook met genummerde hoofdstukken en paragrafen (markdown)
- [ ] Notebook draait van begin tot eind zonder fouten
- [ ] Code voldoet aan PEP8, met comments en zelfgedefinieerde functies/klassen
- [ ] Teamnaam, namen en Kaggle-gebruikersnamen staan in het notebook
- [ ] Minstens één Kaggle-submission met RMSE < 80
- [ ] Minstens 5 modellen + SARIMAX + tweede tijdreeksmodel + hybride model, allemaal getuned met cross-validatie
- [ ] Bronnenlijst in APA-stijl (inclusief GenAI-prompts)
- [ ] Iedereen kan alle code en tekst uitleggen

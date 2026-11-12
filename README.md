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
| Ahmed *[achternaam]* | *[invullen]* | *[invullen]* | `ahmed` |
| Mehmet Kilinc | [@Mehmet-hhs](https://github.com/Mehmet-hhs) | [mehmetkilinc123] | `mehmet` |

**Kaggle-teamnaam:** *[invullen]*

---

## Plan van aanpak

### Taakverdeling

| Onderdeel | Wie | Inhoud |
|---|---|---|
| **Opdracht 1: EDA** | Khaleel | Kaggle-team aanmaken · data inladen (`DataLoader`) · datatypes en basisstatistieken · datakwaliteit (missende waarden, ontbrekende uren, onjuiste waarden) · tijdreekselementen toevoegen · visualisaties van patronen (dag, week, jaar, trend) · relaties tussen de variabelen en `cnt` · data opschonen · belangrijkste bevindingen |
| **Opdracht 2: Time series feature engineering** | Ahmed | Evaluatiemethode (`TimeSeriesSplit`, RMSE) · onderzoek naar trend · seizoenspatronen met Fourier-analyse (periodogram) · cycli en autocorrelatie (ACF/PACF) · overige tijdsfeatures · `FeatureEngineer`-klasse · experiment: invloed van de features · onderbouwde keuze van de features |
| **Opdracht 3: Modelleren** | Mehmet | `ModelTrainer`-klasse (GridSearchCV + cross-validatie) · lineaire regressie met regularisatie (wiskundige uitleg: loss function, Ridge/Lasso) · ensemble-modellen (Random Forest, XGBoost, met wiskundige uitleg) · minstens 5 modellen in totaal · SARIMAX + een tweede tijdreeksmodel · hybride model (residuen) · hyperparameter tuning per model · submission-bestand per model |
| **Bevindingen** | Gezamenlijk | Overzicht van de resultaten · beste model en parameters · Kaggle-scores en verbeteringen |
| **Opdracht 4: Conclusie en aanbevelingen** | Gezamenlijk | Belangrijkste features · aansluiting op de EDA · advies aan CityBike · modeladvies (en andere overwegingen dan alleen de score) |
| **Kaggle** | Gezamenlijk | Alle modellen uploaden · scores bijhouden in de tabel hieronder |
| **Bronnenlijst** | Gezamenlijk | APA-stijl, inclusief GenAI-prompts (zie afspraken) |
| **Eindversie** | Khaleel (samenvoegen), iedereen controleert | Alles samenvoegen in één notebook · notebook draait van begin tot eind · PEP8-controle |

### Afhankelijkheden

De opdrachten bouwen op elkaar voort. Om niet op elkaar te wachten:

1. **Khaleel** zet als eerste `DataLoader` en de opgeschoonde data in `main`. Ahmed en Mehmet
   gebruiken die.
2. **Ahmed** zet zo snel mogelijk een eerste versie van `FeatureEngineer` in `main`. Daarna kan hij
   de features verder verbeteren.
3. **Mehmet** kan meteen beginnen met `ModelTrainer` en de modeluitleg, en test eerst met eenvoudige
   features (uur, weekdag, weer).

### Planning

| Week | Datum | Doel |
|---|---|---|
| 1 | *[invullen]* | Repo en Kaggle-team klaar · `DataLoader` + opgeschoonde data in `main` · eerste Kaggle-submission (simpel model) |
| 2 | *[invullen]* | Opdracht 1 af · eerste versie features in `main` · eerste modellen + submissions |
| 3 | *[invullen]* | Opdracht 2 en 3 af · alle modellen op Kaggle · bevindingen en conclusie samen schrijven |
| 4 | *[invullen]* | Alles samengevoegd · eindcontrole · inleveren (**deadline: [invullen]**) · voorbereiding assessment |

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
hetzelfde `.ipynb`-bestand aanpassen geeft bijna altijd conflicten). Aan het eind voegt Khaleel de
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
3. Khaleel controleert de pull request en klikt op **Merge**.
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
|---|---|---|---|---|---|---|
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |

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

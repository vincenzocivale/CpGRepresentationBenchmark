# EWAS Catalog expansion: stato e regole

Stato al 6 ottobre 2026: **FALLBACK_DESIGN_BUILT**. Entrambi i bulk statici
ufficiali hanno restituito HTTP 503 in nove tentativi ciascuno, con otto retry a
intervalli di dieci secondi. Log: `data/raw/ewas_catalog_expansion/bulk_retry_manifest.json`.
È stato quindi usato il fallback ufficiale [community Zenodo EWAS Catalog](https://zenodo.org/communities/ewas-catalog/).
Tutti i 21 depositi sono inventariati; i risultati dei 19 depositi pertinenti sono
stati scaricati: 29 file, 4.327.425.436 byte. Ogni file è verificato contro l'MD5
pubblicato da Zenodo, ha SHA256 locale e, se compresso, supera la lettura integrale
gzip/controllo CRC ZIP. I due depositi esclusi riguardano proteomica non mirata e
demenza neurodegenerativa; motivazioni nel registro dei depositi.
Nessuna representation è stata caricata o eseguita.

**Limite di copertura:** Zenodo non è una copia completa del Catalog. La selezione
del primario e l'inventario sono completi solo rispetto ai depositi ufficiali
disponibili e inclusi in questo fallback. Non dimostrano che uno studio sia il più
grande nell'intero Catalog; questa verifica resta da fare al ripristino dei bulk.
I depositi non forniscono Catalog StudyID: quel campo è lasciato mancante.
Gli identificativi `zenodo_<record>__<results_filename>` sono chiavi locali
`source_analysis_id`, esplicitamente distinte dagli StudyID del Catalog.

## Risultato del fallback

542 analisi nei metadata; 404 studi delle cinque famiglie con array compatibile e
overlap osservato con l'universo. 351 trait distinti, con un primario ciascuno.

| Famiglia | main_eligible | exploratory | insufficient_support |
|---|---:|---:|---:|
| cancer | 5 | 1 | 6 |
| immune/inflammatory | 11 | 3 | 10 |
| cardiometabolic | 17 | 1 | 280 |
| environmental/behavioral | 3 | 2 | 7 |
| psychiatric | 1 | 0 | 4 |
| Totale | 37 | 7 | 307 |

636 coppie superano il 50% di containment del set più piccolo, su 61.425 coppie
valutate. Esempi: body fat–BMI 95,81% (29.364 loci condivisi), CRP–BMI 71,72%
(23.454), waist-hip ratio–BMI 90,20% (20.177). Questi sono overlap dei positive set,
non stime della sovrapposizione dei partecipanti.

Tutti i positivi con tipo Infinium noto hanno capacità sufficiente per il matching
MAIN 1:1. Le membership positive eliminate per tipo sconosciuto sono 423, sommando
i trait (non 423 loci necessariamente distinti). 188 trait hanno matching parziale
per questo motivo; 163 completo. Nessun tier cambia dopo matching.

Il [report per trait](../data/registry/ewas_catalog_expansion/report.md),
il [registro dei primari](../data/registry/ewas_catalog_expansion/primary_traits.csv),
gli [studi candidati](../data/registry/ewas_catalog_expansion/all_candidate_studies.csv)
e tutte le [coppie segnalate](../data/registry/ewas_catalog_expansion/overlap_flags.csv)
sono versionabili in `data/registry/ewas_catalog_expansion/`.
Spec per famiglie, tier e percorsi dei set con SHA256:
`configs/ewas_v2/catalog_expansion_2026-10-06.json` (design-only, non collegato al runner).
I 351 positive set `.npy` sono in `data/derived/ewas_catalog_expansion/positive_sets/`.

Fonte: [EWAS Catalog, download](https://www.ewascatalog.org/download/).
Il codice ufficiale del Catalog conferma i due URL bulk e precisa che le coordinate
nei results sono hg19: il benchmark deve invece usare il mapping locale delle sonde
a GRCh38, senza confrontare direttamente queste coordinate con il registro.

## Procedura preparata

Script: `scripts/ewas_v2/build_catalog_expansion.py`; fallback:
`download_zenodo_fallback.py` e `prepare_zenodo_fallback.py` nella stessa cartella.

```bash
python scripts/ewas_v2/build_catalog_expansion.py download
python scripts/ewas_v2/build_catalog_expansion.py inventory
# Classificare tutti i trait in trait_categories.csv: le cinque famiglie o out_of_scope.
python scripts/ewas_v2/build_catalog_expansion.py build
```

Per riprodurre il fallback: scaricare i depositi con `download_zenodo_fallback.py`,
normalizzarli con `prepare_zenodo_fallback.py`, poi usare `inventory`/`build` con
`--raw-dir data/raw/ewas_catalog_expansion/zenodo_prepared` e
`--output-dir data/derived/ewas_catalog_expansion`.
La classificazione curata si trova in
`data/registry/ewas_catalog_expansion/zenodo_analysis_metadata.csv`;
`zenodo_records_inventory.csv` riporta tutti i depositi della community e le esclusioni.
Nei pacchetti Zenodo i results sono collegati ai metadata tramite `Results_file`.
Il TSV normalizzato conserva tutti gli hit p<1e-4 e un testimone di copertura per
studio; i conteggi della copertura sono calcolati sul file completo prima di questa
compattazione e conservati nei metadata/manifest.

La classificazione deve essere completata usando i metadata prima di produrre
l'inventario definitivo. Lo script non assegna famiglie tramite corrispondenze
approssimative e non accetta trait lasciati senza classificazione. Ogni esclusione
deve essere documentata nella colonna `classification_note`.

## Regole previste

- Sottosezioni: cancer, immune/inflammatory, cardiometabolic,
  environmental/behavioral, psychiatric.
- Identità del trait: etichetta Catalog, normalizzata solo per maiuscole e spazi.
  Sinonimi clinici ed endpoint diversi non sono aggregati automaticamente.
- Array compatibile: metadata che dichiarano 450K o EPIC/850K, anche misti.
  Compatibilità osservata con l'universo: almeno un locus dei results dello studio
  mappato nel registro. Gli studi senza overlap osservabile sono riportati a parte;
  l'assenza di risultati riportati non dimostra assenza di copertura dell'array.
- Primario per trait: whole blood quando disponibile, poi meta-analysis, poi N
  decrescente, infine StudyID lessicografico. Meta-analysis è identificata dalla
  dicitura esplicita nei metadata/publicazione; più coorti da sole non bastano.
  N mancante/non numerico viene ordinato
  per ultimo e segnalato. Nessuna preferenza usa il numero di hit o un embedding.
- Positive set: **p < 1e-4**, modifica esplicita dell'utente del 6 ottobre 2026.
  La soglia legacy p<1e-7 di `scripts/build_ewas_catalog_sets.py` non è stata modificata.
  Raw = sonde uniche dello studio primario; nell'universo = loci GRCh38 unici dopo
  mapping locale e intersezione con i **408.399** loci di
  `outputs/encode_atlas_v1/loci.parquet`. Le perdite di mapping sono distinte da
  quelle per assenza nell'universo.
- Tier sul numero di positivi **dopo intersezione**: >=200 `main_eligible`,
  100–199 `exploratory`, <100 `insufficient_support`.
- Matching principale: capacità esatta per 1:1 per cromosoma e tipo Infinium I/II,
  senza replacement. Non vengono estratti background. Background escluso dall'unione
  dei nuovi primari e dei 16 set esistenti.
  Positivi con tipo sconosciuto e positivi senza background disponibile sono
  contati separatamente. Il tier dopo matching è riportato separatamente dal tier
  richiesto dopo intersezione. Questa fattibilità riguarda MAIN; S1 non è calcolato.
- Overlap su ogni coppia di primari, anche tra famiglie: intersezione, Jaccard e
  frazione del set più piccolo condivisa. Flag solo se **>50%**, soglia stretta
  richiesta per questa espansione.

## Output previsti

In `data/derived/ewas_catalog_expansion/`: metadata e studi candidati completi,
assegnazioni dei trait, studi esclusi per array, trait senza studio compatibile,
`primary_traits.csv` con StudyID/PMID, tissue, N, array, conteggi, tier e matching,
`positive_sets/*.npy` in ID canonici GRCh38, tutte le coppie di overlap, flag e
`report.md` suddiviso nelle cinque famiglie. Manifest SHA256 degli input e output.

Le categorie includono diagnosi e biomarker continui pertinenti (lipidi,
metaboliti, adiposità, CRP), mantenendo distinte le sottoclassi. Neurologia
degenerativa, età/sesso/ancestry, misure ossee e riproduttive sono fuori scope.
Incidenza/prevalenza, developmental stage e contrasti di esposizione distinti
non vengono aggregati. Il modello è selezionato a parità delle altre preferenze
tramite la chiave sorgente lessicografica: ciò può scegliere un modello basic.
Covariate e analisi sono esposte nel CSV, senza selezione tramite hit o risultati.

I metadata del paper sono usati per colmare lacune identificabili, con provenienza
esplicita. Per esempio: PAH è un'analisi pooled, non una meta-analysis; l'abstract
Zenodo di quel deposito è copiato erroneamente da un altro studio, perciò tissue e
N vengono dal PMID38184627. Per l'ansia materna, il primario ha N6686, non il totale
di progetto7243. Quando presenti, min/max di N per sonda sono riportati; il massimo
è il valore N usato per l'ordinamento. N non disponibili rimangono mancanti.

La copertura dell'intero Catalog e gli StudyID originali restano da riconciliare
quando i bulk tornano disponibili. Il protocollo EWAS-v2 e i suoi set esistenti
non sono stati modificati dall'espansione.

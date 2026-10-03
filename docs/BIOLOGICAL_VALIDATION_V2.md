# Biological validation v2: protocollo ratificato (pre-registrazione)

Stato: **POLITICA DEFINITIVA, RATIFICATA, congelata prima di qualunque metrica di embedding sugli assi v2** (nessun risultato di embedding visto). Data: 2026-10-02.
Fonti dei fatti: `docs/bioval_v2_prep/*_PREP.md` (ultime sezioni), `data/derived/bioval_v2/**/FROZEN*.json` / `MANIFEST*.json` / `marker_provenance_verdict.json` /
`BUILD_CHECK.json` / `assembly_validation.json`, `data/derived/bioval_v2/regulatory_activity/CITATIONS.md`, `src/cpg_repr_benchmark/biological_validation_v2/`,
`data/external/functional/locus_features_v1/` (manifest, `regulatory/track_contract.tsv`), `docs/REGULATORY_REPRESENTATION.md`.
Le decisioni di preparazione dati dei report PREP non sono modificate; le decisioni di analisi ratificate dall'utente sono riportate nelle sezioni 0 e 2. Dove un fatto non e' noto: "non noto".
Matrice machine-readable: `configs/biological_validation_v2/anti_circularity.yaml` (test: `tests/test_bioval_v2_audit_matrix.py`).

## 0. Stato degli endpoint e numeri finali (post-QC / post-matching)

Quattro classi, mutuamente esclusive. **PRIMARY** = un solo endpoint per asse, i 4 endpoint formano la famiglia Holm. **SECONDARY** = riportati, descrittivi, nessun claim.
**EXPLORATORY** = riportati con etichetta esplorativa. **REJECTED AS CIRCULAR** = target derivati dagli input (sezione 3): evidenza legacy, mai endpoint.
Le *sensitivity* sono varianti dichiarate di un target (colonna "Sensitivity"), non una classe a se'.

| Classe | Asse | Endpoint (metrica, direzione: piu' alto e' meglio) | Sensitivity dichiarate |
|---|---|---|---|
| **PRIMARY** | external_methylation_programs | **Profilo WGBS Loyfer**: Spearman(similarita' di profilo di metilazione [Pearson dei beta per gruppo cellulare], coseno di embedding), pooled su strati >1 Mb + inter (peso uguale per strato) | Pearson vs Spearman; `mask_lenient`; per strato di distanza (<1 kb, 1-10 kb, 10-100 kb, 100 kb-1 Mb riportati separatamente); gruppi a campione singolo esclusi; coppie con < 30 gruppi condivisi rimosse |
| **PRIMARY** | 3d_genome | **H1 Micro-C intra-cromosomico 10 kb**: AUROC contatto vs non-contatto appaiato (punteggio = coseno), effect size secondaria = Delta-coseno appaiato | **HFFc6** Micro-C intra 10 kb; per bin di distanza; compartimenti a 50 kb |
| **PRIMARY** | regulatory_activity | **Appartenenza a enhancer FANTOM5**: AUROC (sonda lineare congelata, CV a blocchi) vs controlli appaiati congelati, pool `bg_enh5k` | pool TSS-filtrato `bg_enh5k_tss5k` (3,349 coppie); finestra 500 bp (26,880 coppie); soglie N=1/3/10/20 librerie; matching aggiuntivo su covariate cromatiniche |
| **PRIMARY** | replication_domains | **RT consenso continuo** (`rt_consensus_z`, media degli z di 15 linee UW): Spearman (+ R2) predizione ridge a blocchi cromosomici vs target | RT per singola linea (15); residualizzazione GC / densita' CpG / TSS; linee normali vs tumorali |
| SECONDARY | external_methylation_programs | Marker kNN enrichment (provenienza dei file hg38 **WEAK**, vedi 2.1) | set marker dal paper S4A/S4B lifted (164 U25 + 1,288 U250); solo U25 |
| SECONDARY | 3d_genome | Compartimenti A/B GM12878 Hi-C (solo compartimenti); sonda E1: Spearman(E1, predizione) CV a blocchi, H1 100 kb; Delta-coseno appaiato (effect size) | - |
| SECONDARY | regulatory_activity | Similarita' di profilo di attivita' FANTOM5 (Spearman con coseno), 171,017 coppie | livello campione vs collassato; per categoria |
| EXPLORATORY | 3d_genome | **H1 inter-cromosomico 1 Mb** (drop totale 98.3%, bilancio debole) | - |
| EXPLORATORY | replication_domains | **PMD Decato 2020** (hg19, liftover esplicito a GRCh38; derivato da metilazione): classificazione/sonda per appartenenza a PMD, secondario/esplorativo | - |
| REJECTED (circular) | legacy | CpG island/shore/shelf/open_sea (dense 0-3), gene region (4-7), classi cCRE (8-16), distanza TSS (17), ChromHMM da stessi marchi istonici | evidenza legacy in `bio_validation/`, mai endpoint |

**Numeri finali (da `FROZEN*.json` / `MANIFEST*.json`; sha256 in sezione 7).**

| Asse | Quantita' | Valore |
|---|---|---|
| Universo | CpG autosomici (chr1-22) | 408,399 (solo autosomi, nessun chrX/Y/M, in tutti gli assi) |
| Loyfer | coppie congelate (seed 17, n_shared_groups >= 20, Pearson definito) | **386,268**: <1 kb 37,315; 1-10 kb 38,167; 10-100 kb 38,785; 100 kb-1 Mb 38,819; >1 Mb 38,914; inter 194,268 (sopravvivenza 93.3-97.3% per strato) |
| Loyfer marker | CpG marker unici nei gruppi presenti | **1,257** = U25 157 + U250-only 1,100 (set GitHub hg38); 164 vs 157: 164 U25 in universo includono 7 righe Megakaryocytes (colonna d'atlante senza campioni locali, assente dalla tabella S1) -> 157 nei 39 gruppi presenti. In universo totale: 164 U25 + 1,132 U250 = 1,296 |
| 4DN H1 intra 10 kb | coppie positivo = controllo | **2,689,921** (proiettati 2,911,439; scartati 221,518 = **7.6%**); per fold 567,035 / 665,475 / 456,288 / 398,295 / 602,828 |
| 4DN HFFc6 intra 10 kb | coppie positivo = controllo | **3,724,445** (proiettati 3,919,850; scartati 195,405 = **5.0%**) |
| 4DN H1 inter 1 Mb | coppie dopo `drop` straddle | **9,931** pos + 9,931 controlli; drop totale **98.3%** (538,022 = 89.6% non appaiati + coppie a cavallo di fold); bilancio decile di copertura SMD 2.00 -> 0.30 (debole) |
| 4DN HFFc6 inter | - | non congelato (pool 178 coppie di bin) |
| Compartimenti E1 (100 kb) | CpG con E1 su 408,399 | H1 407,903 (A 342,745 / B 65,158; 496 senza E1); HFFc6 408,223 (A 322,024 / B 86,199; 176 senza); GM12878 408,275 (A 300,171 / B 108,104; 124 senza) |
| FANTOM5 membership | positivi / coppie appaiate / scartati | 12,551 / **11,575** / **976 (7.78%)** |
| FANTOM5 finestra 500 bp | coppie | 26,880 (2,304 scartati, 7.9%) |
| FANTOM5 pool TSS-filtrato | coppie | 3,349 (73.3% scartati; SMD tss_dist -0.117, non < 0.1) |
| FANTOM5 attivita' | coppie finali | **171,017** (100,000 inter + 71,017 intra; 8,490 stesso-enhancer rimosse) |
| RT | CpG con target | **408,345** (54 esclusi per tutte le rappresentazioni, nessuna imputazione) |
| PMD | CpG in >= 1 PMD | 328,894 su 408,399 (tutti assessable) |

Bilancio FANTOM5 (membership primario) dopo matching: |SMD| numerico <= 0.006, probe_I 0.053, probe_unknown -0.013, contesto 0 (esatto): tutti < 0.1.

## Politica di inferenza ratificata (valida per i 4 endpoint primari)

Decisioni ratificate dall'utente: (1) correzione **Holm** sui 4 endpoint primari; (2) CV **deterministica a 5 fold a blocchi cromosomici, seed 17** (`chromosome_blocked_folds(n_folds=5, seed=17)`; mappa cromosoma->fold congelata nei FROZEN);
(3) **similarita' coseno** come punteggio di coppia in tutti gli assi a coppie (Loyfer e 4DN); (4) **AUROC contatto vs non-contatto appaiato** primario per 4DN, **Delta-coseno appaiato** come effect size secondaria; (5) coppie inter-cromosomiche non appaiate scartate **prima** dell'analisi, con drop rate riportato; (6) **nessuna intersezione per rappresentazione** (una rappresentazione che non copre l'universo fallisce esplicitamente).

Regola esatta:
1. Contrasto: per ogni confronto pre-dichiarato (braccio A vs comparatore B, lista congelata) e per ciascun endpoint e: Delta_e = m_A,e - m_B,e (metrica primaria di e, stessi item per A e B).
2. Incertezza: bootstrap per **blocco = cromosoma** (22 blocchi), 1000 repliche, seed fisso (17); a ogni replica si ricampionano gli stessi cromosomi per i due bracci (confronto **appaiato**). Blocco di una coppia = cromosoma di `cpg_i` (le coppie inter sono quindi approssimate: caveat). CI al 95% percentile per ogni m e per ogni Delta.
3. p-value: bilaterale bootstrap, p = min(1, 2 * min(P*(Delta* <= 0), P*(Delta* >= 0))) con correzione (1+k)/(1+B). Con 22 blocchi e B = 1000 la CI e' grossolana (dichiarato).
4. Molteplicita': per ciascun contrasto, i p-value dei 4 endpoint primari formano la famiglia; **Holm a alpha = 0.05 family-wise**. Un asse "supporta" A se il p aggiustato < 0.05 e Delta ha il segno atteso (A > B, direzione "piu' alto e' meglio"); altrimenti nessun claim.
5. Secondari, sensitivity, esplorativi: descrittivi, senza claim; eventuali q-value BH solo informativi. Si riportano **tutti** gli endpoint pre-registrati, anche i nulli.
6. Sonde (ridge/lineari) sui bracci congelati: addestrate solo sui cromosomi di train di ciascun fold; la regolarizzazione e' scelta con CV interna ai soli cromosomi di train; mai su cromosomi di test.

## 1. Principi e ambito

1. **Indipendenza.** Un target di validazione e' ammesso come evidenza principale solo se non e' derivato dagli input della
   rappresentazione (sezione 3). Input del baseline `functional_annotations_pca`: 4,165 tracce ENCODE binarie di overlap di picchi
   (1,959 Histone ChIP-seq, 1,673 TF ChIP-seq di cui 201 CTCF, 533 DNase-seq) piu' 23 colonne dense (island/shore/shelf/open_sea,
   gene_region, cCRE Registry V4, log10 TSS distance da GENCODE v50, 5 breadth derivate dalle tracce). I candidati
   `regulatory_*` usano solo le 4,165 tracce (nessuna colonna dense). Nessun input contiene metilazione, Hi-C, FANTOM/CAGE o replication timing.
2. **Nessuna model selection.** Questi assi non selezionano rappresentazioni, iperparametri, dimensioni o feature set. La selezione
   resta nei protocolli esistenti (`docs/REGULATORY_SELECTION_PROTOCOL.md`); bioval v2 e' confermativa e descrittiva.
3. **Coordinate canoniche**: GRCh38, 1-based (citosina del CG), universo fisso di **408,399 CpG autosomici (chr1-22)**,
   identico a `load_benchmark_universe()` (verificato nei report Loyfer e RT). Il 100% delle posizioni e' una C di CG in hg38.fa.
   Fonti non GRCh38 sono convertite una volta (`coordinates.harmonize`) e il layer di conversione e' registrato nel MANIFEST.
4. **Universo fisso, niente intersezioni per rappresentazione.** Una rappresentazione che non copre l'universo fallisce
   esplicitamente (`require_full_universe_coverage`). I CpG senza valore del target (es. 54 senza RT, NaN Loyfer) sono esclusi
   in modo uguale per tutti i bracci, per regola dipendente solo dal target.
5. **Il matching non usa mai embedding.** `matching.match_controls` accetta solo DataFrame di covariate e rifiuta colonne con
   nomi tipo emb/embedding/pca/latent/z_N. Seed e tie-break sono hash di (id, seed).
6. **Split a blocchi cromosomici** (`splits.chromosome_blocked_folds`, `leave_chromosome_out`, `pair_*_mask`, `assert_no_leak`). Nessun
   cromosoma compare in train e test. Le CI sono bootstrap per blocco (cromosoma) (`metrics.bootstrap_ci_by_block`).
7. **Il legacy `bio_validation` resta evidenza legacy** (CpG island, gene region, probe cCRE/known sets). Non e' rimosso
   ne' modificato. Non entra negli endpoint primari perche' circolare (sezione 3).
8. Ogni valutazione usa la rappresentazione nel track `native_frozen` come estratta (nessuna ri-ottimizzazione su metilazione).
   Le sonde lineari (ridge) sono misure su embedding congelato, non training della rappresentazione.

## 2. Assi di validazione

Convenzioni comuni: vedi sezione 0 (classi e politica di inferenza ratificata). Endpoint primario: uno per asse, coerente con `metrics.PRIMARY_METRICS`.

### 2.1 external_methylation_programs (Loyfer 2023 WGBS)

- **Ipotesi.** Loci con profili di metilazione simili tra tipi cellulari primari hanno rappresentazioni simili; i marker ipometilati
  di un tipo cellulare formano vicinati omogenei.
- **Dati.** Atlante Loyfer et al. 2023, 253 file `hg38.pat.gz` (93 GB, solo pat; i beta sono derivati). 207 campioni atlante (GSE186458)
  + 46 campioni cfDNA/WBC (CNVS-NORM) esclusi dalle matrici di tipo cellulare. 39 gruppi atlante presenti con 205 campioni (tabella S1; Megakaryocytes assente
  in locale). Marker: UXM_deconv `Atlas.U25.l4.hg38.full.tsv` e `Markers.U250.hg38.tsv` (tutti ipometilati, direzione U).
- **Build.** GRCh38 nativo (nessun liftover). CpG.bed rigenerato da hg38.fa.gz (29,401,795 CpG), validato al 100% su 9,520 + 979 blocchi marker pubblicati.
- **Unita'.** Coppie di CpG (profili) e loci marker.
- **Definizione.** Profilo di un CpG = beta per gruppo (media non pesata dei donatori con cov >= 10); `mask_primary` = >= 2 donatori
  (1 se il gruppo ha un solo campione). Similarita' di profilo = Pearson tra gruppi con `min_shared = 20` (sopravvive 96.5-97.6% delle coppie; mediana 39 gruppi condivisi).
  Positivi marker (set GitHub hg38, `Atlas.U25.l4.hg38.full.tsv` + `Markers.U250.hg38.tsv`, commit 8d0bb45) = **1,257 CpG unici nei 39 gruppi presenti** (U25 157 + U250-only 1,100); 1,296 nell'universo includendo i marker Megakaryocytes / gruppi assenti. Riconciliazione 164 vs 157: i 164 CpG U25 in universo comprendono 7 righe Megakaryocytes (nessun campione locale, assente dalla tabella S1) -> 157 nei gruppi presenti (`marker_reconciliation.json`).
  **Provenienza: verdetto WEAK** (`marker_provenance_verdict.json`): le tabelle del paper (S4A/S4B) sono solo hg19 e i file GitHub hg19 le riproducono esattamente (953/953 U25, 9,288/9,288 U250), ma i file hg38 NON sono un liftover delle tabelle (solo 67.1% dei blocchi U25 e 74.0% degli U250 si sovrappongono a un blocco del paper lifted dello stesso tipo): non verificabili come tabelle pubblicate. Per questo il marker-kNN e' **SECONDARY**. Set alternativo (STRONG per identita' del marker, lift nostro): paper S4A/S4B lifted hg19->hg38, 164 U25 + 1,288 U250 CpG unici in universo, usato come sensitivity.
- **Matching / stratificazione.** Coppie campionate con `sample_pairs`: intra-cromosomiche in strati di distanza (<1 kb, 1-10 kb, 10-100 kb, 100 kb-1 Mb, >1 Mb) e inter-cromosomiche
  separate. La similarita' di profilo decade con la distanza (Pearson mediano 0.29/0.11/0.08/0.06/0.05; inter 0.05), quindi **>1 Mb e inter-cromosomiche sono riportate
  separatamente e sono il test principale non locale**; gli strati brevi misurano soprattutto prossimita' genomica. Covariate di matching per i marker: chrom, densita' CpG, GC
  (`build_covariates`).
- **Split.** Blocchi cromosomici (similarita' coseno su embedding congelati: nessun training); per le coppie inter-cromosomiche e' ratificata la policy `drop` per i fold primari (le coppie inter non hanno un blocco unico).
- **Metrica primaria (PRIMARY, ratificata).** Spearman(similarita' di profilo WGBS, coseno di embedding) sulle 386,268 coppie congelate, **pooled sugli strati >1 Mb + inter-cromosomiche con peso uguale per strato**; gli strati brevi (<1 kb ... 100 kb-1 Mb) sono riportati separatamente perche' misurano soprattutto prossimita' genomica;
  direzione: piu' alto e' meglio. Inferenza: sezione 0. **Secondaria (SECONDARY):** marker kNN enrichment (`metrics.knn_enrichment`, k fissato al freeze delle liste braccio/confronto, p-value per permutazione dentro cromosoma), declassato a secondario per provenienza WEAK dei marker.
- **Limiti di potenza.** 1,296 marker nell'universo (1.72%), ~4 CpG per tipo per l'atlante U25 e ~24 con U250: test marker a bassa potenza, molti tipi quasi vuoti; U25 come sensitivity. Il marker kNN e' secondario e non entra nella famiglia Holm.
- **Sensitivity (PRIMARY).** Spearman vs Pearson; `mask_lenient`; solo U25; per strato di distanza; esclusione di gruppi a campione singolo; rimozione coppie con < 30 gruppi condivisi.
- **Circolarita'/confondenti.** Nessun input contiene metilazione. Confondenti: prossimita' genomica, densita' CpG/GC, campionamento sbilanciato verso sangue/epiteli (Blood-T n=22, Endothel n=19, vari gruppi n=1),
  donatori non indipendenti tra etichette fini, marker solo ipometilati. Provenienza campione->gruppo: **205 campioni da tabella S1 pubblicata (Nature 2023) + 48 decisioni manuali** (46 cfDNA/WBC esclusi, 2 Heart-Cardiomyocyte GSM5652214/15 non in S1 esclusi), 168 donatori (`manual_decisions.csv`, `sample_provenance.csv`); la mappatura v1 per nome file e' superata.

### 2.2 3d_genome (4DN)

- **Ipotesi.** Loci in contatto 3D (stesso compartimento A/B, o contatto puntuale) hanno rappresentazioni piu' simili di loci distanza-appaiati senza contatto.
- **Dati.** 4DN, mcool processati (S3 open-data). **PRIMARY: H1-hESC Micro-C 4DNFI9GMP2J8** (set 4DNES21D8SP8). **SENSITIVITY: HFFc6 Micro-C 4DNFI9FVHJZQ.** **SECONDARY, solo compartimenti: GM12878 in situ Hi-C 4DNFIXP4QG5B** (nessuna tabella di contatti).
  md5 del file mcool completo non verificato (scaricate per range solo le risoluzioni necessarie; sha256 dei cooler estratti nel MANIFEST).
- **Build.** GRCh38 **inferita e validata** (`assembly_validation.json`): attributo `genome-assembly` dei cooler "unknown", ma 23/23 lunghezze cromosomiche = hg38.fa in tutti e tre i sistemi e 0/23 = hg19, metadati del portale 4DN `genome_assembly` = GRCh38, 408,399/408,399 posizioni dell'universo sono la C di un CG in hg38.fa, accoppiamento E1-GC coerente (mediana Spearman 0.78 / 0.52 / 0.63 H1/HFFc6/GM12878). Bin = floor((pos-1)/binsize).
- **Unita'.** Locus (compartimento) e coppia di CpG (contatto).
- **Definizioni.** E1: cooltools `eigs_cis` per cromosoma intero, fasatura con GC hg38, A se E1>0 (100 kb primario, 50 kb secondario; flag Spearman(E1,GC)<0.2: nessun cromosoma in H1/HFFc6/GM12878).
  Contatto intra (10 kb): distanza 20 kb-2 Mb, raw >= 8, obs/exp >= 2. Non-contatto intra: pool con bin validi, copertura cis >= 25-esimo percentile, almeno un CpG per bin, pixel assente o obs/exp <= 0.5, stesso cromosoma e bin di distanza (+-1). **Regola di eleggibilita' aggiuntiva (decisione di design pre-freeze, presa prima di qualunque embedding):** la stessa eleggibilita' (bin validi, copertura >= 25-esimo percentile del cromosoma, almeno un CpG) e' applicata anche ai positivi, per evitare un confronto contatti-vs-pool con criteri asimmetrici; rimuove 26,596 coppie di bin H1 su 325,733 (-> 299,137).
  Contatto inter (1 Mb): raw >= 50, obs/exp bilanciato >= 2 sulla media della coppia di cromosomi; non-contatto: raw 0 o obs/exp <= 0.5. **Intra e inter rigorosamente separati** (file e analisi diverse). Proiezione CpG: tutti i CpG bin i x bin j, sottocampione deterministico (seed 17), max K per coppia di bin (K=20 in stima).
- **Matching (congelato).** Strati esatti = cromosoma x 80 bin di distanza log-spaziati (1e4-2.02e6 bp) x quintile della copertura minima di coppia x terzile di GC medio di coppia x terzile di distanza TSS media di coppia (covariate da `covariates/universe_covariates.parquet`, presenti al freeze); 1:1 senza reinserimento, accoppiamento per rango di distanza, seed 17, K=20 CpG per coppia di bin. Eleggibilita' identica sui due lati. Inter: strati = coppia di cromosomi x 3 gruppi di decile di copertura trans. Mai embedding. Bilancio H1: SMD distanza -0.51 -> -0.0002, log10 copertura minima 0.25 -> 0.055, GC 0.078 -> 0.016, TSS 0.075 -> 0.017; densita' CpG (non uno strato) 0.100 -> 0.100 (hg38) e 0.008 -> 0.048 (universo); differenza mediana di distanza in coppia 2.1 kb.
- **Split.** Blocchi cromosomici (5 fold, seed 17); coppie intra = fold del cromosoma; inter con policy ratificata `drop` (`pair_split_by_chrom`, coppie a cavallo di fold con fold = -1, escluse). CpG della stessa coppia di bin non indipendenti: il bootstrap resta a blocchi cromosomici.
- **Metrica primaria (PRIMARY, ratificata).** AUROC (contatto vs non-contatto distanza-appaiato) **intra-cromosomico, H1 Micro-C, 10 kb**, **punteggio di coppia = similarita' coseno**; direzione: piu' alto e' meglio; inferenza: sezione 0. **Effect size secondaria:** Delta-coseno appaiato (media per coppia appaiata di cos(contatto) - cos(controllo)).
  **SENSITIVITY:** HFFc6 Micro-C intra 10 kb (2.0 vs 3.7 M coppie; profondita' diversa). **SECONDARY:** compartimenti GM12878 (solo compartimenti); sonda A/B su E1 (Spearman autovettore E1 vs predizione, sonda lineare CV a blocchi cromosomici, 100 kb, H1); AUROC stratificato per distanza (`stratified_by_distance`). **EXPLORATORY:** H1 inter 1 Mb (9,931 coppie dopo drop totale 98.3% = 89.6% non appaiati + coppie a cavallo di fold; bilancio debole, SMD 0.30); **inter HFFc6 non utilizzabile** a queste soglie (pool 178 coppie di bin). Nessuna di queste entra nella famiglia Holm.
- **Numeri.** H1 intra: 2,348,800 coppie di bin in contatto, 325,733 con CpG su entrambi i lati, 299,137 dopo la regola di eleggibilita' aggiuntiva (-26,596); pool 946,594 coppie di bin vs 296,796 coppie di bin di contatto (K=20 ciascuno: i controlli sono spazialmente piu' diversi); frozen: 2,689,921 coppie (sezione 0). H1 inter: 33,167 contatti (30,019 con CpG), pool 7,074 (solo 8.7% delle 253 coppie di cromosomi con pool >= 10x i contatti).
  Compartimenti 100 kb: vedi sezione 0.
- **Sensitivity.** 50 kb; per bin di distanza; K per coppia di bin; HFFc6; confronto con baseline di sola distanza/covariate (dichiarato).
- **Circolarita'/confondenti.** Nessuna traccia Hi-C/Micro-C/compartimenti/TAD/loop tra gli input. **CTCF (201), RAD21 (14), SMC3 (6) ChIP-seq sono input e legati meccanisticamente agli anchor dei loop: le label di contatto sono parzialmente correlate (non corrette).**
  I compartimenti sono biologicamente vicini a segnale istone/DNase; il segno di E1 e' orientato dal GC (E1 non indipendente dalla composizione di sequenza; HFFc6 accoppiamento E1-GC piu' debole, mediana 0.52 vs 0.78 in H1). Linee cellulari singole, nessun Hi-C tessuto-matched;
  profondita' diversa (3.05 M vs 2.35 M contatti) = batch effect; bias MNase (Micro-C) e DpnII (GM12878).

### 2.3 regulatory_activity (FANTOM5 enhancer)

- **Ipotesi.** CpG dentro enhancer trascritti (CAGE bidirezionale) hanno rappresentazioni distinguibili da CpG di controllo con TSS-distanza, densita' CpG e GC simili; CpG in enhancer con profili di attivita' tessuto-specifica simili hanno rappresentazioni simili.
- **Dati.** FANTOM5 hg38 reprocessed enhancers `F5.hg38.enhancers.bed.gz` (Zenodo 10.5281/zenodo.556775), 63,285 loci, matrice TPM RLE 63,285 x 1,829 librerie; attivita' = log2(TPM+1), espresso TPM >= 1. Insieme "permissive"; il set robusto Andersson 2014 (~38k) non e' disponibile in hg38 e non e' stato convertito.
- **Build.** GRCh38 nativo. Intervallo BED 0-based [start,end), citosina = pos-1.
- **Unita'.** Locus CpG.
- **Positivi.** CpG dentro un intervallo enhancer: 12,551 (3.07%; 8,245 enhancer). **Per il profilo** serve enhancer espresso in >= 5 librerie: 11,490 CpG (7,394 enhancer). **Negativi (pool).** CpG a >= 5 kb da qualunque enhancer FANTOM5 (257,326) e, in un pool piu' stretto, anche a >= 5 kb da TSS GENCODE v50 (`bg_enh5k_tss5k`, 59,315). **Pool principale ratificato: `bg_enh5k` (257,326).** *Deviazione dichiarata dalla proposta precedente (`bg_enh5k_tss5k`):* solo il 22% dei positivi dista >= 5 kb da un TSS (mediana 969 bp), per cui l'appaiamento con il pool TSS-filtrato conserva solo 3,349 coppie (73.3% scartati) con SMD residuo di tss_dist -0.117 (> 0.1): un sottoinsieme selezionato e non rappresentativo. Con `bg_enh5k` lo squilibrio di tss_dist e' rimosso dal matching. La scelta e' stata fatta dalle sole covariate, prima di qualunque embedding. Il disegno TSS-filtrato resta congelato come **sensitivity**.
- **Matching (congelato).** Esatto: cromosoma x contesto genomico (island/shore/shelf/open_sea). Numerico, caliper = 0.25 x SD sull'universo: densita' CpG (1.037), densita' CpG hg38 (7.54), GC (0.0245), log10 distanza TSS (0.258); vicino piu' prossimo su distanza standardizzata sul pool, 1:1 senza reinserimento, seed 17. Distanza dall'enhancer non appaiata (controlli a >= 5 kb per costruzione). `probe_type` NON appaiato (6,429 CpG 'unknown' tenuti come livello a parte, nessun CpG scartato per questo; bilancio solo riportato). Nessun covariato da embedding. Risultato: 11,575 coppie, 976 positivi scartati (7.78%) prima dell'analisi, identici per ogni rappresentazione (lista in `..._dropped_positives.parquet`).
- **Split.** CV a 5 fold a blocchi cromosomici (seed 17); bootstrap per cromosoma.
- **Metrica primaria (PRIMARY, ratificata).** AUROC di appartenenza a enhancer FANTOM5 vs controlli appaiati congelati (pool `bg_enh5k`; sonda lineare congelata, CV a blocchi cromosomici); direzione: piu' alto e' meglio; inferenza: sezione 0. **Secondaria (SECONDARY):** Spearman tra similarita' di profilo di attivita' (Pearson sui profili collassati log2(TPM+1), 638 gruppi) e coseno di embedding, 171,017 coppie congelate.
- **Sensitivity.** Disegno con pool TSS-filtrato `bg_enh5k_tss5k` (3,349 coppie); finestra 500 bp dal punto medio (29,184 CpG -> 26,880 coppie; la finestra 1 kb, 52,953 CpG, non e' congelata); soglie di espressione N = 1/3/10/20 librerie (N=20: 7,708 CpG); matching aggiuntivo su covariate cromatiniche (a posteriori, dichiarato); livello campione vs collassato; per categoria (tessuti/cellule primarie/linee).
- **Circolarita'/confondenti.** FANTOM5 e' un assay indipendente (cattura 5' RNA), nessuna traccia FANTOM/CAGE/eRNA negli input. **Indiretto:** H3K27ac/H3K4me1/DNase/TF ChIP e cCRE pELS/dELS (nel baseline) marcano gli stessi enhancer attivi e GENCODE TSS e' input e covariata: il segnale e' parzialmente mediato da evidenza ENCODE condivisa. Non e' indipendenza piena (classe partially_related, riportata, non corretta). Licenza dati: **CC BY 4.0** (verificata dalla pagina FANTOM e dal record Zenodo; citazioni complete in `data/derived/bioval_v2/regulatory_activity/CITATIONS.md`: Andersson 2014 Nature 507:455; FANTOM Consortium 2014 Nature 507:462; Dalby/Rennie/Andersson, Zenodo 10.5281/zenodo.556775; Lizio 2015 Genome Biol 16:22). Piccola n di campioni per gruppo (mediana 1), 90.8% di zeri nella matrice, gruppi non equivalenti a tessuti indipendenti, 27 librerie senza categoria.

### 2.4 replication_domains

- **Ipotesi.** La rappresentazione codifica il timing di replicazione (eucromatina precoce vs eterocromatina tardiva) in modo continuo lungo il genoma.
- **Dati.** UW Repli-seq WaveSignal (Hansen 2010; ENCODE 2012), **15 linee** (BG02ES, BJ, GM06990, GM12801, GM12812, GM12813, GM12878, HeLa-S3, HepG2, HUVEC, IMR90, K562, MCF7, NHEK, SK-N-SH), bin 1 kb, solo Rep1. Segnale piu' alto = replicazione piu' precoce (non e' un log-ratio).
  Non usati: Koren 2012, 4DN Repli-seq, ENCODE Repli-seq GRCh38 (stesso consorzio degli input), Repli-Atlas. Nessuna sorgente di replica indipendente recuperata.
- **Build.** hg19 -> GRCh38 con liftover (`hg19ToHg38.over.chain.gz`): 3,035,165 -> 2,820,279 bin (200,661 non mappati, 13,209 scartati dalla regola intervallo). 408,345/408,399 CpG con RT in tutte le 15 linee; 54 senza.
- **Unita'.** Locus CpG (osservazioni non indipendenti: 122,492 bin da 1 kb per 408,345 CpG).
- **Definizione.** Valore del bin che contiene il CpG (nessuna interpolazione). **Consenso** `rt_consensus_z` = media dei z-score per linea (>= 3 linee; qui tutte-o-nessuna: 15 o 0 linee per CpG), parametri di z-score in MANIFEST. Target continuo, non positivi/negativi. I 54 CpG senza RT sono esclusi per tutte le rappresentazioni.
- **Matching.** Non applicabile al target continuo; confondenti gestiti come sensitivity (sotto).
- **Split.** Blocchi cromosomici (autocorrelazione: e-folding ~2-2.5 Mb, ~600 blocchi indipendenti effettivi, ordine 500-1,000, stima grezza); blocchi piu' piccoli solo se >= ~5 Mb.
- **Metrica primaria (PRIMARY, ratificata).** Spearman (ed R2 riportato) tra RT consenso continuo e predizione di sonda **ridge a blocchi cromosomici** (5 fold, seed 17, mappa in `FROZEN_ENDPOINT.json`); direzione: piu' alto e' meglio; inferenza: sezione 0. Regolarizzazione scelta con CV interna ai soli cromosomi di train.
- **Sensitivity (SENSITIVITY, ratificata).** Per singola linea RT (15 colonne `rt_<LINE>`, non mediate nel primario); correlazione parziale/residualizzazione per GC, densita' CpG/stato isola, densita' genica/distanza TSS (pre-dichiarata nel PREP); solo linee normali vs tumorali.
- **PMD (SECONDARY/EXPLORATORY, separato).** Decato 2020 (MethPipe `pmd`, 157 campioni: 110 non-TCGA, 39 tumori TCGA, 8 sani TCGA matched). **Build hg19 verificata** (`BUILD_CHECK.json`): fine massima del PMD <= lunghezza hg19 su 22/22 autosomi e > lunghezza hg38 su 14/22; l'inizio dell'intervallo e' la C di un CG in hg38.fa nel 96.3% dei casi dopo liftover hg19->GRCh38 contro 2.2% leggendo le coordinate come GRCh38 (atteso casuale 1.6%) -> **liftover esplicito per campione** a GRCh38, che perde il **2.45%** degli intervalli (4,749 non mappati + 4,995 scartati su 397,664). 328,894/408,399 CpG in un PMD in >= 1 campione; tutti assessable (solo autosomi; un'assenza di campione non e' "no PMD"). **Derivati da metilazione WGBS**: confusi con compito di predizione di metilazione e con l'asse Loyfer; indipendenti dagli input ENCODE. Per questo PMD e' secondario/esplorativo e non entra nella famiglia Holm.
- **Circolarita'/confondenti.** Nessuna traccia Repli-seq/lamina/LAD tra gli input. RT e' fortemente correlato con GC e densita' genica; i CpG da array sono sbilanciati verso RT precoce (consenso z medio 0.76, sd 0.62). Linee tumorali vs normali, residuo: il programma ENCODE e' lo stesso consorzio ma assay/laboratorio diversi.

## 3. Audit di anti-circolarita'

Codici: **I** indipendente; **P** parzialmente correlato (assay biologicamente/meccanisticamente accoppiato; riportato, non corretto); **C** circolare per ritenzione della sorgente (target derivato dall'input).
Classe di circolarita' = cella peggiore (C -> circular; P -> partially_related; altrimenti independent). Fonte machine-readable: `anti_circularity.yaml`.
Input: istone ChIP (1,959), TF ChIP escluso CTCF/cohesin, CTCF/RAD21/SMC3 ChIP (201/14/6), DNase (533), cCRE V4, GENCODE v50 (gene_region, TSS), CpG island (le ultime tre solo in `functional_annotations_pca`),
SEQ = bracci comparatori sequence/foundation-model (input: sequenza di riferimento). I confondenti (GC, densita' CpG/genica) NON sono celle: vanno in sensitivity.

| Target | Istone | TF | CTCF/cohesin | DNase | cCRE | GENCODE | Island | SEQ | Classe | Classe |
|---|---|---|---|---|---|---|---|---|---|---|
| Loyfer profile similarity | I | I | I | I | I | I | I | I | independent | PRIMARY |
| Loyfer marker kNN | I | I | I | I | I | I | I | I | independent | SECONDARY |
| 4DN compartimenti A/B | P | I | I | P | I | I | I | P | partially_related | SECONDARY |
| 4DN contatti intra | I | I | P | I | I | I | I | I | partially_related | PRIMARY |
| 4DN contatti inter | I | I | I | I | I | I | I | I | independent | EXPLORATORY |
| FANTOM5 membership | P | P | I | P | P | P | I | I | partially_related | PRIMARY |
| FANTOM5 activity similarity | P | P | I | P | P | I | I | I | partially_related | SECONDARY |
| Replication timing | I | I | I | I | I | I | I | I | independent | PRIMARY |
| PMD Decato (methylation-derived) | I | I | I | I | I | I | I | I | independent dagli input, derivato da metilazione | EXPLORATORY |
| CpG island/shore/shelf | I | I | I | I | I | I | **C** | P | circular | REJECTED (circular) |
| Gene region | I | I | I | I | I | **C** | I | I | circular | REJECTED (circular) |
| cCRE class | **C** | P | **C** | **C** | **C** | P | I | I | circular | REJECTED (circular) |
| TSS distance | P | I | I | P | P | **C** | P | I | circular | REJECTED (circular) |
| ChromHMM (stessi istoni) | **C** | I | I | P | P | I | I | I | circular | REJECTED (circular) |

Giustificazioni per cella non ovvia:
- **CpG island, gene region, cCRE, TSS**: sono letteralmente colonne dense 0-3, 4-7, 8-16, 17 del baseline: il target e' un re-encoding dell'input (C). **REJECTED AS CIRCULAR: non sono validazione principale**; restano come evidenza legacy
  (`src/cpg_repr_benchmark/bio_validation/`) e come controlli. Per i bracci `regulatory_*` (solo tracce) non sono input, ma restano esclusi dai primari per mantenere un unico protocollo su tutti i bracci. Le island sono definite dalla composizione di sequenza: P per SEQ.
- **cCRE**: le classi V4 sono definite da segnale ENCODE DNase, H3K4me3, H3K27ac e CTCF (conoscenza esterna al repo, **non verificata localmente**): per questo C verso istone/DNase/CTCF. Il manifest registra solo versione e hash del BED.
- **TSS distance**: C per GENCODE; P per H3K4me3/DNase, PLS cCRE e island (co-localizzano con i TSS).
- **ChromHMM**: segmentazione degli stessi marchi istonici: C. Nel codice legacy non c'e' alcun target ChromHMM (grep su `src/`: solo menzioni in docs); e' elencato come candidato escluso.
- **FANTOM5 CAGE**: assay indipendente; P perche' H3K27ac/H3K4me1/DNase/TF e cCRE enhancer-like marcano gli stessi enhancer, e GENCODE TSS e' covariata di matching (membership). Nessun target FANTOM usa cCRE.
- **Loyfer WGBS**: metilazione su cellule primarie, nessun input contiene metilazione. L'associazione cromatina-metilazione e' l'ipotesi, non ritenzione della sorgente. SEQ: I (la sequenza non codifica il profilo tessuto-specifico), ma prossimita' e CpG density sono confondenti gestiti da stratificazione.
- **3D**: assi indipendenti nei dati; P per CTCF/RAD21/SMC3 sui contatti puntuali (anchor di loop), P per istone/DNase sui compartimenti (stato cromatinico) e per SEQ sui compartimenti (E1 orientato da GC). Contatti inter a 1 Mb: nessuna relazione di anchor (esplorativi per bilancio debole, non per circolarita').
- **Replication timing**: nessuna traccia RT negli input (ENCODE ha 104 Repli-seq rilasciati, nessuno nel catalogo). GC e densita' genica sono confondenti, non ritenzione.
- **PMD**: indipendente dagli input ENCODE; dipendente dalla metilazione (confuso con l'asse Loyfer e col compito masking; build hg19 verificata, liftover esplicito); inoltre PMD correla con GC/CpG density e RT (riportato come confondente).
- **Comparatori SEQ**: nessun accesso agli input ENCODE; l'unica relazione e' la sequenza da cui derivano GC/island/CpG density/E1-sign.

## 4. Layer di coordinate e audit di overlap

| Asse | Sorgente | Build nativa | Conversione | Copertura dell'universo (408,399) |
|---|---|---|---|---|
| Loyfer | WGBS pat hg38, CpG.bed rigenerato | GRCh38 | nessuna | >= 20 gruppi: 98.6%; >= 1 gruppo (lenient) 99.88%; in tutti i 39 gruppi (primary) 89.7%; 100% C-di-CG |
| Loyfer marker | UXM_deconv U25/U250 hg38 | GRCh38 | nessuna | 75,158 CpG marker totali; **1,296 nell'universo** (1.72%); 1,257 unici nei gruppi presenti |
| 4DN H1 | Micro-C 4DNFI9GMP2J8 | GRCh38 (attributo "unknown"; inferita e validata) | nessuna | 99.88% CpG con E1 (100 kb); 99.90% in bin validi 10 kb |
| 4DN HFFc6 | Micro-C 4DNFI9FVHJZQ | idem | nessuna | 99.96% CpG con E1 (100 kb); 99.92% in bin validi |
| 4DN GM12878 | Hi-C 4DNFIXP4QG5B | idem | nessuna | 99.97% CpG con E1 (100 kb) |
| FANTOM5 | enhancer hg38 reprocessed | GRCh38 | nessuna | 12,551 CpG in enhancer (3.07%); 29,184 (7.15%) a <=500 bp dal punto medio; 52,953 (12.97%) a <=1 kb |
| FANTOM5 pool | >= 5 kb da enhancer; e da TSS | GRCh38 | nessuna | 257,326 (63.0%); 59,315 (14.5%) |
| RT | UW Repli-seq 15 linee | hg19 | liftover bin (6.6% non mappati + 0.4% scartati) | 408,345 (99.987%) in tutte le 15 linee; 54 senza |
| PMD | Decato 2020 | hg19 (**verificata**, `BUILD_CHECK.json`) | liftover esplicito per campione (2.45% intervalli persi) | 408,399 assessable; 328,894 in PMD in >= 1 campione |

L'universo e' solo autosomico (chr1-22; nessun chrX/Y/M), identico in tutti i report. Gli audit usano `coordinates.overlap_audit` dove applicabile, altrimenti audit per intervallo/bin.

## 5. Dimensioni finali e stato di prontezza

| Asse / endpoint | Dimensione finale (FROZEN) | Classe | Stato |
|---|---|---|---|
| Loyfer: profilo WGBS | 386,268 coppie congelate (194,268 inter; 38,914 >1 Mb) | PRIMARY | **congelato** |
| Loyfer: marker | 1,257 CpG nei gruppi presenti (157 U25 + 1,100 U250-only); 8-64 per tipo (U250) | SECONDARY | **congelato, provenienza WEAK**, bassa potenza |
| 4DN H1 intra (primario) | 2,689,921 coppie pos = ctrl (7.6% scartati) | PRIMARY | **congelato** |
| 4DN HFFc6 intra | 3,724,445 coppie (5.0% scartati) | SENSITIVITY | **congelato** |
| 4DN compartimenti H1 / HFFc6 / GM12878 | 407,903 / 408,223 / 408,275 CpG con E1 | SECONDARY (H1 sonda E1; GM12878) | **congelato** (E1 orientato dal GC) |
| 4DN inter H1 | 9,931 coppie (98.3% drop totale) | EXPLORATORY | **congelato, bilancio debole** |
| 4DN inter HFFc6 | pool 178 | - | **non utilizzabile** |
| FANTOM5 membership | 11,575 coppie (976 scartati, 7.78%); pool `bg_enh5k` | PRIMARY | **congelato** (mediazione ENCODE indiretta) |
| FANTOM5 sensitivity | win500: 26,880 coppie; pool TSS: 3,349 coppie (73.3% scartati) | SENSITIVITY | **congelato** |
| FANTOM5 attivita' | 171,017 coppie | SECONDARY | **congelato** |
| Replication timing | 408,345 CpG; ~600 blocchi efficaci | PRIMARY | **congelato** (hg19 liftover; Rep1) |
| PMD Decato | 328,894 CpG in >= 1 PMD su 408,399 | EXPLORATORY | **pronto** (derivato da metilazione) |

**Namespace degli id dei covariati (risolto).** Il file legacy `genomic_context.parquet` codifica `cpg_idx = chrom_code*1e9 + pos` (id del registro master), mentre l'universo del benchmark usa l'id di `array_cpg_map` (`load_benchmark_universe()`): i due namespace non si sovrappongono (l'avviso `no cpg_idx overlap` di `matching.build_covariates` era questo). Il join e' fatto per **(chrom, pos)**: 408,391/408,399 CpG con corrispondenza esatta; 8 CpG senza riga legacy (sonda assente dalla tabella illumina GRCh38) hanno il contesto **ricalcolato** dalla traccia UCSC cpgIslandExt hg38 con la regola documentata (accordo regola ricalcolata vs legacy 99.9966%); round-trip dell'id verificato (`coordinates/MANIFEST_genomic_context.json`). Covariati finali: `covariates/universe_covariates.parquet` (contesto, densita' CpG universo e hg38, GC su +-500 bp, log10 distanza TSS GENCODE v50, probe_type).
**`probe_type`**: 6,429 CpG 'unknown' (probe su nessun manifest locale), non imputati e **non usati nel matching** (solo bilancio riportato; livello 'unknown' nei modelli di bilancio).

**Provenienza e build (riassunto).** Loyfer: GRCh38 nativo; 205 campioni da S1 pubblicata + 48 decisioni manuali; marker WEAK (sopra). 4DN: GRCh38 inferita+validata. FANTOM5: GRCh38 nativo, CC BY 4.0. RT: hg19 con liftover (6.6% bin non mappati + 0.4% scartati). PMD: hg19 verificata, liftover esplicito (2.45% intervalli persi). Universo: **solo autosomi** (chr1-22).

**Caveat aperti (onesti, non risolti).**
1. Provenienza dei marker hg38 **WEAK**: i file GitHub hg38 non coincidono con le tabelle del paper (67-74% di sovrapposizione); marker-kNN secondario; procedura di derivazione hg38 non documentata.
2. Decato: liftover hg19->GRCh38 perde il 2.45% degli intervalli; PMD derivato da metilazione (circolare rispetto al compito di predizione di metilazione, non agli input ENCODE).
3. RT: solo Rep1 (BJ Rep1 vs Rep2 r = 0.989; le altre 14 linee non hanno Rep2); WaveSignal non e' un log-ratio; autocorrelazione forte (~600 blocchi efficaci); 3 LCL GM altamente correlate; mostly cancer/immortalizzate; nessuna sorgente di replica indipendente; CpG da array sbilanciati verso RT precoce.
4. E1 orientato dal GC (A = Spearman(E1, GC) > 0): compartimenti non indipendenti dalla composizione di sequenza (rilevante per i comparatori SEQ); accoppiamento E1-GC piu' debole in HFFc6 (0.52).
5. 4DN inter: bilancio debole (SMD 0.30) e drop 98.3%: solo esplorativo.
6. Input CTCF (201) / RAD21 (14) / SMC3 (6) ChIP sono meccanisticamente legati agli anchor dei loop: i contatti intra sono `partially_related`, non corretti.
7. FANTOM5: sovrapposizione **indiretta** con marchi enhancer (H3K27ac/H3K4me1/DNase/TF, cCRE pELS/dELS) e TSS GENCODE come covariata/input: indipendenza non piena. Set "permissive", non il robust Andersson 2014. Pool principale `bg_enh5k` non filtrato per TSS (deviazione dichiarata sopra); il design TSS-filtrato e' mal bilanciato (SMD -0.117).
8. L'affermazione che le classi cCRE V4 sono definite da segnale ENCODE DNase/H3K4me3/H3K27ac/CTCF e' **conoscenza esterna, non verificata localmente** (il manifest registra solo versione e hash del BED).
9. Una sola linea cellulare per sistema 3D (H1, HFFc6, GM12878): nessun Hi-C tessuto-matched; profondita' diversa (3.05 M vs 2.35 M contatti) = batch effect; bias MNase (Micro-C) e DpnII (GM12878); md5 mcool completo non verificato.
10. Cooler 4DN con `genome-assembly` "unknown" (GRCh38 inferita e validata, non dichiarata); pixel duplicati H1 sommati (5,806 chiavi, numeri post-deduplica).
11. **Copia decompressa `data/external/reference/hg38.fa` (3.27 GB) creata** (e relativo `.fai`) per le verifiche di build e il calcolo GC: e' un derivato di `hg38.fa.gz` (sha256 `c1dd8706...27d4` del fasta usato nelle validazioni), gitignored, non parte dei dati congelati.
12. Regola di eleggibilita' dei positivi 4DN (copertura >= 25-esimo percentile): decisione di design pre-freeze, non derivata da dati di embedding; e' un filtro che restringe i contatti a bin ben coperti (rimuove 8.2% delle coppie di bin H1).
13. Bootstrap per cromosoma con 22 blocchi: CI grossolana; coppie inter assegnate al blocco di `cpg_i`; CpG nella stessa coppia di bin non indipendenti.
14. Campioni Loyfer: sbilanciamento verso sangue/epiteli, donatori non indipendenti (205 campioni = 168 donatori), Megakaryocytes assente, marker solo ipometilati.
15. FANTOM5: mediana 1 libreria per gruppo, 90.8% zeri, 27 librerie senza categoria, gruppi non equivalenti a tessuti indipendenti.

## 6. Cosa NON e' stato fatto / vietato, regole di freeze

**Non fatto e vietato:**
- nessun confronto tra embedding su assi bioval v2, nessun metrico di embedding (coseno, AUROC, Spearman, kNN) calcolato prima del freeze di questo documento e dei file di sezione 7;
- nessuna UMAP/visualizzazione o ispezione qualitativa degli embedding su questi target;
- nessuna model selection, tuning o scelta di rappresentazione/dimensione/soglia guidata da questi assi;
- nessun uso di set di test (inclusi i protocolli di split del masking benchmark) per tarare parametri di bioval v2;
- nessuna GPU per la preparazione; nessuna modifica di `outputs/`, `src/cpg_repr_benchmark/bio_validation/`, dei dati congelati o delle soglie dei PREP;
- nessun matching/campionamento che usi embedding o loro derivati.

**Regole di freeze.**
1. Questo documento, `anti_circularity.yaml` e i manifest di sezione 7 sono committati PRIMA di calcolare qualunque metrica di embedding bioval v2 (hash del commit registrato).
2. Fissati e congelati (nessun punto aperto): politica di inferenza (sezione 0), 5 fold a blocchi seed 17, coseno come punteggio, AUROC primario 4DN con Delta-coseno secondario, `drop` per le coppie inter, matching e liste di coppie (sezione 7), pool FANTOM5 `bg_enh5k`. Restano da registrare al lancio, senza modificare i target: lista dei bracci/confronti e il valore di k del kNN (secondario).
3. La rappresentazione (file store/embedding, hash, parametri) viene congelata e ne viene registrato lo sha256 nel manifest prima del primo calcolo.
4. **I sottocomandi di freeze (`scripts/bioval_v2/freeze_4dn_pairs.py`, `freeze_fantom5_matching.py`, `loyfer_freeze_pairs.py`, e le preparazioni `prepare_*`, `build_bioval_v2_covariates.py`) NON devono essere rieseguiti dopo che esiste qualunque metrica di embedding bioval v2.** Una riesecuzione cambierebbe liste/sha256 o aprirebbe gradi di liberta' post-hoc; se serve una modifica, e' una deviazione documentata e analizzata come esplorativa.
5. Dopo il freeze: niente modifiche a target, soglie o metrica primaria; ogni deviazione e' documentata come tale. Le tabelle dei risultati riportano tutti gli endpoint pre-registrati, anche quelli nulli.

## 7. Artefatti congelati (sha256)

I file grandi sono in `data/derived/bioval_v2/` (gitignored); i manifest sotto sono la pista di verifica. Ogni file congelato FANTOM5 e' stato costruito due volte con sha256 identico (`FROZEN.json: determinism`).

**Manifest e verdetti** (sha256 del file JSON):

| File | sha256 |
|---|---|
| `3d_genome/FROZEN.json` | `743803ef7275bb00d214e754faac4df15c7dcade4c776b3fa149f45dfc0fe927` |
| `3d_genome/MANIFEST.json` | `68c87ec90019eb709fb6239d5d2ff31f91e3a93db2a2e5ef9e02ef090e042a28` |
| `3d_genome/assembly_validation.json` | `95349508dfec56e5c24ce29dcf5be659c08285ba290b24e4efba86aa0d5bc1cf` |
| `regulatory_activity/FROZEN.json` | `e0dba7e8770754608eae1368e04a5f285408005ad7d550e2fb1caff19cdd6e0a` |
| `regulatory_activity/MANIFEST.json` | `9cade60bc37c872314bd6a3545fe8c6ecb33c2a3d4986a51813b322812c74c7e` |
| `replication_domains/FROZEN_ENDPOINT.json` | `0ab4a6130d9790ca742a095c04e6236a16a5dc65f007d0f7edd46f98154d1eee` |
| `replication_domains/MANIFEST.json` | `f7c602b0b9badb4beb107404ff3fa838b82de7501f7267d121a270fc063fb016` |
| `replication_domains/MANIFEST_rt.json` | `756e74e05bc736fd43ae5a2e8081675e8120130d11344c792572204b36a70fdf` |
| `external_methylation_programs/MANIFEST.json` | `63576a0e28e5883d7beba7c7b33c1c7a148e6ad6d56a8029e58728fcfe307daa` |
| `external_methylation_programs/marker_provenance_verdict.json` | `22ebb6d348c43dcc774b5271529715a54d1a8e0bedadbb7819dae8fde5821b80` |
| `pmd_decato2020/MANIFEST_pmd.json` | `b8bbdd53d35b9d2459e4534d4af086a9bc9456b28276df9e055ee1002bb090bf` |
| `pmd_decato2020/BUILD_CHECK.json` | `ef5699280057f37f32c9aa123d377dc238794ee6cdac5dc9f378b260d25e9618` |
| `covariates/MANIFEST_covariates.json` | `2485cd61ab5b52fe581940707af0afd9271bce4a0349172f62bdbe85e0bf8dac` |
| `coordinates/MANIFEST_genomic_context.json` | `511f05a20fc15ca8e6f5c3f3a77fb860b4284d6ae1bf46d14609436f9507956c` |

**File di dati congelati:**

| File | sha256 |
|---|---|
| `external_methylation_programs/frozen_pairs_seed17.parquet` (386,268 coppie) | `fbb9440e066bcf3a8380b3d2b690e1b6b5dc9d6cefd090e70182bbe12e282ce7` |
| `3d_genome/frozen_pairs_H1_intra10kb_seed17.parquet` | `968e10846bc5d5ec2ab69805f2e5c2ece0fd028d280af1ddc5e516f728fb3b0e` |
| `3d_genome/frozen_pairs_HFFc6_intra10kb_seed17.parquet` | `6e91aa310c3141cc00b6e2eed370a43205dadbe9637c3304b2cc08cf0f728f02` |
| `3d_genome/frozen_pairs_H1_inter1Mb_seed17.parquet` | `06d6b569d3bdc591896ed38cee997695fe80512e1a03bde30f6d60544e9e01b8` |
| `3d_genome/frozen_compartments_H1_100kb.parquet` | `10d7a9280d32b813dae4e026d383d0a2e730da73083df7eb6c82b28e8ca70220` |
| `3d_genome/frozen_compartments_HFFc6_100kb.parquet` | `682967d947f326ce507c64eaa4482a1e11537f406d4fa34c36857b70f8d001b4` |
| `3d_genome/frozen_compartments_GM12878_100kb.parquet` | `103234b7ccf2f67bca74df284669e569a4982a0a599c48598f5918cb9ee3b89c` |
| `regulatory_activity/frozen_enhancer_membership_seed17.parquet` | `9f0ed5a8b2ea402db674fc695e848d26ef8028b0e3cfb83821dabe7187d039e8` |
| `regulatory_activity/frozen_enhancer_membership_win500_seed17.parquet` | `54fbe2e264aeba4ef9ed5a6377362cf295752e544c4c366cf2e8fff7e8d98c72` |
| `regulatory_activity/frozen_enhancer_membership_tss5kpool_seed17.parquet` | `3e868f9e3f33113f2dec42638788652948e23ddc44a0bed78abbec957bb7d7ef` |
| `regulatory_activity/frozen_activity_pairs_seed17.parquet` | `4e365148b8680e7c349b6ee69e476cb446d1a7bf1abf0e7b48dfb579c001cb7e` |
| `regulatory_activity/..._seed17_dropped_positives.parquet` | `d597259467128c93d5909402bd03bb413f7bb8fb5adc5234f03fd979c92f5e5f` |
| `replication_domains/cpg_rt_universe.parquet` | `fa4fcd3c2f841a9db9454544103c580dcd87e0ea760e95832d2dc2ec1d90cef6` |
| `covariates/universe_covariates.parquet` | `466428137dcc1a80f998b2dd07a52812b015f239a4a798b12ce77660f69e3997` |
| `coordinates/genomic_context_canonical.parquet` | `5c1eb4c0dc398588316eadf219c653feadfe3de4007f2f9ecf8f8c3ca5528196` |
| `external_methylation_programs/loyfer_markers_reconciled.parquet` | `073325a0f472930b77f9e02acd0e46f293655c0056c32923c55e24b9a5ec06e6` |
| `pmd_decato2020/cpg_pmd_summary_universe.parquet` | `06d70eba96d1873c3b39d9ed90f617b10566f693bc1c06e1df131ca4313a7437` |

Altri sha256 (win500 / tss5k dropped positives, report 4DN) sono in `3d_genome/FROZEN.json` e `regulatory_activity/FROZEN.json`.

**Copertura completa dei checksum.** I manifest per asse non coprono tutti i derivati (es. matrici Loyfer, tabelle marker, `sample_provenance`, PMD). `data/derived/bioval_v2/MANIFEST_checksums.json` (generato da `scripts/bioval_v2/build_checksum_manifest.py`) registra sha256 e dimensione di **tutti** i file derivati al freeze; i parquet/h5/npy restano fuori da git (directory ignorate) e sono referenziati solo per hash.

**Manifest versionati con `git add -f`** (piccoli, sotto directory ignorate): `FROZEN*.json`, `MANIFEST*.json` (incluso `MANIFEST_checksums.json`), `marker_provenance_verdict.json`, `BUILD_CHECK.json`, `assembly_validation.json`.

**Fonte esterna non versionata: traccia UCSC `cpgIslandExt` hg38.** `data/bio_annotations/cpgIslandExt.hg38.txt.gz` (717,984 B, sha256 `2339f8bad0ec9993211287b319b8d59ddeff30a5010505f649817332ed8222f8`, upstream Last-Modified 19 ott 2022) NON e' committato per cautela sulla politica di ridistribuzione UCSC (la pagina licenza UCSC consente l'uso dei file ma prevede limiti per alcune tracce di terzi). Si riottiene in modo deterministico con `python scripts/fetch_cpgislandext_hg38.py` (URL `https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/cpgIslandExt.txt.gz`; rifiuta il file se lo sha256 non coincide). Verificato il 2026-10-03: il ri-download e' identico byte per byte. Serve solo a ricalcolare il contesto di 8 CpG (sezione 4) e a ricostruire i covariati.

## 8. Registro del freeze

Il tag git `bioval-v2-protocol-freeze-v1` punta al commit di freeze, il cui SHA e' registrato qui sotto in un commit successivo (un commit non puo' contenere il proprio hash). Quel commit modifica solo questa riga, non il protocollo.

- Commit di freeze: _(registrato nel commit successivo)_
- Tag: `bioval-v2-protocol-freeze-v1`

# Freeze confermativo v2 (BOZZA) - valutazione one-shot sul TEST TCGA del pannello post-hoc

Stato: preparato il 2026-10-11. `freeze_state: draft`, `test_set_authorized: false`. Il TEST TCGA non e' stato letto, valutato ne' toccato
(nessuna predizione, nessuna riga della matrice di test). Sostituisce per questo pannello `docs/CONFIRMATORY_FREEZE_DRAFT.md`
(che resta come storico). Nessun commit e' stato fatto; nessun risultato congelato esistente e' stato modificato.

## 1. Domanda e linguaggio
Come nel draft: "i modelli addestrati a predire misure funzionali tendono a dare rappresentazioni piu' rilevanti per la metilazione".
Non ammesso: "la supervisione funzionale e' necessaria", "X batte il candidato regolatorio". Nessuna PCA/SVD nel pannello.

## 2. Pannello: 8 arm x 3 seed (17/42/97) = 24 run, variante latent/locus
| Classe | Arm |
|---|---|
| misurata (riferimento) | `regulatory_raw2492` (Histone+DNase raw, 2.492 binarie, nessuna riduzione) |
| sequence-derived | `cpgpt_large_locus`, `deepcpg_dna_locus`, `ntv3_pre_latent_native` |
| sequence-to-function | `alphagenome_locus`, `enformer_latent_native`, `ntv3_post_latent_native`, `sei_latent_native` |

Esclusi per decisione dell'utente: `regulatory_histone_dnase` (SVD-256, rimosso anche dai confronti), tutte le varianti `functional`.
I 24 run sono stati scelti PRIMA di vedere il test; epoche e `best.pt` sono gia' fissati (120 epoche addestrate, `best.pt` = minimo MSE di validazione a
mask 0.50, nessuna ri-ottimizzazione, nessuna selezione che usi il test). Sei seed 42 e 97 erano ancora in training al momento della preparazione:
nel manifest sono `status: pending`; il freeze non puo' diventare final ne' il runner partire finche' tutti i 24 non sono completi e hashati
(`python scripts/build_test_confirmation_v2_freeze.py` rigenera manifest e config appena i run compaiono su hal con marker
`outputs/logs/remote_queue_done/sei_latent_s{seed}`, 120 epoche e metriche di validazione).

## 3. Disegno fisso e hash
Identico per tutti (dataset TCGA array, `patient_split_seed 20260925`, split pazienti/loci da `outputs/encode_atlas_v1/{patients,loci_seen}.npz`,
loci genome-wide seen-only, prior da train x train, stesso modello, mask 0.15/0.30/0.50/0.70/0.90, `mask_seed 17001`, `panel_size 2048`,
`panel_repeats 1`). Nessun cambio a modello, split, mask, seed. Hash (sha256) in
`configs/experiments/test_confirmation_v2/freeze_manifest.yaml`: per ogni run `best.pt` (con epoca), `resolved_config.yaml`, store HDF5, `patient_split.npz`,
`locus_split.npz`, `prior_logit.npy`, config di valutazione (file e forma canonica), MSE di validazione a mask 0.50; piu' sha256 dei protocolli globali.

## 4. Meccanismo del lucchetto (modifiche minime)
- Config di valutazione (24, `configs/experiments/test_confirmation_v2/<arm>__seed<S>.yaml`): copia del `resolved_config.yaml` del run; differiscono SOLO
  `evaluation.patient_view: test`, rimozione di `require_patient_view`, `test_authorization: tcga_posthoc_freeze_v2`, `test_freeze_manifest`,
  `test_freeze_run_id`, `preserve_run_files: true` (verifica automatica nello script di build: qualunque altra differenza e' un errore). Layout `split_dirs`,
  `allow_overwrite_split_dir: false`, `num_workers: 0`.
- Output solo in `<run>/evaluation/test/` (`split_output_dir` rifiuta qualsiasi percorso fuori o dentro `evaluation/validation/`); una cartella test non vuota non e' riusata.
- Nuova policy opt-in in `eval_layout.authorizer_from_cfg` -> `experiments/test_confirmation_v2.py`. Sblocca solo se: manifest `test_set_authorized: true` E `freeze_state: final`;
  tutti i 24 run `complete` e tutti gli hash (best.pt, config, split, prior, config di valutazione di tutti i run; store dell'arm valutato, tutti gli store nel runner) coincidono;
  la config in uso e' identica a quella congelata; la config non contiene `require_patient_view` ne' `allow_overwrite_split_dir`; manifest, config di valutazione e codice sul
  percorso di valutazione sono committati e puliti in git. La procedura TCGA storica (`matrix.yaml`) non e' toccata.
- I run con `require_patient_view: validation` non possono valutare il test (`guards.enforce_patient_view`, invariato).
- Unica modifica a codice esistente: `resolve_representation(write_manifest=...)` e `run_masking_benchmark.py` (`evaluation.preserve_run_files`), perche' in modo `evaluate` il
  resolver riscriveva `representation_manifest.json` del run congelato. Default invariato.
- Runner `scripts/run_test_confirmation_v2.py`: senza `--execute` e' solo un controllo; con `--execute` e' sequenziale, idempotente (marker
  `evaluation/test/.done_tcga_posthoc_freeze_v2.json`), sotto `flock /data2/home/vcivale/.gpu_extract.lock` e `nice -n 10`, si ferma al primo errore o a una cartella test parziale.
- Test sintetici: `tests/test_test_confirmation_v2.py`.

## 5. Metriche e analisi
Primaria: MSE a mask 0.50 sui pazienti di test. Secondarie: MAE, MAS-PCC, MAC-PCC, skill vs prior, sweep mask.
Analisi: delta MSE appaiato (arm - riferimento del confronto) a mask 0.50, stesso seed; bootstrap incrociato paziente x blocco genomico 1 Mb, 2000 repliche,
seed 17 (`encode_atlas/statistics.py`); IC95 per seed e sulla media dei seed (IC sui pazienti/loci, non sull'inizializzazione). Nessun test inferenziale tra modelli,
nessun p-value composito tra seed, classi di 1-4 modelli solo descrittive.

## 6. Confronti primari (prespecificati, direzione attesa: delta < 0 per l'arm funzionale/misurato)
- P1: ciascun arm sequence-to-function (AlphaGenome, Enformer, NTv3-post, Sei) vs ciascun riferimento sequence-derived CpGPT-large e DeepCpG-DNA: 8 confronti.
- P2: NTv3-post vs NTv3-pre (stessa famiglia): 1 confronto.
- P3: raw2492 vs CpGPT-large e vs DeepCpG-DNA: 2 confronti.
Totale: 11 confronti confermativi (x 3 seed), dichiarato; IC95 non corretti per molteplicita'. Holm sull'insieme P1-P3 solo come analisi di sensibilita'.
Descrittivi (non confermativi): ogni arm vs raw2492 (7 confronti); NTv3-pre/CpGPT/DeepCpG tra loro. Nessun confronto con SVD-256.
Soglia di interpretazione proposta: "tendono a" solo se l'IC95 esclude 0 con lo stesso segno in tutti e 3 i seed.

## 7. Cosa e' post-hoc
Tutti i modelli del pannello oltre ai 3 arm congelati originali (CpGPT, DeepCpG, Histone+DNase) sono stati aggiunti post-hoc, dopo aver visto metriche di VALIDAZIONE;
la scelta della variante latent e del pannello e' pero' precedente al test. Il test TCGA non e' mai stato letto. GSE40279 e' escluso da questo freeze (gia' consumato dai vecchi arm).
Le analisi EWAS v3, bioval v2, genomic-context restano post-hoc e fuori dal freeze.

## 8. Limiti
Dimensioni native non matched (128-21.907 prima dell'adapter); scala/dati di pretraining diversi; DeepCpG/CpGPT supervisionati su metilazione, quindi "sequence-derived" = input solo sequenza;
seed non indipendenti (stessi split); le varianti dello stesso modello non sono repliche; classi di 1-4 modelli; il codice di valutazione e lo store sono quelli al momento del freeze:
gli store sono hashati ora (per alcuni arm il config non aveva sha256: l'hash attuale non e' confrontabile con quello usato in addestramento, solo con i file correnti);
modifiche non committate a `engine.py`/`hdf5_store.py` (float16 su disco, cast a float32 sul device, lossless) vanno committate prima del freeze.

## 9. Decisioni residue per l'utente
1. Approvare pannello, confronti P1-P3 (11), direzione attesa, soglia "tendono a", Holm solo come sensibilita'.
2. Attendere Sei s42/s97 su hal, rigenerare il manifest e rileggere gli hash.
3. Committare codice + config + manifest (draft), poi un secondo commit che imposta `freeze_state: final` e `test_set_authorized: true`.
4. Classificazione delle classi (DeepCpG/CpGPT sequence-derived; AlphaGenome e Sei sequence-to-function).
5. Trattamento di SVD-256 nel testo (ora fuori dal freeze) e della Fig. 4.
6. GSE40279 come analisi esplorativa separata o esclusa.
7. Dichiarazione dei confondenti come limiti.

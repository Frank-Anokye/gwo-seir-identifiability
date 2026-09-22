# SEIR Model Fitting and Parameter Identifiability Using Grey Wolf Optimization on Ghana's First COVID-19 Wave

This repository holds the code, data, results, and preprint for a study in two parts.

The first part builds and tests the Grey Wolf Optimizer, a general purpose search algorithm (which mimics the social heirarchy and the hunting strategy of grey wolves) on four standard benchmark functions with known correct answers, at three problem dimensions (d=2,5,and 10) each, twelve test cases in total, to confirm the implementation works correctly before it is trusted on real data.

The second part uses that same optimizer (unmodified codebase) to fit a Susceptible,
Exposed, Infectious, Recovered (SEIR) disease model to real, publicly reported COVID-19 case data from Ghana's first outbreak wave, 14 March to 15 October 2020. Fitting the model without any constraints reveals a genuine identifiability problem. The reporting fraction and the number of people exposed at the start of the outbreak cannot be separated from the case count data alone, even though the model as a whole fits well. This is shown directly, by repeating the fitting
process many times from different random starting points and
comparing the results, instead of reporting a single fitted curve. The problem is then resolved using outside evidence from Ghanaian seroprevalence studies and international under-reporting estimates, after which the transmission rate becomes tightly and consistently identified. Because a single, constant transmission rate cannot reproduce an outbreak curve that flattens out long before a large share of the population is infected, the model is extended to allow one estimated change in the transmission rate over time. This improves the fit to an R^2 of 0.934 and finds that the effective reproduction number dropped from about 2.4 to about 1.1 around day 50 of the outbreak, a time that is consistent with Ghana's documented lockdown and mask-mandate timeline.

The full write-up is in [`paper/preprint.pdf`](paper/preprint.pdf)
(compile it yourself, or see the compiled PDF in the same folder).

## Repository structure

```
.
├── code/                Python source code
├── data/                Raw and processed Ghana COVID-19 case data
├── figures/              All figures used in the paper
├── paper/                LaTeX source, references, and the compiled preprint
├── results/              Numerical result tables produced by the code
├── LICENSE
├── README.md
└── requirements.txt
```

### code/

- `gwo.py` — Grey Wolf Optimizer, written from scratch, works with any objective function
- `benchmark_functions.py` — Sphere, Shifted Sphere, Rastrigin, Rosenbrock
- `run_benchmarks.py` — runs the 12 benchmark test cases
- `seir_model.py` — the SEIR model, in both its single-rate and two-phase transmission-rate forms
- `fit_seir_ghana.py` — unconstrained 3-parameter fit, plus the identifiability check
- `fit_seir_ghana_constrained.py` — fit with the reporting fraction fixed from outside evidence
- `fit_seir_ghana_two_phase.py` — fit allowing the transmission rate to change once over time
- `generate_additional_figures.py`, `generate_more_figures.py`, `generate_landscape_grid.py` — produce the remaining figures used in the paper

### data/

- `raw/` — the unedited case count file as downloaded from the source below
- `ghana_covid19_first_wave.csv` — Ghana's data cut down to the 216 day window used in this study

### paper/

- `preprint.tex` — the LaTeX source
- `references.bib` — the reference list
- the compiled PDF

## Data source

The case count data is Ghana's daily cumulative confirmed COVID-19 cases, published by the Center for Systems Science and Engineering at Johns Hopkins University (JHU CSSE), and downloaded from the project's public data repository on GitHub: [https://github.com/CSSEGISandData/COVID-19]. The exact file used is (https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv), kept unedited in this repository at data/raw/. No numbers in the processed file were changed, removed, or invented.

## How to reproduce everything

1. Install the dependencies.

   ```bash
   pip install -r requirements.txt
   ```

2. Run the benchmark validation.

   ```bash
   cd code
   python3 run_benchmarks.py
   ```

3. Run the unconstrained SEIR fit.

   ```bash
   python3 fit_seir_ghana.py
   ```

4. Run the constrained fit (reporting fraction fixed from outside evidence).

   ```bash
   python3 fit_seir_ghana_constrained.py
   ```

5. Run the two-phase (time-varying transmission rate) fit.

   ```bash
   python3 fit_seir_ghana_two_phase.py
   ```

6. Generate the remaining figures.

   ```bash
   python3 generate_additional_figures.py
   python3 generate_more_figures.py
   python3 generate_landscape_grid.py
   ```

7. Compile the paper (requires a LaTeX distribution with the
   `IEEEtran` class, available on Overleaf or via `texlive-publishers` on Debian and Ubuntu). Four passes, in this order, are needed so the reference list resolves fully.

   ```bash
   cd ../paper
   pdflatex preprint.tex
   bibtex preprint
   pdflatex preprint.tex
   pdflatex preprint.tex
   ```

Every script finds its own file locations automatically, based on its own position on disk, not on the folder it happens to be run from. Every fitting run also uses a fixed, recorded random seed, so every number in the paper can be
reproduced exactly.

## Summary of key results

- **Benchmark validation** — the Grey Wolf Optimizer reliably recovers the known global minimum of Sphere, Shifted Sphere, and Rastrigin, and comes close to the known minimum of the harder Rosenbrock function, across 20 independent runs at every tested size (`results/benchmark_results_table.csv`).
- **Unconstrained fit** — the transmission rate is consistentlynrecovered, but the reporting fraction and the initial exposed count are not. Many different combinations of these two values reach an almost identical fit (`results/seir_fit_all_runs.csv`,
  `results/identifiability_summary.txt`).
- **Constrained fit** — fixing the reporting fraction at 0.05, based on real Ghanaian seroprevalence data and international under-reporting estimates, resolves this. The transmission rate converges to 0.1921 in every one of 20 independent runs (`results/identifiability_constrained_summary.txt`).
- **Two-phase fit** — allowing the transmission rate to change once, at a day estimated from the data, captures the whole shape of Ghana's first wave (R-squared of 0.934), and finds a drop in the effective reproduction number from about 2.4 to about 1.1 around day 50 of the outbreak
  (`results/seir_two_phase_summary.txt`).

## Limitations

These are discussed in full in the paper. In short: the reporting fraction used is an informed estimate built from real evidence, not a value measured directly for Ghana's first wave. The SEIR model does not include age structure, spatial structure, or a direct representation of testing capacity over time. The two-phase transmission-rate model is a simplified stand-in for what was likely a more gradual real change in transmission. Only Ghana's first wave is modelled, not later waves shaped by variants and vaccination.

## License

See [`LICENSE`](LICENSE) for reuse terms. The underlying case count data belongs to Johns Hopkins University CSSE and is used here under their terms of public release.

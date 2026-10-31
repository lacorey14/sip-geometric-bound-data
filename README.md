# Data and scripts for a geometric upper bound on SIP relaxation times

Source tables and Python scripts for the manuscript *A geometric upper bound on relaxation times in spectral induced polarization*.

The scripts regenerate the manuscript figures and the in-band two-Debye contact fits from the files in this archive. Archived finite-element spectra are not recomputed here. Contact times and the two-sphere comparison are read from the deposited tables.

## Layout

- `source_data/` — comma-separated values for Figures 1–4 and Supporting Information Tables S1–S2 and Figures S1–S2
- `results/` — numerical tables read by the figure scripts
- `inputs/dimensionless_regime_map.json` — archived interior-conductivity sweep used for Figure 3b
- `inputs/phase_diagram.csv` — archived two-sphere contact spectra used by `fit_two_mode_contact.py`
- `plot_three_regime_figure.py` — Figures 1 and 3
- `plot_peak_exclusion_figure.py` — Figure 2
- `plot_moment_validation.py` — Figure 4
- `plot_contact_weights.py` — Figure S1
- `plot_relaxation_spectrum.py` — Figure S2
- `fit_two_mode_contact.py` — in-band two-Debye fits written to `results/two_mode_contact_fits.json`

## Reproduce

```bash
python3 -m pip install -r requirements.txt
python3 plot_three_regime_figure.py
python3 plot_peak_exclusion_figure.py
python3 plot_moment_validation.py
python3 plot_contact_weights.py
python3 plot_relaxation_spectrum.py
python3 fit_two_mode_contact.py
```

The figure scripts write SVG, PDF, PNG, and TIFF files under `results/` and `figures_vector/`.

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Repository: https://github.com/lacorey14/sip-geometric-bound-data

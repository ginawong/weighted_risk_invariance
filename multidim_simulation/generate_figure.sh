#!/bin/bash

# generate the results for each subfigure
python sweep_hparams_figure_a.py || exit 1
python sweep_hparams_figure_b.py || exit 1
python sweep_hparams_figure_c.py || exit 1

# aggregate results and create final plot
python plot_figure.py || exit 1

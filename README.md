# Rogalla et al. (2022) analysis and Mn model forcing setup code

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This repository contains post processing scripts and notebooks to reproduce the results presented in the manuscript *Sediments in sea ice drive the Canada Basin surface Mn maximum: insights from an Arctic Mn ocean model* by B. Rogalla, S. E. Allen, M. Colombo., P. G. Myers, K. J. Orians (2022). It is divided into:

* **forcing** - Notebooks that are used to create the forcing files of the Mn model.
* **paper materials** - Notebooks that are used to create the figures and tables in the paper. File naming conventions; if the Jupyter Notebook starts with:
  - "S" --> supplementary materials
  - "M" --> methods
  - "R" --> results
  - "D" --> discussion
  - "E" --> extra
  
The following number(s) corresponds to the paper element number. Background maps are loaded as pickles to speed up the plotting. These pickles are created in the notebook "map-pickles.ipynb".  
* **calculations** - Mixture of bash scripts to calculate monthly averages, python code to calculate Mn transport, and notebooks used for simple calculations.

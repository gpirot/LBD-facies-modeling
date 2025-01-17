# LBD-facies-modeling
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/gpirot/LBD-facies-modeling/blob/main/LICENSE)

This repository provides data and code to generate stochastic facies models of the Lower Burdekin Delta aquifer. It allows extracting facies information from borhole log descriptions (provided by the [State of Queensland](https://www.business.qld.gov.au/industries/mining-energy-water/resources/geoscience-information/gsq), Australia), generating stochastic facies models of the aquifers, and calibrate the model algorithm parameters to match some summary statistics.

## Requirements
The following python packages are used:
   - bayesian-optimization
   - matplotlib
   - numpy
   - geone
   - geostatspy
   - seaborn
   - scipy
   - pandas
   - pickle
   - pyevtk


## Examples
A series of *Python* notebooks demonstrate the different steps of the modeling approach and how to use the different functions. 
The notebooks need to be executed in the numbered sequence in order to produce all necessary files.
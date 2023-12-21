# PAL-distortion-scaling
Data and analysis scripts for the study "Seeing the future of progressive glasses: a new perceptual approach to spectacle lens design"

## Publication - Preprint
The preprint of the study is available at https://doi.org/10.31234/osf.io/pge5n

Cite as:
```
Sauer, Y., Künstle, D., Wichmann, F., & Wahl, S. (2023).
Seeing the future of progressive glasses: a new perceptual approach to spectacle lens design.
https://doi.org/10.31234/osf.io/pge5n
```

## Content
[/scripts](/scripts): analysis scripts

[/measurements](/measurements): raw measurement files. Each subfolder contains data for a single subject, csv files with the subjects answers and csv files with gaze and head tracking data.

[/data](/data): data compiled as Pandas DataFrames and saved as pickle. Trial data is saved as a DataFrame for each subject. Analysed subject behvaviour, scaling functions and embedding scores are saved in combined DataFrames.

[/figures](/figures): figures for the manuscript and Python scripts used to create them.

## Requirements
The tracking analysis uses [gaze3d](https://github.com/YannickSauer/gaze3d)

## Analysis
1. analyse_trials.py process subjects answers and perform ordinal embedding
2. analyse_tracking.y for processing the raw gaze and head tracking data. Creates processed tracking files for each subject and one additional behaviour.pkl with analysed behavioural measures for all subjects combined.
3. gaze_analysis.ipynb for deeper analysis of gaze data. 
4. statistics.ipynb for statistical analysis (differences between behaviour groups and linear mixed model)

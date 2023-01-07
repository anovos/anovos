# Anovos Geospatial Use Case Demo

This use case demo can be run directly using Jupyter notebook.
The source data is taken from [Kaggle](https://www.kaggle.com/datasets/ramjasmaurya/us-police-shootings-from-20152022),
but tweaked slightly to showcase the geospatial functionalities of Anovos.

## Data Installation

You'll find the `geospatial_use_case_demo/`  folder and a notebook within the folder.
To run this notebook, you will need to copy the data from
[this path](https://mobilewalla-anovos.s3.amazonaws.com/geospatial_use_case_demo/data.zip), unzip it and paste the
`data/` folder inside `geospatial_use_case_demo/`.
## Folder Structure

After pasting the downloaded `data/` folder, the structure of `geospatial_use_case_demo/` is as below:

geospatial_use_case_demo
<ul>
  <li>data</li>
    <ul>
      <li>shootings_modified.csv</li>
    </ul>
  <li>sample_report
    <ul>
      <li>basic_report.html</li>
      <li>ml_anovos_report.html</li>
    </ul>
  </li>
  <li>anovos_geospatial_use_case_demo.ipynb</li>
  <li>README.md</li>
</ul>

## Report Generation

Now, you can run the `anovos_geospatial_use_case_demo.ipynb` to explore Anovos' geospatial functionalities directly.
Alternatively, sample reports have been saved in `geospatial_use_case_demo/sample_report/` folder,
and you can find the _Geospatial Analyzer_ tab inside the `ml_anovos_report.html`.

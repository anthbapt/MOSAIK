---
title: 'Spatial trancriptomics workflow'
tags:
  - Python
  - Spatial-transcriptomics
  - Visualisation
  - Multi-modality
authors:
  - name: Anthony Baptista
    orcid: 0000-0002-8514-0250
    affiliation: "1, 2"
affiliations:
 - name: Cancer Bioinformatics, School of Cancer & Pharmaceutical Sciences, Faculty of Life Sciences and Medicine, King's College London, London, United Kingdom
   index: 1
 - name: The Alan Turing Institute, The British Library, London, NW1 2DB, United Kingdom.
   index: 2
date: 29 April 2024
bibliography: bibliography.bib
---


# Summary
Spatial transcriptomics (ST) has revolutionized transcriptomic analysis by preserving tissue architecture, enabling researchers to study gene expression within its spatial context. Understanding a cell’s position relative to its neighbors and extracellular structures provides crucial insights into cellular phenotype, function, and disease progression—particularly in cancer, where the tumor microenvironment (TME) influences key processes such as chemoresistance [@mehraj_tumor_2021]. The commercialization of ST platforms has broadened access to these technologies, earning ST the title of *“Method of the Year 2020”* by *Nature Methods* [@marx_method_2021].

Imaging-based fluorescence in situ hybridization (FISH) technologies provide high-multiplex, subcellular-resolution transcriptomic data across over one million cells. These platforms, such as CosMx by NanoString and Xenium by 10x Genomics, offer high sensitivity and specificity, facilitating the exploration of cell atlases, cell–cell interactions, and the phenotypic architecture of the TME [@chen_spatially_2015, @vandereyken_methods_2023].

Despite the promise of spatial transcriptomics, technical challenges remain. Two primary issues are: (1) the integration of raw data into analysis workflows, and (2) the accurate association of transcripts with individual cells.

To address the first challenge, we have developed a unified workflow that supports both CosMx and Xenium data. While Xenium data align relatively well, CosMx data lacked robust tooling, which we have now integrated into a seamless and consistent pipeline. Moreover, we developped this workflow by using the most suitable Python package spatialData library [@Marconato2025], which provide a suitable object to integrate both the spatial elements (images, space location of transcripts, cell segmentaton labels, cell segmentation polygons) and transcriptomics information, which is an annotated datframe which defines single-cell transcriptomics.


Addressing the second challenge requires precise spatial delineation of cells—making cell segmentation a critical first step. Segmentation quality directly impacts the accuracy of all downstream analyses. Our current workflow integrates native segmentation approaches: CosMx uses a Cellpose-based method [@Stringer2021], and Xenium applies a Voronoi expansion strategy. However, users can also opt for alternative or custom segmentation tools, which may offer better performance but require careful parameter tuning—often difficult to implement in tools like Xenium Ranger (10x Genomics) or AtoMx (NanoString).

This integrated and consistent pipeline provides a foundation for downstream modeling and analytical functions, and it offers a scalable solution for tackling key challenges in spatial transcriptomics—particularly in the context of multimodal data integration.


# Statement of need
# Overview of the workflow
# Related software
# Planned Enhancements

# Acknowledgements
Anthony Baptista acknowledges support from the CRUK City of London Centre Award CTRQQR-2021100004.

# References


## Introduction

At King's College London, the [Spatial Biology Facility](https://www.kcl.ac.uk/research/facilities/the-spatial-biology-facility) utilizes both CosMx and Xenium technologies. The data is then integrated into a Python workflow, allowing us to fully leverage spatial biology and facilitate multimodal integration.

## 🎯 Roadmap

### Numerical materials

- [Napari v0.4.17](https://github.com/napari/napari/releases/tag/v0.4.17) + CosMx plugin (`napari/napari_CosMx-0.4.17.3-py3-none-any.whl`)
- CosMx/Xenium conda environment (`env/cosmx.yml`)
- CosMx/Xenium Reader (`src/reader/`)
- CosMx/Xenium QC scripts (`src/qc/`)
- CosMx/Xenium resegmentation scripts (`src/segmentation/`)
- CosMx Cell composition code (`src/composition/`)
- CosMx/Xenium analysis codes (`src/analysis/`)

### Integration workflow

![Workflow](figures/sbf_workflow.png)

## How to use Napari

1. Install Napari 0.4.17.
2. Launch Napari and open the IPython console (symbol "**>_**").
3. Install the CosMx plugin:

    ```bash
    pip install napari_CosMx-0.4.17.3-py3-none-any.whl
    ```

4. Drag the `napari_cosmx_launcher` folder into the Napari window. You can download it from [this link](https://emckclac-my.sharepoint.com/:f:/g/personal/k2481276_kcl_ac_uk/EkO8xJFpX8ZBv_lq-5zRHQQBjcoTOE8PONclhqUfj20zSw?e=WtIiKX)
5. In the right panel, select the parent folder that contains your CosMx run.
6. Choose the output folder.
7. Click the "**Stitch**" button.
8. Wait for the stitching to finish (the only indicator is the loading cursor).
9. After stitching, the output directory will contain:
   - an `images` folder with all FOVs
   - a `targets.hdf5` file with the transcripts
10. Restart Napari 0.4.17 and drag the project folder into the window.

11. Once loaded, use the panels to explore:

    - **Morphology Images**: Add fluorescent channels.
    - **RNA Transcripts**: Add transcripts.
    - **Layer list**: Manage transcripts, channels, and segmentation.
    - **Layer controls**: Adjust visualization.

## SpatialData object overview

The **SpatialData** object forms the foundation for analyzing spatial omics data.

### Core concepts

A SpatialData object is a container for **Elements**, either:

- **SpatialElements**:
    - **Images**: e.g. H&E stains.
    - **Labels**: Segmentation maps.
    - **Points**: Transcripts or landmarks.
    - **Shapes**: Cell/nucleus boundaries or ROIs.
- **Tables**: Annotate spatial elements or store metadata (non-spatial).

### Categories

- **Rasters**: Pixel-based data (Images, Labels)
- **Vectors**: Geometry-based data (Points, Shapes)

### Transformations

- **Vectorization**: Converts Labels → Shapes (`shapely` polygons)
- **Rasterization**: Converts Shapes/Points → Labels (2D image representation)


## 📫 Contact

For any questions, reach out to **Anthony Baptista**:  
📧 [anthony.baptista@kcl.ac.uk](mailto:anthony.baptista@kcl.ac.uk)

## ♻️ License

This work is licensed under:

- **MIT license** (for code)
- **Creative Commons Attribution 4.0 International license** (for docs)

You may share/adapt for any purpose, including commercially, with proper credit and no added restrictions.


# References

## Contents

- Primary Literature
    - Cell phone based colorimetric analysis for point-of-care settings (2019)
    - From Sophisticated Analysis to Colorimetric Determination: Smartphone Spectrometers and Colorimetry (2018)
    - Clustering and Classification of a Qualitative Colorimetric Test (2018)
    - An intelligent mobile-enabled expert system for tuberculosis disease diagnosis in real time (2018)
    - A Smartphone-Based Automatic Measurement Method for Colorimetric pH Detection Using a Color Adaptation Algorithm (2017)
    - Digitization of Colorimetric Measurements for Quantitative Analyses Using a Smartphone (2017)
    - A smartphone readable colorimetric sensing platform for rapid multiple protein detection (2017)
    - Utilization of Smartphone as a Colorimetric Detector for Chemical Analyses (2017)
    - A smartphone algorithm with inter-phone repeatability for the analysis of colorimetric tests (2014)
    - Point-of-care colorimetric detection with a smartphone (2012)
- OpenCV
    - Background Subtraction

## Primary Literature

### Cell phone based colorimetric analysis for point-of-care settings (2019)

Nice figures! This paper emphasizes the importance of using multiple color spaces. It goes into a lot of detail on the subtle differences that arise from how you take the picture.

- https://doi.org/10.1039/C8AN02521E
- https://pubs-rsc-org.proxy.library.nd.edu/en/content/articlelanding/2019/an/c8an02521e#!divAbstract

### From Sophisticated Analysis to Colorimetric Determination: Smartphone Spectrometers and Colorimetry (2018)

Doesn't provide me with the sort of technical details that I need, but it's a very nice review.

- https://doi.org/10.5772/intechopen.82227
- https://www.intechopen.com/online-first/from-sophisticated-analysis-to-colorimetric-determination-smartphone-spectrometers-and-colorimetry

### Clustering and Classification of a Qualitative Colorimetric Test (2018)

Another not so useful for us article. This time it's about quantifying ELISA. They neural networks and stuff.

- https://doi.org/10.1109/iCCECOME.2018.8658480
- https://ieeexplore-ieee-org.proxy.library.nd.edu/document/8658480

### An intelligent mobile-enabled expert system for tuberculosis disease diagnosis in real time (2018)

Another ML model for ELISA.

- https://www-sciencedirect-com.proxy.library.nd.edu/science/article/pii/S0957417418304214

### A Smartphone-Based Automatic Measurement Method for Colorimetric pH Detection Using a Color Adaptation Algorithm (2017)

They use a small lightbox attached to their phone to help out with ambient light changes, unknown built-in automatic image correction, etc. They use another interesting color space. They make a calibration curve. There's some good math in here.

Nothing about background correction.

- https://doi.org/10.3390/s17071604

### Digitization of Colorimetric Measurements for Quantitative Analyses Using a Smartphone (2017)

A master's thesis on the viability of phone cameras as detectors for colorimetric tests. He goes into great detail on the differences between color spaces. He also makes wonderful observations on the varying performance across devices (Samsung, LG, etc.)

- https://pdfs.semanticscholar.org/6fe1/5e821ad54b3d9fdef6ba9d95f09387da1b5d.pdf

### A smartphone readable colorimetric sensing platform for rapid multiple protein detection (2017)

This group forgoes RGB entirely. They use a decision tree instead.

- https://pubs-rsc-org.proxy.library.nd.edu/en/content/articlelanding/2017/an/c7an00990a#!divAbstract

### Utilization of Smartphone as a Colorimetric Detector for Chemical Analyses (2017)

This is a thesis that does some PAD style work (and a lot of other semi-related stuff). I didn't get a lot from this besides that we should pay attention to (and possibly experiment with) other color spaces.

- http://dspace.calstate.edu/bitstream/handle/10211.3/194491/HuangChaoNan_Thesis2017.pdf?sequence=4

### A smartphone algorithm with inter-phone repeatability for the analysis of colorimetric tests (2014)

These folks convert from RGB colorspace to some color spaces that I've never heard of. They don't provide justification for why they do this.

They use the coordinates on the new color space to predict the concentrations of unknowns by calculating the Euclidean distance to measured calibration points.

They don't verify the integrity of their work across multiple devices.

- https://doi.org/10.1016/j.snb.2014.01.077

### Point-of-care colorimetric detection with a smartphone (2012)

It's great seeing such work being done in 2012. I think I had a Razer phone back then. Anyway, this shows that RGB color space is useless (for their particular test). However, CIE color space worked pretty well.

- https://doi.org/10.1039/c2lc40741h
- https://www.researchgate.net/publication/230895153_Point-of-care_colorimetric_detection_with_a_smartphone

## OpenCV

### Background Subtraction

Describes how to use OpenCV to do background subtraction (on videos).

- https://docs.opencv.org/3.2.0/db/d5c/tutorial_py_bg_subtraction.html
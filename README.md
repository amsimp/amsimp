# AMSIMP - Numerical Weather Prediction using Machine Learning

![AMSIMP Build](https://github.com/amsimp/amsimp/workflows/Build%20AMSIMP/badge.svg)
![Anaconda-Server Badge](https://anaconda.org/amsimp/amsimp/badges/version.svg)
![GitHub](https://img.shields.io/github/license/amsimp/amsimp.svg?style=flat-square)
![Anaconda-Server Badge](https://anaconda.org/amsimp/amsimp/badges/downloads.svg)
![GitHub last commit](https://img.shields.io/github/last-commit/amsimp/amsimp.svg?style=flat-square)

AMSIMP is an open-source solution that leverages machine learning to improve numerical weather prediction. Read the [paper](https://github.com/amsimp/papers/raw/master/scifest/national/project-book/main.pdf). Due to data corruption on my end, the model no longer functions.

**Features:**

* Fast and accurate, AMSIMP's neural networks provide high quality weather forecasts and predictions.
* AMSIMP offers the pretrained operational AMSIMP Global Forecast Model (AMSIMP GFM) architecture. It is trained on a dataset from the past decade, ranging from the year 2009 to the year 2016. Over time, a future model will be trained on a larger dataset.
* AMSIMP offers near real-time numerical weather prediction through initialisation conditions provided by the Global Data Assimilation System.
* The core of AMSIMP is well-optimized Python code. A performance increase of 6.18 times can be expected in comparison against a physics-based model of a similar resolution.
* AMSIMP's high level and intuitive syntax makes it accessible for programmers and atmospheric scientists of any experience level.
* Distributed under the [GNU General Public License v3.0](https://github.com/amsimp/amsimp/blob/master/LICENSE), AMSIMP is developed [publicly on GitHub](https://github.com/amsimp/amsimp).

## Installation

This package is available on [Anaconda Cloud](https://anaconda.org/amsimp/amsimp), and can be installed using conda:

```bash
$ conda install -c amsimp amsimp  
```

For more information, please [read the documentation](https://docs.amsimp.com) on the website.

## License
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see [this webpage](https://www.gnu.org/licenses/).

## Call for Contributions
AMSIMP appreciates help from a wide range of different backgrounds. Work such as high level documentation or website improvements are extremely valuable. Small improvements or fixes are always appreciated.

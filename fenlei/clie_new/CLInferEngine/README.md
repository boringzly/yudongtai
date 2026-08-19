# CLInferEngine 2.0.5

*Organization:* Cangling AI Team

*Author:* Yang Xuan, Chen Pan, Zan Luyang

<br/>

## CLIE Introduction

CLInferEngine is also called AutoTest.

CLInferEngine is designed for CLSegmentation and CLChangeDetection.

CLInferEngine should be worked with main framework.

## 2.0.5 (2022-02-26)

- Fixed a bug for image_reader in last row
- Fixed a bug for num of bands in exception value proc

## 2.0.4 (2022-01-13)

- Fixed a bug when using failed image writer

## 2.0.3 (2022-01-13)

- Fixed a bug when skip error data
- Fixed a bug when polygonize a failed raster data

## 2.0.2 (2022-01-13)

- Fixed opt.retry ref error in run_lib
- Fixed test_data.image_reader.nbands ref error in run_lib

## 2.0.1 (2022-01-13)

- Fixed a bug when using distributed infer

## 2.0 (2022-01-13)

- Reimplement ImageIO interface for very very big file.
- Supported overlap setting.
- Supported multi batch size.
- Supported multi GPU including standard mode and distributed mode.
- Optimized RAM utils.
- Supported in-time infer progress information.
- Almost reimplement the whole project.
- Wish you could enjoy this version.

## 1.0 (2021-08-10)

- This is a initial version forked from CLSegTools.
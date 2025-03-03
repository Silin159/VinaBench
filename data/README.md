# VinaBench Data

### VinaBench narratives and annotations
Please follow this [link](https://drive.google.com/file/d/1GtKTcQbcHx2RFU8FVcu6JaQxkecOJ3l7/view?usp=sharing) to download the VinaBench narrative collections and annotations of narrative constraints, and unzip `annotations.zip` under this repository.

Multiple portions of narratives are included in `annotations`:
- [Visual Writing Prompts (VWP)](https://arxiv.org/abs/2301.08571): `vwp_train.json` and `vwp_test.json`
- [Storyboard20K](https://arxiv.org/abs/2404.15909): `sb20k_train.json` and `sb20k_test.json`
- [StorySalon](https://arxiv.org/abs/2306.00973) (original): `salon_train.json` and `salon_test.json`
- StorySalon (splitted short version): `salon_short_train.json` and `salon_short_test.json`

### VinaBench visual narrative images
VinaBench collections of visual narrative images:
- Visual Writing Prompts (VWP): images could be downloaded via the links provided in the `vwp_{train|test}.json`, or from this [link](https://drive.google.com/file/d/1GG0tlnOOSQYNtAkcu2thXXDfXA7J3fkC/view?usp=sharing) and unzip `images.zip` under this repository.
- Storyboard20K: please refer to the Storyboard20K [repository](https://github.com/showlab/Long-form-Video-Prior) (Source Movie Frames) to get the images of this portion, and put the image folder `storyboard20k` under this repository.
- StorySalon: our preprocessed images could be downloaded via this [link](https://drive.google.com/file/d/163zOLYRFFw6D4swZGnm6aTzMKl6EJDak/view?usp=sharing) and unzip `Image_inpainted.zip` under this repository.

### (Optional) Scripts for VinaBench annotations
```
bash annotate.sh
```
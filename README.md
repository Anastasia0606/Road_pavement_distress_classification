### Classification of road pavement distress images based on deep learning algorithms and RADAM architecture

We applied pretrained ResNet34 and ResNet18 to the image dataset of almost 45 000 images of 7 classes of destressors. Different augmentations from Albumentation library were used. To improve model performance and shorten time of model training we used RADAM method (Leonardo Scabini, Kallil M. Zielinski, Lucas C. Ribas, Wesley N. Gonçalves, Bernard De Baets, Odemir M. Bruno, RADAM: Texture recognition through randomized aggregated encoding of deep activation maps, Pattern Recognition, Volume 143, 2023, 109802, ISSN 0031-3203, https://doi.org/10.1016/j.patcog.2023.109802.) This method was specifically developed for texture recognition, so we found it appropriate for our task.

Module Data_preprocessing.py contains the code for creating description dataframe from original data, data analisys, data visualisation and data split for further experiments.

Notebook Augmentations.ipynb contains visualisations for different type of augmentations. This information may be useful in choosing transforms for data augmentations.

Notebook main.ipynb contains code and results of our experiments with classical deep-learning approach. Pretrained models ResNet34 and ResNet18 were imported from module ResNet.py, that was loaded from torchvision Docs (https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html)

Notebook main_Radam.ipynb contains code and results of our experiments with Radam method. To use it we also load modules feature_extraction.py, RNN.py and models.py from git repository (https://github.com/scabini/RADAM/blob/32c3d099acdfa89e783ae1dd63f1fe1d75bfea0c/README.md)

Presentation of the project in power point:
https://docs.google.com/presentation/d/1EVkZbjSNa_gxWpyzboqt8pgQwD725fnU/edit?usp=sharing&ouid=113357925123457690517&rtpof=true&sd=true

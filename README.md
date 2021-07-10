# Deep Learning Program
## Liver segmentation based on deep convolutional networks

U-NET and UNET ++ models based on RESNET were constructed, and the models were evaluated and tested in the liver Segmentation dataset of the Medical Segment Decathlon Challenge. The experimental results show that both models have good effects, while the UNET ++ model based on RESNET has better performance. In addition, we combined Focal Loss and Dice Loss to design a mixed Loss function to obtain more accurate segmentation results.

We also combined U-Net with void convolution, used void convolution to replace conventional convolution operation to expand the convolution receptive field, and constructed and trained DC-UNET. Due to resource constraints, we have not been able to test the segmentation effect of this model, which can be used as a direction for future work.

Data set comes from the "Medical Segmentation Decathlon Challenge" liver Segmentation of data set and can be downloaded from the website (http://medicaldecathlon.com/).

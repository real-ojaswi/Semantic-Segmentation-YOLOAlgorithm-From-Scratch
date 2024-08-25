## Implementation of Semantic Segmentation Logic from Scratch
The repository consists of the codes for developing **Semantic Segmentation Algorithm** from scratch using Custom U-Net. 

The file *SegmentationCOCODataset.ipynb* consists the code for semantic segmentation using MSE Loss and Dice Loss on images of few selected classes of COCO dataset. *DataGeneratorCocoSegmentation.py* has the code for extracting the dataset using COCO Api. The same code is also in the notebook file but commented. Here the training logic has been adapted from DLStudio (https://engineering.purdue.edu/kak/distDLS/DLStudio-2.3.3_CodeOnly.html). The training has been carried out using MSE Loss, Dice Loss, and combined, and the result has been compared.

The file *SegmentationPurdueDataset.ipynb* consists of the code for semantic segmentation using MSE Loss and Dice Loss. The SemanticSegmentation inner class from DLStudio (https://engineering.purdue.edu/kak/distDLS/) has been overloaded and used. The training is done on PurdueShapes5MultiObjectDataset that comes with DLStudio.

Some of the results using a) MSE Loss b) Dice Loss c) Combined Loss on CoCo Dataset on 'motorcycle' and 'dog' class:

-
  ![result_MSE](https://github.com/thenoobcoderr/Semantic-Segmentation-Algorithm-from-scratch/assets/139956609/f8f9014c-ef32-4712-b451-83796a58435b)
-
  ![result_Dice](https://github.com/thenoobcoderr/Semantic-Segmentation-Algorithm-from-scratch/assets/139956609/fe182911-2593-4828-842a-dfb788990fb0)
-
  ![result_combined](https://github.com/thenoobcoderr/Semantic-Segmentation-Algorithm-from-scratch/assets/139956609/eaf42272-df3e-4c00-b744-9bb61fe91701)




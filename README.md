# machine-learning-classification-between-suicidal-ideation-and-depression

[Classification Between Suicidal Ideation and Depression Through Natural Language Processing using Recurrent Neural Network](https://ejournal.uin-suska.ac.id/index.php/IJAIDM/article/view/17485/pdf) is a group research paper for Research Methodology in Computer Science course in the fourth semester, which was published in the [Indonesian Journal of Artificial Intelligence and Data Mining (IJAIDM)](https://ejournal.uin-suska.ac.id/index.php/IJAIDM/index). Below is the abstract to the paper. The paper can be viewed through the [IJAIDM website](https://ejournal.uin-suska.ac.id/index.php/IJAIDM/article/view/17485/pdf) or by opening the pdf file within the [repository](https://github.com/rhe-naldy/machine-learning-classification-between-suicidal-ideation-and-depression/blob/main/17485-63577-1-PB.pdf)

# ABSTRACT
The use of machine learning has been implemented in various ways, including to detect depression in individuals. However, there is hardly any research done regarding classification between suicidal ideations and depression among individuals through text analysis. Differentiating between depression and suicidal ideation is crucial, considering the difference in treatment between the two mental illness. In this paper, we propose a detection model using Recurrent Neural Network (RNN) in the hopes to improve previous models made by other researchers. By comparing the proposed model with the previous works as the baseline model, we discovered that the proposed model (RNN) performed better than the baseline models, with the accuracy of 86.81%, precision of 97.13%, recall score of 94.69%, f1 score of 95.90%, and area under the curve (AUC) score of 92.84%.

### Code Explanation
1. [unsupervised-clustering.py](https://github.com/rhe-naldy/machine-learning-classification-between-suicidal-ideation-and-depression/blob/main/unsupervised-clustering.py)

In this file, we used Uniform Manifold Approximation and Projection (UMAP) to reduce the dimensionality of the word embeddings and then implemented Gaussian Mixture Model (GMM) to cluster the dataset based on probabilities

2. [threshold-based-label-correction.py](https://github.com/rhe-naldy/machine-learning-classification-between-suicidal-ideation-and-depression/blob/main/threshold-based-label-correction.py)

In this file, we implemented a threshold-based label correction with the threshold set to 0.9. The implementation of Gaussian Mixture produced two different labels: the original ground-truth labels and the new unsupervised clustering labels. The new labels will be utilized to correct the ground-truth labels or the original labels. When the produced label has a probability above the tuned threshold, the original label will be replaced by the new label produced by Gaussian Mixture. Otherwise, the original label will not be replaced. By implementing a threshold-based label correction, we avoid the elimination of noisy labels.

3. [model-training.py](https://github.com/rhe-naldy/machine-learning-classification-between-suicidal-ideation-and-depression/blob/main/model-training.py)

In this file, we constructed a Recurrent Neural Network (RNN) as the proposed classifier. Our RNN model implemented a 5-unit Gated Recurrent Unit (GRU) layer as the first layer, utilizing rectified linear unit (ReLU) as the activation function, and he uniform as the kernel initializer. The first layer is then directly flattened, followed by a 64-unit dense layer, utilizing rectified linear unit (ReLU) as the activation function, and he uniform as the kernel initializer. In the end, a sigmoid function was applied to calculate the final output. The model architecture can be seen as below.

![image](https://user-images.githubusercontent.com/45966986/197165614-c9c4dd27-8a0b-4c95-88f0-0089c0480d8e.png)

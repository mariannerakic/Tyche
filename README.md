# Tyche: Stochastic In-Context Learning for Medical Image Segmentation


Official implementation of: _Tyche: Stochastic In-Context Learning for Medical Image Segmentation_.  
[Marianne Rakic](https://mariannerakic.github.io/), [Hallee W. Wong](https://halleewong.github.io/), [Jose Javier Gonzalez Ortiz](https://josejg.com/),  
[Beth A. Cimini](https://www.broadinstitute.org/bios/beth-cimini), [John V. Guttag](https://people.csail.mit.edu/guttag/) \& [Adrian V. Dalca](https://www.mit.edu/~adalca/)


## Abstract
Existing learning-based solutions to medical image segmentation have two important shortcomings. First, for each new segmentation task, usually a new model has to be trained or fine-tuned. This requires extensive resources and machine-learning expertise, and is therefore often infeasible for medical researchers and clinicians. Second, most existing segmentation methods produce a single deterministic segmentation mask for a given image. However, in practice, there is often considerable uncertainty about what constitutes the _correct_ segmentation, and different expert annotators will often segment the same image differently. We tackle both of these problems with _Tyche_, a model that uses a context set to generate stochastic predictions for previously unseen tasks without the need to retrain. Tyche differs from other in-context segmentation methods in two important ways.  

1. We introduce a novel convolution block architecture that enables interactions among predictions.
2. We introduce in-context test-time augmentation, a new mechanism to provide prediction stochasticity.

When combined with appropriate model design and loss functions, Tyche can predict a set of plausible diverse segmentation candidates for new or unseen medical images and segmentation tasks without the need to retrain.

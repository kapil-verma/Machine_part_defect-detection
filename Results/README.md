## Approaches:
Initial approach was to apply object localization and image cropping for all of the images to increase consistency i.e. to locate the machine part first and then the defects. The issue with this approach was that with computer Vision, It would require to rewriting our code to find our regions of interests whenever there are changes in product types. 
<br>Another approach, to make it easier for the model to learn how to differentiate between the defective and healthy parts, by feeding in processed images with convolution kernels. But these approaches would have made the model weaker in the sense that in those cases it would need only object- localized and specially processed images to give accurate results.<br>
> Hence final decision was to use the original images as it is with some minor processing of rotation, resizing and grayscale conversion (just for facilitating faster computation) so that model can learn to predict the defects from images with no particularly defined scale, angle, light settings etc. Later Approach of filtering with Kernel was retained for the purpose of manual inspection
(some samples are provided here).

### Deep Learning Architectures:
As we are having insufficiency for training data (just 250 images) for applying deep learning so I have employed transfer learning to mitigate that problem.
1)	**Convolutional neural network (CNN)** as the baseline model: 3 Conv2D/MaxPooling2D pairs as the feature extractor and 3 Dense layers as the classifier.
Transfer Learning Models
2)	**InceptionV3**: [Keras Application InceptionV3](https://keras.io/applications/#mobilenet) fine-tuning the classifier by using 1 GlobalAveragePooling2D layer and 2 Dense layers.
3)	**MobileNet**: [Keras Application MobileNet](https://keras.io/applications/#inceptionv3) fine-tuning the classifier by using 1 GlobalAveragePooling2D layer and 2 Dense layers

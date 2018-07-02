# tr.contrib.learn , classifier, prediction
# 1. Dataset 
# 2. Classifier
# 3. Train
# 4. Evaluate
# 5. Predict

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"
# Read data from .csv file
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING,
    target_dtype=np.int,# 0-2
    features_dtype=np.float)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TEST,
    target_dtype=np.int,#0-2
    features_dtype=np.float)
# All feautes has real values.
feature_columns = [tf.contrib.layers.real_valued_column("",dimension=4)]
# DNN with 30, 30, 30 unit(Neuron)
# iris has 4 features so the number of dimensions is 4
# n_claases = 3 (3 types)
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,hidden_units=[30,30,30],n_classes=3,model_dir="/tmp/iris_model")
# Train, the number of iteration is 2000
classifier.fit(x=training_set.data,y=training_set.target,steps=2000)
# evaluate the model
accuracy_score = classifier.evaluate(x=test_set.data,y=test_set.target)["accuracy"]
# The accuracy 0.966667
print("accuracy:",accuracy_score)
# To predict the new samples, we used predict

new_samples = np.array([[6.4,3.2,4.5,1.5],[5.8,3.1,5.0,1.7]],dtype=float)
y = list(classifier.predict(new_samples,as_iterable=True))
# The Prediction : [1 1]
# iris vesicolor
print("Prediction",str(y))

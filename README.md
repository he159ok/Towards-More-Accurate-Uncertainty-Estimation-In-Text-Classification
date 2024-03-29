This is code for Towards More Accurate Uncertainty Estimation In Text Classification

This is a readme file including a brief instruction of Baseline and detailed instruction for MSD model.

_______________________________________________________________________________________
Brief instruction of the code of Baselines

Please refer to the github released by the authors of NAACL2019 paper: Mitigating Uncertainty in Document Classification

Their Github link: https://github.com/xuczhang/UncertainDC

For the experiments of CNN on the Ylep,  BiGRU on Amazon and XLnet on Amazon, which are not tested by the authors of baselines, we set -metric-margin=0.1 and -metric-param=0.1 for them.
_______________________________________________________________________________________
_______________________________________________________________________________________
Detailed instruction of the code #450 of EMNLP2020 paper: Towards More Accurate Uncertainty Estimation In Text Classification

This paper aims at generating more accurate uncertainty score based on DNN models in text classification with human involvement.

We propose MSD with three independent components to improve the CWS by reducing overconfidence of winning score and handling impact of different uncertainty.

Extensive experiments on four real-world datasets show that MSD improves the accuracy of uncertainty scores in a flexible way.

______
The running enviroment: 
python = 3.6.8
Pytorch = 1.0.1
numpy = 1.16.2
Scikit-learn = 0.20.3

Please download the data.zip file from emnlp 2020 supplementary material, which has only 20News dataset due to upload limiation. And please make the unzip "data" file in same path of "main.py".

Please download "glove.6B.zip" from https://nlp.stanford.edu/projects/glove/ and unzip it. Please make the unzipped file "glove.6B" in the root directory under the "data" file.

______


To use the code, run 'main.py' for cnn and BiGRU, run 'main_XLnet.py' for XLnet , which both include train, eval and test processes.

1. Train process
'''
E.g. Train MSD with mix-up and self-ensebmling components by BiGRU on Amzon dataset:

python main.py
-emnlp
-epochs
4
-dataset
amazon
-device
0
-early-stop
1000
-mixup
-selfensemble
-intraRate
0.1
-lstm
```
Where you can apply "-mixup" to set whether apply mix-up or not; apply "-self-ensemble" to set whether apply self-ensembling or not, and "-intraRate [float]" to set $\lambada_2$ to the value of [float]; apply "-dataset [dataset_name]" to set different dataset, which [dataset_name] equals to one of [20news, imdb, amazon, autogsr, yelp] in string format; apply "-lstm" to set whether apply BiGRU or not, the default DNN is CNN.

'''
E.g. Train MSD with mix-up and self-ensebmling components by XLnet on Amzon dataset:
python main_XLnet
-epochs
4
-dataset
amazon_xlnet
-device
0
-test-interval
100
-batch-size
32
-dropout
0.3
-emnlp
-mixup
-selfensemble
-intraRate
0.1
'''
Where the [amazon_xlnet] are saved last hidden state from pretrained XLnet model and apply for the XLnet experiments only. 

2. Val process
As for val process, we will save the best model weights when epoches increased by training, the "best" is decided by the improved micro F1 in 0% eliminated ratio.

3. Test process
We apply stored model weights names as './snapshot/stored.pt' to run test, which apply respective parameters to have same components and DNN model as ones in the training process.

'''
E.g. Train MSD with mix-up and self-ensebmling components by BiGRU on Amazon dataset:
python main.py
-test
-snapshot
'./snapshot/stored_weight.pt'
-dataset
amazon
-use_idk
-emnlptev
-device
1
-individual_eval
-mixup
-selfensemble
-lstm
-MdistTest
'''
where "-MdistTest" is whether to apply distinctiveness score in the test process or not.

If apply "-MdistTest", it is necessary to calculate the mean and inverse of covariance matrix by running as "main_testMyMetric.py",
The mean and covariance matrix of the 20news, imdb, amazon, amazon (for XLnet feature) and yelp have been uploaded.
'''
For the above testing example, we should run below firstly,
python main_testMyMetric.py
-emnlp
-dataset
amazon
-device
1
-mixup
-snapshot
./snapshot/2020-05-28_00-49-56/best_steps_10100.pt
-calmeanvar
-selfensemble
-lstm
'''

## Note
This release is only available for EMNLP 2020 Review.

# wavenet_classifier_torch
pyTorch implementation of a WaveNet Classifier for supervised learning. 

-----------------

## General:
Initially created for time-series classification, expects 1-d inputs of fixed length (must be provided on 'init'). The last layer is left without an activation function to allow for either multi-class of multi-label classification.

## Requirements:
 - python3
 - torch
    either through pip:
    ```
    pip install torch===1.6.0 -f https://download.pytorch.org/whl/torch_stable.html
    ```
    or conda:
    ```
    conda install pytorch -c pytorch
    ```

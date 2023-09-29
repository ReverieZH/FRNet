# FRNet

## 1.Requirements

Here, we list our environment for the experiment on both *Linux* or *Window*.

```
# install

python 3.7
torch == 2.0.0
torchvision == 0.15.0
torch-dct == 0.1.6
numpy == 1.19.5
Pillow == 9.4.0
pysodmetrics == 1.3.1
```

The package ``torch-dct`` is used for the differential discrete cosine transformation in PyTorch, and more details can be found in this [repo](https://github.com/zh217/torch-dct). **Note**: A higher version for PyTorch has included this function and it may cause some problem. You should modify the source code of ``torch-dct`` or our code to solve the problem.

The package ``pysodmetrics`` is used for calculating the metrics for camouflaged object detection based on Python, as COD and SOD share similar metrics. The usage of this package can be found in [link](https://github.com/lartpang/PySODMetrics).

***

## 2.Data

### 2.1 COD

Before training the network, please download the cod train data:

```
---- TrainDataset
	 |---- Edge
       |---- ****.jpg
   |---- image
       |---- ****.jpg
   |---- mask
       |---- ****.png
```

Before testing the network, please download the cod test data:

* CAMO dataset

```
---- CAMO
	 |---- Edge
       |---- ****.jpg
   |---- image
       |---- ****.jpg
   |---- mask
       |---- ****.png
```

* CHAMELEON

```
---- CHAMELEON
   |---- Edge
       |---- ****.jpg
   |---- image
       |---- ****.jpg
   |---- mask
       |---- ****.png
```

* COD10K

```
---- COD10K
   |---- Edge
       |---- ****.jpg
   |---- image
       |---- ****.jpg
   |---- mask
       |---- ****.png
```

- NC4K

```
---- NC4K
   |---- Instance
       |---- ****.jpg
   |---- image
       |---- ****.jpg
   |---- mask
       |---- ****.png
```

**Recommendation**: you could extract train data and put them to the folder (``./data/COD/TrainDataset``). Then, you could extract test data and put them to the folder (``./data/COD/test``).The test folder should contain three folders: ``CAMO/, CHAMELEON/, COD10K/, NC4K/``.

### 2.2 FOSSIL3K

FOSSIL3K is currently not available to the public.If you need the val image, please contact with me

***

## 3.Train

## 4.Test and Evaluation

### 4.1 COD

#### Test

It is very simple to test the network. You can follow these steps:

1. You need to download the model weights [[Baidu Yun, wnxy]]( https://pan.baidu.com/s/1HKLhlCuWvfHT42G0EhSKPQ) or [[GoogleDrive]]( https://drive.google.com/file/d/1jSIrWKTFJ0wvXeiv3cF8qtCnrypVBlQX/view?usp=drive_link)

2. Change the model path in ``infer_freq.py`` Line. 22 to your need. 

   ```
   21  exp_name = 'FRNet'  
   22  net_path = os.path.join('./ckpt', exp_name, 'model.pth')
   ```

3. Change the data path in  ``config.py``  to your need

   ```
   datasets_root = './data/COD'  # test root floder
   
   chameleon_path = os.path.join(datasets_root, 'test/CHAMELEON')
   camo_path = os.path.join(datasets_root, 'test/CAMO')
   cod10k_path = os.path.join(datasets_root, 'test/COD10K')
   nc4k_path = os.path.join(datasets_root, 'test/NC4K')
   ```

2. Run ``infer_freq.py``.

#### Evaluation

You can use **Matlab** or **Python** script for evaluation.

* Python

You need to change the path of the ground-truth in ``config.py`` and the predictions in ``eval.py``. Using the python script is more simple and efficient.

ground-truth path in ``config.py`` 

predictions path in ``eval.py`` 

```
18  results_path = './results'
19	save_name = 'FRNet'  
20	save_dir = os.path.join(results_path, save_name)  # result dir
```

Then run ``python eval.py``. You can get the *Smeasure, mean Emeasure, weighted Fmeasure, and MAE*.

**Note**: We also upload the results [[Baidu Yun, gq5l]](https://pan.baidu.com/s/1-LB-PVZP5q0bJfHCpUJ4Gg ) or  [[GoogleDrive]](https://drive.google.com/file/d/12xvSM3BAlrp5WZ95t-4ZQHhVhOOanUVR/view?usp=drive_link).

* Matlab

You can also use the one-key evaluation toolbox for benchmarking provided by [Matlab version](https://github.com/DengPingFan/CODToolbox).

### 4.2 FOSSIL3K

#### Test

You can follow these steps as above:

1. You need to download the model weights [[Baidu Yun, 7jjz]]( https://pan.baidu.com/s/1NOQLVsZsra5_TOBzeFDWQg ) or [[GoogleDrive]](https://drive.google.com/file/d/1RnxoL1mNohh9Qvsd_6dzRkyaxIi_obRt/view?usp=share_link) and put it to the folder ``./ckpt/FOSSIL/``

2. Change the model path in ``infer_freq.py`` Line. 22 to your need. 

   ```
   21  exp_name = 'FOSSIl'  
   22  net_path = os.path.join('./ckpt', exp_name, 'model_fossil.pth')
   ```

3. Change the data path in  ``config.py``  to your need

   ```
   datasets_root = './data/NEW'  # test root floder
   
   fossil_val_root = os.path.join(datasets_root, 'test/FOSSIL3K')
   ```

4. Change the infer data in ``infer_freq.py`` Line 60

   ```
   60 to_test = OrderedDict([
                          ('FOSSIL', fossil_val_root),
                          ]) 
   ```

5. Run ``infer_freq.py``.

#### Evaluation

You can use **Matlab** or **Python** script for evaluation.

* Python

You need to change the path of the ground-truth in ``config.py`` and the predictions in ``eval.py``. Using the python script is more simple and efficient.

ground-truth path in ``config.py`` 

predictions path in ``eval.py`` 

```
18  results_path = './results'
19	save_name = 'FRNet'  
20	save_dir = os.path.join(results_path, save_name)  # result dir
```

Change the infer data in ``eval.py`` Line 60

Then run ``python eval.py``. You can get the *Smeasure, mean Emeasure, weighted Fmeasure, and MAE*.

**Note**: We also upload the results [[Baidu Yun, womg]](https://pan.baidu.com/s/1t92fPvPOafhqaoDK9MjGRA ) or  [[GoogleDrive]](https://drive.google.com/file/d/17iCW_UAjWttrKkMlAHtqNZw3uJuCEpAd/view?usp=drive_link.

* Matlab

You can also use the one-key evaluation toolbox for benchmarking provided by [Matlab version](https://github.com/DengPingFan/CODToolbox).




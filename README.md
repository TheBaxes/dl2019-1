# Colombian Traffic Sign detection

This project uses a modified implementation of YoloV3 in TensorFlow 2.0 cloned from zzh8829 repo.

For evaluation we used rafaelpadilla Object Detection Metrics in order to calculate mAP with a IOU threshold of 0.1

You can see a demo copying this notebook: https://colab.research.google.com/drive/1e3BHLXI3A7Vc-2ot2sTGyejDXwS5ExOx

#### Pip

```bash
pip install -r requirements.txt
```

#### Conda

```bash
conda env create -f environment.yml
conda activate yolov3-tf2
```

### Detection

```bash
python detect.py --weights model/yolov3_train_50.tf --image tests --classes data/traffic/names.txt --tiny --output output --val_dataset data/traffic.000

```


## References

- https://github.com/zzh8829/yolov3-tf2
- https://github.com/rafaelpadilla/Object-Detection-Metrics

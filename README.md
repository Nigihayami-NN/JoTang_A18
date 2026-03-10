# JoTang_A18
prepare for CSIEC

数据集配置文件在./CustomDataSet.yaml,里面记得写类别编号

数据集文件夹是./CustomDataSet/

之后把Images和Labels放里面，结构如下：
```
JoTang_A18
├── ultralytics
└──CustomDataSet
|     |-Images> train/ ; val/ ; test/ 
|     |-Labels> train/ ; val/ ; test/ #三个文件夹
|
|----CustomModel.yaml
|----CustomDataSet.yaml
```
其中每个文件夹中，rgb与thermal交替出现，如:000_rgb.png,000_thermal.png,111_rgb.png,111_thermal.png
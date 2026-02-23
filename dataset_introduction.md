# MS COCO Detection 数据集介绍

## 概述

本项目使用的数据集为 [detection-datasets/coco](https://huggingface.co/datasets/detection-datasets/coco)，是 **MS COCO（Common Objects in Context）** 数据集在 HuggingFace 上的 Parquet 格式版本。COCO 是计算机视觉领域最广泛使用的目标检测基准数据集之一，包含丰富的真实场景图片及对应的物体标注信息。

## 数据集规模

| 划分 | 样本数 |
|------|--------|
| train | 117,218 |
| val | 4,950 |
| **总计** | **122,218** |

## 数据字段

每条样本包含以下字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `image_id` | int | 图片唯一 ID |
| `image` | PIL Image | 图片数据 |
| `width` / `height` | int | 图片宽高（100–640 px，尺寸不固定） |
| `objects` | Sequence | 该图片中所有物体的标注信息 |

### `objects` 字段结构

每张图片包含一个或多个物体标注，结构如下：

```json
{
  "bbox_id":  [1038967, 1039564],
  "category": [45, 50],
  "bbox":     [[1.08, 187.69, 612.67, 473.53],
               [249.6, 229.27, 565.84, 474.35]],
  "area":     [120057.14, 49577.94]
}
```

- **bbox_id** — 标注框唯一 ID
- **category** — 类别索引（0–79），对应 80 个 COCO 物体类别
- **bbox** — 边界框坐标 `[x_min, y_min, x_max, y_max]`（绝对像素值）
- **area** — 边界框面积（像素²）

## 80 个物体类别

| 类型 | 类别名称 |
|------|----------|
| 人与动物 | person, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe |
| 交通工具 | bicycle, car, motorcycle, airplane, bus, train, truck, boat |
| 户外物体 | traffic light, fire hydrant, stop sign, parking meter, bench |
| 运动器材 | frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket |
| 餐具与食物 | bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake |
| 家具 | chair, couch, bed, dining table, toilet |
| 电子设备 | tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator |
| 日用品 | backpack, umbrella, handbag, tie, suitcase, book, clock, vase, scissors, teddy bear, hair drier, toothbrush, potted plant |

## 本项目中的使用方式

### 数据划分

```
full_dataset (117,218 张)
├── train_dataset: 前 3,000 张  → LoRA 微调
└── eval_dataset:  第 3,001–3,050 张 → 幻觉检测与分析
```

仅使用了一个较小的子集（3,050 / 117,218），以适配 8 GB 显存的硬件限制。

### 标签构建

将每张图片的物体类别去重、排序后用分号拼接为文本标签，作为 BLIP 模型的训练目标：

```
"car; person; skateboard"
"bottle; bowl; cup; knife; scissors; sink; spoon"
```

### 在幻觉分析中的角色

1. **训练阶段** — 用 3,000 张图片的「图片 → 物体列表」对微调 BLIP
2. **检测阶段** — 对 50 张 eval 图片生成物体列表，与 GT 标签对比，识别模型幻觉出的物体
3. **归因阶段** — 用 TracIn 追溯哪些训练样本对幻觉贡献最大

## 参考链接

- HuggingFace 数据集页面：<https://huggingface.co/datasets/detection-datasets/coco>
- COCO 官方网站：<https://cocodataset.org/>

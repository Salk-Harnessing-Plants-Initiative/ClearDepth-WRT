# ClearDepth-WRT (ClearDepth Wall Root Tracker)

## Purpose and background
**Purpose**: Segment images and extract traits from segmentation masks for clearpot images with different species (canola and soybean).

**Background**: To be updated with manuscript


## Installation


1. **Clone the repository**:  
   ```
   git clone https://github.com/Salk-Harnessing-Plants-Initiative/ClearDepth-WRT.git
   ```

2. **Navigate to the cloned directory**:  
   ```
   cd ClearDepth-WRT
   ```

## Organize the pipeline and your images
Models can be downloaded from [Box](https://salkinstitute.box.com/s/cqgv1dwm1hkf84eid72hdjqg47nwbpo5).

Please make sure to organize the downloaded pipeline, model, and your own images in the following architecture:

```
ClearDepth-WRT/
├── images/
│   ├── experimental design (e.g., day7, or batchA)/
│   │   ├── image (e.g., E-102_1.PNG)/
├── model/
│   ├── label_class_dict_lr.csv (class color)
│   ├── model name
├── env.yaml (environment file)
```

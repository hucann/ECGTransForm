## Reimplement Paper Workflow
**Step 1: High-Level (Paper reading)**

Note down key points in paper (abstract, intro, conclusion; figures & diagrams)

**Step 2: Medium-Level (Code Mapping)**

Map out the repo structure 
Identify "main flow": config → data loading → model init → training → evaluation

---

## ECGTransform

### Problem  
ECG arrhythmia classification is difficult due to:  
- Complex spatial + temporal patterns
- Class imbalance (e.g., more normal beats than rare arrhythmias)
- Limitations of traditional models in capturing both global + local patterns 

### Task (& motivation)
ECG arrhythmia classification: predict one label per heartbeat

In clinical practice, cardiologists look at the distribution and timing of abnormal beats, eg 
  - % of abnormal beats in 24 hours
  - Time intervals when arrhythmias occurred
  - Relationship with physical activity or medication

Example of possible use case:
-	Episode detection (e.g., 5+ PVCs → non-sustained VT).
- Arrhythmia burden analysis (e.g., % of abnormal beats).
- Trend visualization over long monitoring periods.

Possible further work 
- Raw ECG → Beat-level classification → Segment/episode classification → Final report

---

### Model Architecture 
#### 1. CNN - Multi-scale Convolutions (MSC)
Uses different kernel sizes (5, 9, 11) to capture fine-to-coarse spatial features.
#### 2. CRM (Channel Recalibration Module)
Like Squeeze-and-Excitation: re-weights important feature channels dynamically.
#### 3. Bidirectional Transformer
Processes both the normal and reversed input to capture temporal info from past and future.
#### 4. Context-Aware Loss (CAL)
A custom cross-entropy loss that assigns logarithmic weights to underrepresented classes.

### Training Techniques  
- Optimizer: **Adam (lr=1e-4, betas=(0.9,0.999))**  
- Loss: **Weighted cross-entropy** (N1 stage weight = 1.5)  
- Regularization: L2 (1e-3), Gradient clipping (5.0)  
- **Mini-batch size**: 20, sequence length = 15  

### Dateset
- MIT-BIH: 5-class beat classification (N,S,V,F,Q)
  - Annotated by two or more cardiologists (for each record)
- PTB: 2-class classification (normal VS myocardial infarction)

---

### Code Flow
```plaintext
main.py
  └── trainer.py
        ├── data_configs.py        → dataset-specific info (e.g., 5 classes for MIT-BIH)
        ├── hparams.py             → training config (epochs, batch size)
        ├── dataloader.py          → loads train/val/test .pt files, returns DataLoader
        ├── models.py              → defines ecgTransForm model
        ├── utils.py               → metrics, logs, model checkpoints
```


#### 1. main.py
- Entry point of the training script.
- Parses arguments like dataset (mit or ptb), path, device, etc.
- Instantiates the trainer and calls trainer.train().

#### 2. trainer.py
- Coordinates training, validation, testing, and logging.
- Loads:
  - Dataset config from data_configs.py
  - Hyperparameters from hparams.py
- Constructs the model (ecgTransForm)
- Loads data with data_generator() from dataloader.py
- Trains model, evaluates after every epoch, saves logs & checkpoints

#### 3. dataloader.py
- Loads .pt data files (preprocessed data files).
- Wraps them into a PyTorch Dataset and DataLoader.
- Also computes **class weights** for the custom loss function.
  - Log scaling of inverse class frequency -> focus on underrepresented classes

#### 4. models.py
- Defines the **model architecture** ecgTransForm, composed of:
  - 3 parallel conv layers (MSC)
  - CRM with Squeeze-and-Excitation
  - BiTrans (Transformer + reversed sequence)
  - Classifier head

#### 5. data_configs.py
- Defines dataset-specific parameters:
  - Class names, sequence length, input channels, etc.
- mit() and ptb() are classes that define these constants.

#### 6. hparams.py
- Stores training hyperparameters like:
  - Batch size, learning rate, epochs, feature dimension (1×128)
- Used when initializing the trainer.

#### 7. utils.py
- Utility functions for:
  - Logging and checkpointing
  - Fixing random seed
  - Metric calculations (Accuracy, F1)
  - Copying code files for backup
  - Optional UMAP visualization of learned features

---

### Data Preprocessing
- Only lead II is used (most common lead)
- Each heartbeat is extracted, centered around the R-peak
- Each beat is normalized to 186 samples
- Beats are labeled into 5 AAMI classes: N, S, V, F, Q
- The labels are derived from the original annotations and grouped based on AAMI standard
  - AAMI standard defines five classes of interest: normal (N), ventricular (V), supraventricular (S), fusion of normal and ventricular (F) and unknown beats (Q)

### To run: 
- conda activate wfdb-env
- python main.py --dataset mit --device cpu
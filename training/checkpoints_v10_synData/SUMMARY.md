# Experiment 10: Synthetic Data Pre-training (MobileNetV3)

## Objective
Pre-train the model strictly on perfectly clean synthetic patches, leveraging `train_syn.py`. The synthetic patches feature varying backgrounds, noisy Go boards, diverse angles, and strictly accurate `B`/`W`/`E` labels. 

## Experimental Setup
- **Architecture**: `MobileNetV3-Small`
- **Patch Size**: `48`
- **Transforms**: Full suite (Horizontal/Vertical Flip, 15° Rotation, Brightness/Contrast/Saturation Jitter)
- **Heuristic Filtering**: `False` (bypassed brightness filtering since synthetic labels are mathematically accurate)

## Results 
- **Epochs**: `4` (Early stop due to near-perfect accuracy convergence)
- **Training Accuracy**: `99.92%`
- **Validation Accuracy**: `99.98%`

## Conclusion
The model converged incredibly fast (4 epochs), hitting extreme accuracy limits on the synthetic patch set. The dataset contains ~265k training patches and ~50k validation patches. Continued training would yield rapidly diminishing returns. This checkpoint will serve as a robust baseline for fine-tuning on dirtier, real-world data distributions.

# Data Usage Summary - Auto Insurance Churn Training

## Previous Run (20% Sample)
- **Original Dataset**: 1,407,073 rows
- **Sampled Data**: 281,415 rows (20% of total)
- **Training Set**: 225,132 rows (80% of sampled)
- **Test Set**: 56,283 rows (20% of sampled)
- **Training Time**: ~30 seconds (EBM), ~6 seconds (XGBoost-GPU)

## Current Run (100% Full Dataset)
- **Original Dataset**: 1,407,073 rows
- **Sampled Data**: 1,407,073 rows (100% - no sampling)
- **Expected Training Set**: ~1,125,658 rows (80%)
- **Expected Test Set**: ~281,415 rows (20%)
- **Expected Training Time**: ~2-5 minutes (estimated)

## Why Data Was Reduced Before?
The script had `SAMPLE_FRACTION = 0.2` set at line 350, which randomly sampled only 20% of the data for faster initial training and testing. This is a common practice when:
- Testing code with large datasets
- Iterating quickly during development
- Running quick experiments

## What Changed?
Updated `SAMPLE_FRACTION = None` to use the complete dataset for final training.

## Training Configuration
- **CPUs Used**: 40 cores (distributed processing)
- **GPUs Available**: 4x NVIDIA GeForce GTX 1080 Ti
- **GPU Usage**: 
  - EBM: CPU-based (uses all 40 cores in parallel)
  - XGBoost: GPU-accelerated (uses CUDA)
- **Output Directory**: `ebm_results_multi_gpu/`

## Models Being Trained
1. **EBM (Explainable Boosting Machine)**: Microsoft InterpretML
   - Glass-box model with built-in interpretability
   - Uses gradient boosting with intelligibility constraints
   - CPU-distributed across 40 cores

2. **XGBoost-GPU**: GPU-accelerated gradient boosting
   - For performance comparison
   - Uses CUDA acceleration on GTX 1080 Ti GPUs
   - Typically faster training but less interpretable

## Performance Metrics (20% Sample Run)
### EBM Model:
- Test Accuracy: 88.67%
- Test ROC-AUC: 69.80%
- Test F1-Score: 22.77%
- Precision: 48.98%
- Recall: 14.83%

### XGBoost-GPU Model:
- Test Accuracy: 85.18%
- Test ROC-AUC: 69.47%
- Test F1-Score: 41.37%
- Precision: 37.30%
- Recall: 46.45%

## Top Features Identified (from 20% sample)
1. days_tenure (0.4739)
2. days_tenure & length_of_residence (0.0650)
3. length_of_residence (0.0499)
4. days_tenure & income (0.0389)
5. city (0.0379)

## Notes
- The full dataset training will produce more accurate and robust models
- Results should be more reliable with complete data
- Training time will be proportionally longer but still efficient with multi-core/GPU setup

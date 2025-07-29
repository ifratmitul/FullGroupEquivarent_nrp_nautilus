# Nautilus HPC Training Guide

A complete step-by-step guide for running machine learning experiments on NRP Nautilus HPC cluster.

## Prerequisites
- Access to Nautilus cluster with kubectl configured
- Your namespace: `longweiwangusd`
- Dataset files ready for upload
- Training code ready

## File Structure
```
├── 10layer_full_eqv_imagenet.py  # Your training code
├── exp_pod.yaml                  # Interactive experiment pod
├── pvc.yaml                      # Persistent Volume Claim (100GB)
├── train_job.yaml                # Training job configuration
└── upload_pod.yaml               # Data upload utility pod
```

## Step 1: Create Persistent Storage (100GB)

```bash
# Create PVC for storing datasets and models
kubectl apply -f pvc.yaml

# Verify PVC is bound
kubectl get pvc -n longweiwangusd
```

Expected output: `STATUS: Bound`

## Step 2: Upload Dataset

```bash
# Deploy upload pod
kubectl apply -f upload_pod.yaml

# Wait for pod to be running
kubectl get pods -n longweiwangusd

# Upload your datasets (replace with your actual filenames)
kubectl cp ../../Downloads/imagenet100.zip longweiwangusd/data-upload-pod:/data/

# Access pod to unzip datasets
kubectl exec -it data-upload-pod -n longweiwangusd -- bash

# Inside the pod:
apt update && apt install -y unzip
cd /data
unzip imagenet100.zip
ls -la  # Verify extraction
exit

# Clean up upload pod
kubectl delete pod data-upload-pod -n longweiwangusd
```

## Step 3: Interactive Development (Optional)

```bash
# Deploy interactive pod with GPU for testing
kubectl apply -f exp_pod.yaml

# Wait for pod to be running
kubectl get pods -n longweiwangusd

# Access the pod
kubectl exec -it imagenet-experiment-pod -n longweiwangusd -- bash

# Setup environment inside pod
cd /data
cat > setup_environment.sh << 'EOF'
#!/bin/bash
echo "Setting up Anaconda environment for ImageNet experiments..."
conda update -n base -c defaults conda -y
conda create -n imagenet-exp python=3.9 -y
source activate imagenet-exp
conda install pytorch==2.2.0 torchvision==0.17.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install tensorflow==2.15.0 keras==2.15.0 e2cnn==0.1.5
pip install "numpy>=1.23.5,<2.0.0" matplotlib==3.8.2 seaborn==0.12.2
pip install scikit-learn==1.3.0 opencv-python==4.8.0.76 pillow==10.2.0
echo "Environment setup complete!"
EOF

chmod +x setup_environment.sh
./setup_environment.sh

# Activate environment and test
source activate imagenet-exp
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
nvidia-smi

# Copy your training code
# From local machine (new terminal):
kubectl cp 10layer_full_eqv_imagenet.py longweiwangusd/imagenet-experiment-pod:/data/imagenet_experiment.py

# Test run (optional)
python imagenet_experiment.py

# Exit pod when ready
exit
```

## Step 4: Production Training Job

```bash
# Delete interactive pod (required - PVC can only mount to one pod)
kubectl delete pod imagenet-experiment-pod -n longweiwangusd

# Deploy training job
kubectl apply -f train_job.yaml

# Monitor job status
kubectl get jobs -n longweiwangusd
kubectl get pods -n longweiwangusd

# Follow training logs
kubectl logs job/imagenet-training-job -n longweiwangusd -f

# Check job completion
kubectl get jobs -n longweiwangusd
```

## Step 5: Access Trained Model

```bash
# Once training completes, create a new pod to access results
kubectl apply -f exp_pod.yaml

# Access the pod
kubectl exec -it imagenet-experiment-pod -n longweiwangusd -- bash

# Check saved model
cd /data
ls -la *.pth

# Download model to local machine (from local terminal)
kubectl cp longweiwangusd/imagenet-experiment-pod:/data/10layerfulleqv_imagenet100.pth ./trained_model.pth

# Clean up
exit
kubectl delete pod imagenet-experiment-pod -n longweiwangusd
```

## Step 6: Cleanup

```bash
# Delete completed job
kubectl delete job imagenet-training-job -n longweiwangusd

# Delete PVC (only when completely done - this deletes all data!)
kubectl delete pvc experiment-data-pvc -n longweiwangusd
```

## Important Notes

### Resource Specifications
- **PVC**: 100GB storage (`rook-ceph-block`)
- **GPU**: 1x RTX A6000 (48GB GPU memory)
- **RAM**: 32GB (Nautilus pod limit)
- **CPU**: 16 cores (Nautilus pod limit)
- **Shared Memory**: 8GB

### Package Versions
- `pytorch==2.2.0`
- `torchvision==0.17.0`
- `tensorflow==2.15.0`
- `keras==2.15.0`
- `e2cnn==0.1.5`
- `numpy>=1.23.5,<2.0.0` (for e2cnn compatibility)

### Common Issues & Solutions

**PVC Multi-Attach Error**: Only one pod can mount `ReadWriteOnce` PVC at a time
```bash
kubectl delete pod <conflicting-pod-name> -n longweiwangusd
```

**GPU Pod Stuck in ContainerCreating**: Check if RTX A6000 is available
```bash
kubectl get nodes -l nvidia.com/gpu.product=NVIDIA-RTX-A6000
```

**NumPy Compatibility**: If getting `np.float` errors
```bash
pip install "numpy>=1.19.0,<1.24.0"
```

### Useful Commands

```bash
# Check all resources in namespace
kubectl get all -n longweiwangusd

# Check PVC status
kubectl get pvc -n longweiwangusd

# Check available GPU nodes
kubectl get nodes -L nvidia.com/gpu.product

# Check events for debugging
kubectl get events -n longweiwangusd --sort-by='.lastTimestamp'

# Describe pod for detailed info
kubectl describe pod <pod-name> -n longweiwangusd
```

## File Locations in PVC

After completing the setup, your `/data` directory contains:
- `ImageNet100/` - Extracted dataset
- `imagenet_experiment.py` - Training code  
- `setup_environment.sh` - Environment setup script
- `10layerfulleqv_imagenet100.pth` - Trained model (after training)
- `imagenet100.zip` - Original dataset zip (optional to keep)# FullGroupEquivarent_nrp_nautilus

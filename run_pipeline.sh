#!/bin/bash
set -e  # Exit on error

#SBATCH --job-name=weibo_pipeline
#SBATCH --output=logs/pipeline_%j.log
#SBATCH --error=logs/pipeline_%j.err
#SBATCH --time=10:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8

# Create logs directory if it doesn't exist
mkdir -p logs

# Immediate output to file for debugging
exec 1>logs/debug1.log 2>&1

echo "============================================"
echo "SLURM Job Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "============================================"

# Initialize Python environment
echo "Activating Python virtual environment..."
source .env/bin/activate

# Check versions
echo "Python version:"
python --version
echo "Pandas version:"
python -c "import pandas; print(pandas.__version__)"

# Run the pipeline
echo "============================================"
echo "Running Weibo Pipeline..."
echo "============================================"
python weibo_pipeline.py

# Check exit status
if [ $? -eq 0 ]; then
    echo "============================================"
    echo "Pipeline completed successfully!"
    echo "End time: $(date)"
    echo "============================================"
else
    echo "============================================"
    echo "Pipeline failed with exit code: $?"
    echo "End time: $(date)"
    echo "============================================"
    exit 1
fi

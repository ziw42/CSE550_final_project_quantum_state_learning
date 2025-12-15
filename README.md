This is the code for the final project: quantum state learning, of CSE550.

## Environment setup

This repo was developed with Python 3.7 + TensorFlow 1.15. Do not install higher version of Python since we need TensorFlow 1.xx, which is not satble or even available in higher Python version.


```powershell
conda env create -f environment.yml
conda activate cse550
python -c "import tensorflow as tf; import google.protobuf; print(tf.__version__)"
```

### Notes / common issues

- If you see `Descriptors cannot be created directly` when importing TensorFlow 1.xx, ensure `protobuf==3.20.3` is installed (already pinned in `environment.yml`).
- If you see `cudart64_100.dll not found`, it is a GPU runtime warning from TensorFlow and can be ignored if you are running on CPU.

## Run training

For RBM:

```powershell
./pipeline.cmd
```

For RNN:
```powershell
./pipeline_rnn.cmd
```

These two scripts will automatically sample from TFIM and generate the training data. Then it will also run training. For testing, please sample from ./MPS_POVM_sampler/noisygeneration.py again with exact same parameters (noise strength, number of qubits, and so on...)

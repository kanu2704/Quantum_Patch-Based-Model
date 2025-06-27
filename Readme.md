# OptimisedQCNN: A Hierarchical Quantum-Classical CNN for MNIST

## 1. Project Overview
This Jupyter notebook implements an **Optimised Quantum Convolutional Neural Network (QCNN)** hybridised with classical preprocessing to classify MNIST digits. The core workflow:

1. **Extract image patches** at two levels (12×12 → 6×6 → 3×3).  
2. **Encode** each patch into a small quantum circuit with trainable parameters.  
3. **Aggregate** quantum measurement outputs into a classical feature vector.  
4. **Train** end-to-end via TensorFlow, minimising expensive quantum simulator calls.

---

## 2. Requirements & Installation
Install dependencies at the top of the notebook:

```bash
pip install qiskit==1.4.2 tensorflow numpy pandas matplotlib pennylane silence-tensorflow
pip install pennylane-qiskit qiskit-ibm-runtime qiskit-aer opencv-python
```

- **TensorFlow**: classical training loop and data handling  
- **Pennylane + Qiskit**: quantum circuit definitions & simulation  
- **OpenCV & NumPy**: efficient image resizing & patch extraction  
- **Matplotlib**: plotting  
- **silence-tensorflow**: suppress TF warnings  

---

## 3. Imports & Setup
```python
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import cv2, numpy as np, warnings
from silence_tensorflow import silence_tensorflow
import pennylane as qml

silence_tensorflow()
tf.keras.backend.set_floatx('float32')
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
```
- Set TF to use **float32** for speed.  
- Silence unwanted TF warnings.  
- Initialize Pennylane for quantum operations.

---

## 4. Hyperparameters & Quantum Device
```python
patch_size_level1 = 12
n_qubits = 4
n_layers = 2
dev = qml.device("default.qubit", wires=n_qubits)
```
- **`patch_size_level1`** defines the first-level patch size.  
- **`n_qubits`**, **`n_layers`** configure circuit capacity.  
- Use the local `"default.qubit"` simulator for fast batching.

---

## 5. Quantum Circuit Definition
```python
@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    # Input encoding via RY rotations
    for i, x in enumerate(inputs):
        qml.RY(x, wires=i)
    # Variational layers: trainable RZ + ring entanglement
    for l in range(n_layers):
        for i in range(n_qubits):
            qml.RZ(weights[l, i], wires=i)
        qml.broadcast(qml.CNOT, wires=range(n_qubits), pattern="ring")
    # Measure expectation of Pauli-Z on each qubit
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
```
- **Angle encoding**: map scaled pixel values to rotation angles.  
- **Variational layers**: alternate trainable RZ rotations with CNOT entanglement in a ring.  
- **Output**: list of expectation values (one per qubit).

---

## 6. Data Loading & Preprocessing
```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = resize_images(x_train, size=(12, 12))
x_test  = resize_images(x_test,  size=(12, 12))
```
- Normalize pixel values to [0,1].  
- Resize 28×28 images to 12×12 using OpenCV for patch extraction.

---

## 7. Patch Extraction Utilities
```python
def extract_all_patches_vectorized(images, patch_size):
    # Uses NumPy stride tricks to yield non-overlapping patches
    ...
```
- Extracts **all** level-1 patches at once into shape `(N, n_patches, patch_area)`  
- Vectorised for performance; avoids Python loops during training.

---

## 8. Hierarchical Forward Pass
```python
def forward_two_levels(img_idx, patches_level1):
    # Level-1: quantum encode 36 patches → 36 × n_qubits outputs
    l1_outputs = process_patches_batch(patches_level1, weights_level1)

    # Aggregate into 6×6 grid, create 9 super-patches
    l2_inputs = aggregate_to_level2(l1_outputs)
    # Level-2: quantum encode 9 super-patches
    l2_outputs = process_patches_batch(l2_inputs, weights_level2)

    # Concatenate level-2 outputs into final feature vector
    return tf.concat(l2_outputs, axis=0)
```
- **`process_patches_batch`** batches circuit calls for speed.  
- Two-level approach reduces quantum invocation count.

---

## 9. Training & Testing
```python
@tf.function
def train_step(img_idx, y_true):
    with tf.GradientTape() as tape:
        preds = forward_two_levels(img_idx, patches_level1[img_idx])
        loss  = loss_fn(y_true, classifier(preds))
    grads = tape.gradient(loss, all_weights)
    optimizer.apply_gradients(zip(grads, all_weights))

def test_model():
    # Forward pass for each test image, compute accuracy
    ...
```
- Use `@tf.function` for graph-mode acceleration.  
- Only **classical** weights are updated; quantum outputs treated as fixed.

---

## 10. Execution & Timing
```python
# Pre-extract all patches once
patches_level1 = extract_all_patches_vectorized(x_train, patch_size_level1)

# Training loop
start_time = time.time()
for epoch in range(epochs):
    for i in range(len(x_train)):
        train_step(i, y_train[i])
train_time = time.time() - start_time

# Testing
test_accuracy, test_time = test_model()
```
- Pre-extraction and batching minimise Python overhead.  
- Records total training and test durations.

---

## 11. Visualization
- **Training curves**: loss & accuracy vs. epoch using Matplotlib.  
- **Sample predictions**: display resized 12×12 inputs with predicted vs. true labels.  
- **Timing summary**: overlay total runtimes on a summary plot.

---

## 12. Optimization Highlights
- **Vectorised** patch extraction  
- **Batch** quantum circuit execution  
- Two-level **hierarchy** reduces circuit calls  
- Classical-quantum **hybrid** training in TensorFlow  
- Final section prints overall **accuracy** and **timings**.

---

## How to Run
1. Clone or download this notebook.  
2. Install the required packages (see section 2).  
3. Launch Jupyter and **run all cells** in order.  
4. Adjust `n_qubits`, `n_layers`, or patch sizes to experiment.

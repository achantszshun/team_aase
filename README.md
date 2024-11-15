# team_aase
**Team AASE - Deepview Competition**

## Project Summary
This project aims to reconstruct accurate 3D surface meshes from noisy 2D ultrasonic scan slices using deep learning. The pipeline includes data preprocessing, normalization, model training, and visualization. We leverage a 3D U-Net architecture to process volumetric data and generate high-fidelity surface reconstructions.

---


## **Setup Instructions**

### **Prerequisites**
Install the required packages:
```bash
pip install numpy pandas matplotlib torch torchvision scikit-image open3d
```


### **Folder Structure**
Ensure your repository has the following structure:
```
/volumes                 # Contains raw ultrasonic scan files (.raw)
/meshes                  # Contains ground truth mesh files (.ply)
/processed_scans         # Stores processed .npy files
/code
    ├── process_scans.ipynb
    ├── extract_data.py
    ├── simple_unet.py
    ├── model_2.ipynb
    ├── ply_visualizer.ipynb
    ├── raw_compressed_visualization.ipynb
```


### **Run the Pipeline**

#### **1. Preprocess Data**
Run `process_scans_in_batches()` from `process_scans.ipynb` to convert `.raw` files into `.npy` format.

#### **2. Extract Data Points**
Use `extract_point_data()` in `extract_data.py` to extract significant points.

#### **3. Train the Model**
Run the training pipeline in `model_2.ipynb`.

#### **4. Visualize Results**
Use `ply_visualizer.ipynb` and `raw_compressed_visualization.ipynb` for point cloud and slice visualizations.

### **1. Data Preprocessing and Normalization**
   - **Objective:** Convert `.raw` files into normalized `.npy` arrays for faster processing and prepare data for model training.
   - **Files:**
     - `process_scans.ipynb`: Batch processes `.raw` files into `.npy` format and stores them in the `/processed_scans` folder.
     - `extract_data.py`: Extracts significant points from the scan based on intensity thresholds.
     - `df.ipynb`: Converts scan data into structured dataframes for analysis.
   - **Steps:**
     1. Place `.raw` files in the `/volumes` folder.
     2. Run `process_scans_in_batches()` from `process_scans.ipynb` to generate `.npy` files.
     3. Use `extract_point_data()` and `extract_vertices()` functions from `extract_data.py` to extract and visualize key points.

### **2. Model Architecture**
   - **Objective:** Use a 3D U-Net to process voxel grids and reconstruct surface meshes.
   - **Files:**
     - `simple_unet.py`: Implements a 3D U-Net model.
     - `model_2.ipynb`: Sets up and trains the model.
   - **Steps:**
     1. Import the `SimpleUNet3D` class from `simple_unet.py`.
     2. Prepare voxel grids from processed data using `create_voxel_grid()` in `report.ipynb`.
     3. Train the model using the training pipeline in `model_2.ipynb`.

### **3. Training Pipeline**
   - **Objective:** Train the 3D U-Net on prepared datasets and save trained weights.
   - **Files:**
     - `model_2.ipynb`: Contains training code.
   - **Steps:**
     1. Load the processed data and convert it into PyTorch tensors.
     2. Define the loss function (MSE) and optimizer (Adam).
     3. Train the model:
        ```python
        num_epochs = 10
        for epoch in range(num_epochs):
            # Training loop here
        ```
     4. Save the model using:
        ```python
        torch.save(model.state_dict(), "model_weights.pth")
        ```

### **4. Visualization**
   - **Objective:** Visualize reconstructed surfaces and compare them with ground truths.
   - **Files:**
     - `ply_visualizer.ipynb`: Visualizes point clouds and meshes.
     - `raw_compressed_visualization.ipynb`: Displays 2D slices and downsampled data.
     - `ply_function.ipynb`: Generates point clouds from `.ply` files.
   - **Steps:**
     1. Use `plot_point_cloud()` in `ply_visualizer.ipynb` to visualize point clouds.
     2. Use `plot_raw_downsampled_data()` in `raw_compressed_visualization.ipynb` to visualize downsampled slices.
     3. Generate final 3D meshes using the `marching_cubes` algorithm:
        ```python
        verts, faces, _, _ = measure.marching_cubes(reconstructed_data, level=threshold)
        ```

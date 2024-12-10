# INSTALL.md

## Instructions to Install and Setup SeamlessM4T for Fine-tuning on Indic Data

Follow these steps to set up SeamlessM4T for fine-tuning on Indic data.

---

### 1. Clone the Repository

```bash
git clone https://github.com/AshwinSankar17/seamless_communication.git
cd seamless_communication
```

---

### 2. Set Up the Environment

#### Using Conda

1. Create and activate a Conda environment from the `environment.yml` file:
    ```bash
    conda env create -f environment.yml
    conda activate seamless
    ```

2. Install the package in editable mode:
    ```bash
    pip install -e .
    ```

---

### 3. Navigate to the Fine-tuning Directory

Move to the directory where fine-tuning scripts are located:
```bash
cd src/seamless_communication/cli/m4t/finetune/
```

---

### 4. Prepare Datasets

#### Update Variables in `prepare_datasets.sh`

1. Open `prepare_datasets.sh` in your preferred text editor:
    ```bash
    nano prepare_datasets.sh
    ```
    
2. Replace the placeholders with appropriate values:
    - `HF_TOKEN`: Your Hugging Face token.
    - `save_dir`: Directory to save prepared datasets.
    - `hf_cache_dir`: Cache directory for Hugging Face datasets.

3. Save and close the file.

#### Run the Script
Prepare datasets by running the script:
```bash
bash prepare_datasets.sh
```

---

### 5. Prepare Indic Voices Data

#### Update Variables in `prep_indicvoices.sh`

1. Open `prep_indicvoices.sh` in your preferred text editor:
    ```bash
    nano prep_indicvoices.sh
    ```

2. Replace the placeholders with appropriate paths:
    - `LOCAL_PATH`: Path to your local Indic voice data.
    - `OUTPUT_PATH`: Directory to save processed Indic voice data.

3. Save and close the file.

#### Run the Script
Prepare Indic voices data by running the script:
```bash
bash prep_indicvoices.sh
```

---

### 6. Configure Fine-tuning

#### Update `run_finetune.sh`

1. Open `run_finetune.sh` in your preferred text editor:
    ```bash
    nano run_finetune.sh
    ```

2. Add paths to your training and testing manifest files:
    - Update `train_manifest.json` paths for training datasets.
    - Update `test_manifest.json` paths for evaluation datasets.

3. Ensure that all required configurations are correctly set.

---

### 7. Start Fine-tuning

1. Start the fine-tuning process with the following command:
    ```bash
    WANDB_API_KEY=<your_wandb_api_key> bash run_finetune.sh
    ```

2. Replace `<your_wandb_api_key>` with your Weights & Biases API key.

---

### Notes

- Ensure all dependencies are properly installed via Conda and `pip`.
- Verify that the paths and tokens are valid for seamless operation.
- Logs and metrics will be tracked using Weights & Biases. Ensure your API key is active.

For further details, refer to the documentation or project README.
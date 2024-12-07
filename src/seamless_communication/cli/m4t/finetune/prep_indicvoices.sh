#!/bin/bash

# Configurations
REMOTE_PATH="e2e/asr-transcription/final_releases/"  # Change to your remote folder path
LOCAL_PATH="/data/indicvoices/iv_full_data"                      # Change to your local folder path
PYTHON_SCRIPT="create_indicvoices.py"    
OUTPUT_PATH="/data/indicvoices/iv_prep_data"                    # Python script to run after extraction

if [[ -d "$LOCAL_PATH" ]]; then
    echo "Local path exists: $LOCAL_PATH"
else
    echo "Local path does not exist. Creating: $LOCAL_PATH"
    mkdir -p "$LOCAL_PATH"
    mkdir -p "$OUTPUT_FOLDER"
fi

# # Step 1: Download contents of the remote folder using mc cp
# echo "Downloading contents from $REMOTE_PATH to $LOCAL_PATH..."
# mc cp --recursive "$REMOTE_PATH" "$LOCAL_PATH"

# if [[ $? -ne 0 ]]; then
#     echo "Error: Failed to download files. Exiting."
#     exit 1
# fi
# echo "Download complete."

# Step 2: Untar all .tar files in the local folder
echo "Extracting .tar files in $LOCAL_PATH..."
for FILE in "$LOCAL_PATH"/*.tgz*; do
    if [[ -f "$FILE" ]]; then
        echo "Extracting: $FILE"
        tar -xvzf "$FILE" -C "$LOCAL_PATH"
        if [[ $? -eq 0 ]]; then
            echo "Successfully extracted: $FILE"
            rm "$FILE"  # Optional: Remove tar file after extraction
        else
            echo "Failed to extract: $FILE"
        fi
    fi
done
echo "Extraction complete."

#Step 3: Downsample wav files
# find "$LOCAL_PATH" -type f \( -name "*.wav" \) -print0 | xargs -0 -I {} -P 128 bash -c 'ffmpeg -y -loglevel warning -hide_banner -stats -i $1 -ar $2 -ac $3 "${1%.*}_${2}.wav" && rm $1 && mv "${1%.*}_${2}.wav" $1' -- {} 16000 1

# Step 3: Run the Python script
echo "Running Python script: $PYTHON_SCRIPT"
python3 "$PYTHON_SCRIPT" --input_folder $LOCAL_PATH --output_folder $OUTPUT_PATH

if [[ $? -ne 0 ]]; then
    echo "Error: Python script execution failed."
    exit 1
fi
echo "Python script executed successfully."

echo "All steps completed successfully."

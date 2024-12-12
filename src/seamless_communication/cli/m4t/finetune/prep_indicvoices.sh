#!/bin/bash

# Configurations
REMOTE_PATH="e2e/asr-transcription/final_releases/"  # Change to your remote folder path
LOCAL_PATH="/data/iv"                      # Change to your local folder path
PYTHON_SCRIPT="create_indicvoices.py"    
OUTPUT_PATH="/data/indicvoices/iv_prep_data"                    # Python script to run after extraction

# Ensure LOCAL_PATH exists
if [[ -d "$LOCAL_PATH" ]]; then
    echo "Local path exists: $LOCAL_PATH"
else
    echo "Local path does not exist. Creating: $LOCAL_PATH"
    mkdir -p "$LOCAL_PATH"
fi

# Ensure OUTPUT_PATH exists
if [[ -d "$OUTPUT_PATH" ]]; then
    echo "Output folder exists: $OUTPUT_PATH"
else
    echo "Output folder does not exist. Creating: $OUTPUT_PATH"
    mkdir -p "$OUTPUT_PATH"
fi

# # Step 1: Download contents of the remote folder using mc cp
echo "Downloading contents from $REMOTE_PATH to $LOCAL_PATH..."
mc ls $REMOTE_PATH | awk '{print $NF}' | parallel -j16 --bar mc cp $REMOTE_PATH/{} $LOCAL_PATH/{}

# Initialize a flag to track if any file is missing
missing_flag=0

# Check each file from the source list against the local destination
echo "Verifying files..."
mc ls $REMOTE_PATH | awk '{print $NF}' | while read -r file; do
    if [ ! -f $LOCAL_PATH/$file ]; then
        echo "Missing: $file"
        missing_flag=1
    fi
done

# Exit if any file is missing
if [ "$missing_flag" -eq 1 ]; then
    echo "Verification failed. Some files are missing."
    exit 1
fi

echo "All files downloaded successfully."

# Step 2: Untar all .tar files in the local folder
echo "Extracting .tar files in $LOCAL_PATH..."
# Function to extract and optionally delete the archive
extract_tgz() {
    local file="$1"
    echo "Processing: $file"
    if tar -xzf "$file" -C "$LOCAL_PATH"; then
        echo "Extraction successful: $file"
        rm "$file"  # Optional: Remove tar file after extraction
    else
        echo "Extraction failed: $file"
    fi
}
find "$LOCAL_PATH" -type f -name "*.tgz*" | parallel --no-notice -j 16 --bar extract_tgz
echo "Extraction complete."

#Step 3: Downsample wav files
find "$LOCAL_PATH" -type f \( -name "*.wav" \) -print0 | xargs -0 -I {} -P 128 bash -c 'ffmpeg -y -loglevel warning -hide_banner -stats -i $1 -ar $2 -ac $3 "${1%.*}_${2}.wav" && rm $1 && mv "${1%.*}_${2}.wav" $1' -- {} 16000 1

# Step 3: Run the Python script
echo "Running Python script: $PYTHON_SCRIPT"
python3 "$PYTHON_SCRIPT" --input_folder $LOCAL_PATH --output_PATH $OUTPUT_PATH

if [[ $? -ne 0 ]]; then
    echo "Error: Python script execution failed."
    exit 1
fi
echo "Python script executed successfully."

# Iterate through all language folders
for lang_dir in "$OUTPUT_PATH"/*/; do
    echo "Processing language folder: $lang_dir"
    
    # Paths for training and validation manifests
    train_manifest="$lang_dir/train_manifest.json"
    valid_manifest="$lang_dir/valid_manifest.json"

    # Initialize or clear the manifest files
    > "$train_manifest"
    > "$valid_manifest"

    # Collate train transcripts
    # find "$lang_dir" -type d -path "*/train/transcripts" -exec find {} -type f -name "*.json" -exec cat {} + >> "$train_manifest" \;
    find $lang_dir -type f -wholename "*/train/transcripts/*.json" -exec cat {} >> $train_manifest \;&
    # Collate valid transcripts
    find $lang_dir -type f -wholename "*/valid/transcripts/*.json" -exec cat {} >> $valid_manifest \;&
    # find "$lang_dir" -type d -path "*/valid/transcripts" -exec find {} -type f -name "*.json" -exec cat {} + >> "$valid_manifest" \;

    # echo "Manifests created: $train_manifest, $valid_manifest"
done
wait

train_manifest="$OUTPUT_PATH/train_manifest.json"
valid_manifest="$OUTPUT_PATH/valid_manifest.json"

# Initialize or clear the manifest files
> "$train_manifest"
> "$valid_manifest"

find $OUTPUT_PATH -type f -wholename "*/train_manifest.json" -exec cat {} >> $train_manifest \;
find $OUTPUT_PATH -type f -wholename "*/valid_manifest.json" -exec cat {} >> $valid_manifest \;

echo "All steps completed successfully."

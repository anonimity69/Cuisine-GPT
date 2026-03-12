import os
import json
import time
import ollama
from tqdm import tqdm

DATA_DIR = "Data"
OUTPUT_DIR = "Output"
MODEL_NAME = "qwen3:4b" 

def generate_qna(context_text, source_file, max_retries=3):
    prompt = f"""
    Context: {context_text}
    
    Based on the text above, generate exactly 3 benchmark question and answer pairs.
    Return the result as a JSON list of objects.
    
    Required Keys:
    - 'Query': The question.
    - 'Gold Answer': Detailed answer including cuisine type.
    - 'Source/Reference': Use a link from the text if available, otherwise use: {source_file}.
    """
    
    for attempt in range(max_retries):
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{'role': 'user', 'content': prompt}],
                format='json' 
            )
            content = response['message']['content']
            return json.loads(content)
        except Exception:
            time.sleep(1)
    return []

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # First, collect all files so we can count them for the progress bar
    tasks = []
    folders = ["Blog", "Wikibooks", "Wikipedia"]
    for folder in folders:
        folder_path = os.path.join(DATA_DIR, folder)
        if os.path.exists(folder_path):
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(".json"):
                        tasks.append((folder, root, file))

    print(f"Starting Q&A generation for {len(tasks)} files using {MODEL_NAME}...")

    # tqdm progress bar
    for folder, root, file in tqdm(tasks, desc="Generating Benchmarks", unit="file"):
        input_file_path = os.path.join(root, file)
        
        with open(input_file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                # Cap context to prevent 4B model from getting overwhelmed
                text_content = json.dumps(data)[:12000] 
                
                qna_output = generate_qna(text_content, file)
                
                if not qna_output:
                    continue
                    
                filename_without_ext = os.path.splitext(file)[0]
                output_filename = f"{folder}_{filename_without_ext}_qa.json"
                output_file_path = os.path.join(OUTPUT_DIR, output_filename)
                
                output_data = {
                    "source": input_file_path,
                    "qna_pairs": qna_output
                }
                
                with open(output_file_path, "w", encoding="utf-8") as out_f:
                    json.dump(output_data, out_f, indent=4)
                    
            except Exception as e:
                # Log errors to a file instead of breaking the progress bar
                with open("error_log.txt", "a") as err_log:
                    err_log.write(f"Error in {input_file_path}: {str(e)}\n")

if __name__ == "__main__":
    main()
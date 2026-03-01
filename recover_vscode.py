import os
import json
import shutil
import urllib.parse

history_dir = os.path.expandvars(r'%APPDATA%\Code\User\History')
project_root = r'C:\Users\rawat\Desktop\New folder\Grotesque ai'

for d in os.listdir(history_dir):
    entries_file = os.path.join(history_dir, d, 'entries.json')
    if not os.path.isfile(entries_file):
        continue
    
    try:
        with open(entries_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        resource = data.get('resource', '')
        if 'Grotesque%20ai' in resource or 'Grotesque ai' in resource:
            # Decode URI encoding
            file_path = urllib.parse.unquote(resource).replace('file:///', '').replace('/', '\\')
            
            # Use the latest entry
            latest_entry = data['entries'][-1]['id']
            source_file = os.path.join(history_dir, d, latest_entry)
            
            if os.path.isfile(source_file):
                print(f"Restoring {file_path}")
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                shutil.copy2(source_file, file_path)
    except Exception as e:
        print(f"Failed to process {d}: {e}")

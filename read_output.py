import sys
try:
    with open('scan_output.txt', 'r', encoding='utf-16') as f:
        lines = f.readlines()
        for line in lines[-50:]:
            print(line.strip())
except Exception as e:
    print(f"Error: {e}")

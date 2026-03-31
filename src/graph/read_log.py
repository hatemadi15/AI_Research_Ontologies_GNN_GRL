try:
    with open('final_debug.txt', 'r', encoding='utf-8') as f:
        content = f.read()
except:
    try:
        with open('final_debug.txt', 'r', encoding='mbcs') as f:
            content = f.read()
    except:
        with open('final_debug.txt', 'rb') as f:
            content = f.read().decode('utf-16-le', errors='ignore')

print("--- TAIL OF LOG ---")
print(content[-1000:])

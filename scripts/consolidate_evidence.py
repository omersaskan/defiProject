import os

files = [
    'evidence_1.out',
    'evidence_2.out',
    'evidence_3.out',
    'evidence_4.out',
    'evidence_5.out',
    'training_proof.log'
]

with open('evidence_consolidated.txt', 'w', encoding='utf-8') as fout:
    for f in files:
        if os.path.exists(f):
            fout.write(f"\n{'='*20} {f} {'='*20}\n")
            # Try multiple encodings for Windows redirection artifacts
            content = ""
            for enc in ['utf-16-le', 'utf-16', 'utf-8']:
                try:
                    with open(f, 'r', encoding=enc, errors='replace') as fb:
                        content = fb.read()
                        break
                except:
                    continue
            fout.write(content)
        else:
            fout.write(f"\nFile {f} not found.\n")

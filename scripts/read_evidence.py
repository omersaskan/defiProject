import os

files = [
    'evidence_1.out',
    'evidence_2.out',
    'evidence_3.out',
    'evidence_4.out',
    'evidence_5.out',
    'training_proof.log'
]

for f in files:
    if os.path.exists(f):
        print(f"\n{'='*20} {f} {'='*20}")
        try:
            # Try utf-16le first for redirected windows output
            with open(f, 'r', encoding='utf-16-le', errors='replace') as fb:
                print(fb.read())
        except:
            with open(f, 'r', encoding='utf-8', errors='replace') as fb:
                print(fb.read())
    else:
        print(f"\nFile {f} not found.")

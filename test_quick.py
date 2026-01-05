
# Quick test to verify Genesis-FN installation.

import sys
import os

def test_imports():
    print("Testing Genesis-FN imports...")
    
    packages = [
        'numpy',
        'pandas', 
        'sklearn',
        'scipy',
        'torch',
        'transformers',
        'nltk'
    ]
    
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg}")
    
    print("\nTo run Genesis-FN:")
    print("python genesis_fn_complete.py --data_path train.tsv")
    print("\nFor faster testing (without BERT):")
    print("python genesis_fn_complete.py --data_path train.tsv --no_bert")

if __name__ == "__main__":
    test_imports()

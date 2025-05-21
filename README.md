# UMatchDIS

UMatchDIS is a Python wrapper script for image matching developed at Maastricht University (UM) Library. 
It is a wrapper around the [Deep Image Search (DIS)](https://pypi.org/project/DeepImageSearch/) library, which is a powerful tool for image similarity search and clustering.
> DeepImageSearch is a powerful Python library that combines state-of-the-art computer vision models for feature
> extraction with highly optimized algorithms for indexing and searching. This enables fast and accurate similarity search
> and clustering of dense vectors (...) 

_Read more on https://github.com/TechyNilesh/DeepImageSearch_

### Activate venv
```bash
source .venv/bin/activate
```

### Run with defaults
```bash
python3 umatchdis.py
```

### Run with own config
```bash
python3 umatchdis.py --config docker/my_uconfig.py
```
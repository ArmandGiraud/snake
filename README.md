# Snake
---
- Pure Python Snake
- Wall version only for now
- No external lib beyond numpy/cupy

### Play in jupyter snake_python:

## Player Mode
```python
from utils import Snake
sizes = (8, 6) # grid size
sn = Snake(sizes)
```

## Bot training Policy Gradient & numpy 

```bash
python snake_rl.py
```

## Display game every n iteration

```bash
python snake_rl.py -d
```

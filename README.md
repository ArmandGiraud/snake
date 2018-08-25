# Snake
---
- Pure Python Snake
- Wall version only for now
- No external lib beyond numpy/cupy

-- reward:
  - 10 grab a food piece
  - 1 did not crash

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


### Coming soon
- pytorch version with GPU support
- pygames interface
- new RL algos

----
Simple Grid format:


[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],                                                                                                                                                        
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],                                                                                                                                                        
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],                                                                                                                                                        
 [1, 0, 2, 2, 3, 0, 0, 0, 0, 1],                                                                                                                                                        
 [1, 0, 0, 0, 0, 0, 0, 0, 4, 1],                                                                                                                                                        
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],                                                                                                                                                        
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],                                                                                                                                                        
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]] 
 
 - 3: Snake head
 - 2: snake body
 - 4: snake food

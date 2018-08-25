# Snake
---
- Pure Python Snake
- Wall version only for now
- No external lib beyond numpy/cupy

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
## :bulb: Simple Grid format:


[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],                                                                                                                                                        
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],                                                                                                                                                        
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],                                                                                                                                                        
 [1, 0, 2, 2, 3, 0, 0, 0, 0, 1],                                                                                                                                                        
 [1, 0, 0, 0, 0, 0, 0, 0, 4, 1],                                                                                                                                                        
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],                                                                                                                                                        
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],                                                                                                                                                        
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]] 
 
 
 :paperclip: Digits meaning:
 
 - 1: wall, grid borders
 - 3: Snake head
 - 2: snake body
 - 4: snake food


 :paperclip: reward:
  - 10 grab a food piece
  - 1 did not crash


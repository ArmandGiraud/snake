import time
import math
import random
from IPython.display import clear_output
import copy
from pprint import pprint

class Grille(list):
    def __init__(self):
        pass
    
    def get_coords(self, coords):
        x = coords[1]
        y = coords[0]
        return self[y][x]
    
    def get_size(self):
        return len(self), len(self[0])
    
    def set_value(self,coords, value):
        x = coords[1]
        y = coords[0]
        self[y][x] = value


def build_grille(sizes):
    large, high = sizes
    if large < 1 or high < 1:
        print("WARNING: 0 sized or negative grid")
    if large < high:
        print("WARNING: grille should be larger than high")
    if large > 23 or high > 50:
        print("WARNING: grille is too big")
    grille = Grille()
    first_row = [1] * (large + 2)
    grille.append(first_row)

    for row in range(high):
        new_row = [1] + [0] * large + [1]
        grille.append(new_row)
    last_row  = [1] * (large + 2)
    grille.append(last_row)
    return grille


def initialize_snake(grille, sizes):
    large, high = sizes
    init_x =  math.ceil(large /2)
    init_y =  math.ceil(high/2)
    init_coords = (init_x, init_y)
    if grille[init_y][init_x] != 0:
        print("snake initialized on grid border")
    else:
        grille[init_y][3] = 3 # snake head
        grille[init_y][2] = 2  #snake body
        grille[init_y][1] = 2  #snake body
    return grille

input_to_dir = {
    4 :"left",
    6 :"right",
    8 :"up",
    5 :"down"
}

def run(grille_intialized, bot_mode = False, bot = None, n_food=1):

    history = {}
    history["snake"] = []
    history["grille"] = []

    last_head = find_head(grille_intialized)
    last_head = (last_head[0], last_head[1] - 1)
    grille = grille_intialized
    snake = initial_snake(grille.get_size()[0]-2)
    did_eat = True
    grille = spawn_food(grille, snake, n_food=n_food)
    score = 0
    while True:
        history["snake"].append(snake)
        g = copy.deepcopy(grille)
        history["grille"].append(g)

        #pprint(grille)
        current_head = find_head(grille)
        user_input = move(g, bot_mode, bot) #
        
        last_direction = define_movement(last_head, current_head)
        #print("last_direction", last_direction)
        if check_input(last_direction, user_input) == 1:
            user_input_corrected = "continue"
            # last direction does not change
        else:
            user_input_corrected = user_input
        if user_input_corrected == 8:
            last_direction = "up"
        if user_input_corrected == 6:
            last_direction = "right"
        if user_input_corrected == 4:
            last_direction = "left"
        if user_input_corrected == 5:
            last_direction = "down"

        #print("direction", last_direction)
        #print("user_input", user_input)
        grille, snake, score = continuing(last_direction,
                                          snake,
                                          grille,
                                          score)
        if not grille:
            #pprint(grille)
            #print("crash: Final Score is {}".format(score))
            yield score, g, user_input, True

        last_head = current_head
        #clear_output()
        #print("Score : {}".format(score))
        yield score, g, user_input, False

def initial_snake(high):
    # (x, y)
    head = (math.ceil((high -1 )/2), 3)
    
    snake = [(head[0], head[1] - 2), (head[0], head[1] - 1), head]
    return snake

def define_movement(last_head, current_head):
    var_row = last_head[0] - current_head[0]
    var_col = last_head[1] - current_head[1]
    if (var_row, var_col) == (0, 1):
        direction = "left"
    elif (var_row, var_col) == (0, -1):
        direction = "right"
    elif (var_row, var_col) == (-1, 0):
        direction = "down"
    elif (var_row, var_col) == (+1, 0):
        direction = "up"
    else:
        print('problem with directions')
    return direction

def check_input(last_direction, user_input):
    if user_input not in [4, 5, 6, 8, 9]:
        #print("impossible")
        return 1
    
    user_input = input_to_dir[user_input]
    if last_direction == user_input:
        #print("impossible")
        return 1
    elif last_direction == "left" and user_input == "right":
        #print("impossible")
        return 1
    elif last_direction == "right" and user_input == "left":
        #print("impossible")
        return 1
    elif last_direction == "up" and user_input == "down":
        #print("impossible")
        return 1
    elif last_direction == "down" and user_input == "up":
        #print("impossible")
        return 1
    else:
        #good input
        return 0

def continuing(last_direction, snake, grille, score):
    # 1: modify grille
    # replace value of new head by a 3
    # add coords of directions to coord of head to find new head
    score=0
    did_eat = False
    new_head = add_tuples(snake[-1] ,  dir_to_coord[last_direction])
    # crash
    if grille.get_coords(new_head) != 0 and grille.get_coords(new_head) != -9:
        return None, None, score
    # did eat
    if grille.get_coords(new_head) == -9:
        did_eat = True
        
    grille.set_value(new_head, 3) # position new head
    grille.set_value(snake[-1], 2) # replace old head with body
    
    if did_eat:
        pass
    else:
        grille.set_value(snake[0], 0) # replace queue of snake by 0
    # 2 modify snake
    def shift(l, n):
        return l[n:] + l[:n] # shift list to left

    if did_eat:
        queue = snake[0]
        snake = [queue] + shift(snake, 1)
        snake[-1] = new_head
    else:
        snake = shift(snake, 1)
        snake[-1] = new_head
    
    if did_eat:
        score = 1
        grille = spawn_food(grille, snake)
    else:
        score = 0  
    return grille, snake, score



dir_to_coord = dict(
    right = (0, 1),
    left = (0, -1),
    up = (-1, 0),
    down = (+1, 0)
)
def add_tuples(a, b):
    return tuple((sum(a) for a in zip(a,b)))

def find_snake(grille):
    snake = {}
    snake["body"] = []
    for i , rows in enumerate(grille):
        for j, value in enumerate(rows):
            if value == 3:
                snake["head"] = (i, j)
            if value == 2:
                snake["body"].append((i,j))
    return snake

def find_head(grille):
    return find_snake(grille)['head']

#last_head = find_head(grille_intialized)
#snake = find_snake(grille_intialized)             

def spawn_food(grille, snake, n_food=1):
    num_food=0
    while True:
        sizes = grille.get_size()
        r = random.randint(a=1, b= sizes[0] - 2 )
        c = random.randint(a=1, b= sizes[1] - 2 )
        food_spawn = (r,c)
        if food_spawn not in snake:
            grille.set_value(food_spawn, -9)
            num_food +=1
        if num_food == n_food:
            return grille

def move(state, bot_mode = False, bot = None):
    if bot_mode:
        # time.sleep(1.2) add to see the bot move
        user_input = bot.compute(state)
    else:
        user_input = int(input('Move:'))
    return user_input

class Snake():
    def __init__(self, sizes, bot = None, n_food=1):
        self.bot = bot
        self.sizes = sizes
        self.n_food = n_food
    
    def play(self):
        bot_mode = False
        if self.bot:
            bot_mode = True
        for score, grille, user_input, is_done in run(self.initial_grille, bot_mode, self.bot, self.n_food):
            yield score, grille, user_input, is_done
            if is_done:
                break

    
    def reset(self):
        grille = build_grille(self.sizes)
        self.initial_grille = initialize_snake(grille, self.sizes)
        initial_grille = copy.deepcopy(self.initial_grille)
        return initial_grille



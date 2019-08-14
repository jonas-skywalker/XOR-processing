import neuralnetwork
import random
import my_matrix_lib as matrix


nn = neuralnetwork.load_json("XOR_data.json")
step = 10


def setup():
    noStroke()
    size(500, 500)
    
    
    
def draw():
    noStroke()
    background(255)
    for x in range(0, width, step):
        for y in range(0, height, step):
            x_map = map(x, 0, width, 0, 1)
            y_map = map(y, 0, height, 0, 1)
            result = nn.feed_forward([x_map, y_map])
            data = result.matrix_data[0][0]
            point_alpha = map(data, 0, 1, 0, 255)
            fill(point_alpha)
            rect(x, y, step, step)
    noLoop()
        

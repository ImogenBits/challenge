from PIL import Image
from tqdm import tqdm

image = Image.open("SECRET.png").convert("F")
width, height = image.size
result = Image.new("F", (width, height))

ROUNDS = 32

for i in tqdm(range(width)):
    for j in range(height):
        value = 0
        di, dj = 1337, 42
        for k in range(ROUNDS):
            di, dj = (di * di + dj) % width, (dj * dj + di) % height
            value += image.getpixel(((i + di) % width, (j + dj + (i + di)//width) % height))
        result.putpixel((i, j), value / ROUNDS)

result = result.convert("RGB")
result.save("out.png")


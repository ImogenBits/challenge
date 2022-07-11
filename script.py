from PIL import Image
from tqdm import tqdm


image = Image.open("out.png").convert("F")
width, height = image.size
result = Image.new("F", (width, height))
ROUNDS = 32

for i in tqdm(range(width)):
    for j in range(height):
        di, dj = 1337, 42
        val = image.getpixel((i, j))
        for k in range(ROUNDS):
            di, dj = (di * di + dj) % width, (dj * dj + di) % height
            pos = ((i + di) % width, (j + dj + (i + di)//width) % height)
            c = result.getpixel(pos)
            result.putpixel(pos,c + val / ROUNDS)

result = result.convert("RGB")
result.save("decoded.png")

from tqdm import tqdm
from sympy.polys import Poly
from sympy import symbols, gcd
from sympy.discrete.transforms import fft, ifft
from PIL import Image
from numpy import asarray, ndarray, reshape

SCALE = 32
SIZE = 512 // SCALE
INDICES = 32 // SCALE

def check_circulant():
    rows = []
    for j in tqdm(range(SIZE)):
        for i in range(SIZE):
            row = []
            di, dj = 1337, 42
            for _ in range(INDICES):
                di, dj = (di * di + dj) % SIZE, (dj * dj + di) % SIZE
                row.append(((i + di) % SIZE, (j + dj + (i + di)//SIZE) % SIZE))
            rows.append(sorted([x + y * SIZE for (x, y) in row]))

    res = []
    for r in tqdm(range(SIZE * SIZE - 1)):
        res.append(rows[r] == [(i - 1) % (SIZE * SIZE) for i in rows[r + 1]])

    print(any(res))


def enc_coords(size: int, indices: int) -> list[tuple[int, int]]:
    out = []
    i, j = 0, 0
    di, dj = 1337, 42
    for _ in range(indices):
        di, dj = (di * di + dj) % size, (dj * dj + di) % size
        x, y = ((i + di) % size, (j + dj + (i + di)//size) % size)
        out.append((x, y))
    return out


def circulant_vec(size: int, indices: int) -> list[float]:
    sequence = [0.] * size * size
    for x, y in enc_coords(size, indices):
        sequence[x + y * size] = 1

    sequence.append(sequence[0])
    sequence.pop(0)
    return sequence[::-1]


def assoc_poly():
    sequence = circulant_vec()
    x = symbols("x")
    f = Poly(sequence[::-1], x)
    cg = [0] * (SIZE * SIZE + 1)
    cg[0] = 1
    cg[-1] = -1
    g = Poly(cg, x)
    
    print("gcd")
    res = gcd(f, g)
    print(res)


def circ_inv_mul(matrix: list[float], vec: list[float] | ndarray) -> list[float]:
    num = fft(vec)
    denom = fft(matrix)
    frac = [x/y for x, y in zip(num, denom)]
    return ifft(frac)


def encode(image: Image.Image) -> Image.Image:
    width, height = image.size
    result = Image.new("F", (width, height))
    for i in tqdm(range(width)):
        for j in range(height):
            value = 0
            di, dj = 1337, 42
            for _ in range(INDICES):
                di, dj = (di * di + dj) % width, (dj * dj + di) % height
                value += image.getpixel(((i + di) % width, (j + dj + (i + di)//width) % height))
            result.putpixel((i, j), value / INDICES)
    return result


def decode(image: Image.Image) -> Image.Image:
    matrix = circulant_vec()
    image_arr = asarray(image)
    shape = image_arr.shape
    image_arr = image_arr.flatten()
    decoded_arr = circ_inv_mul(matrix, image_arr)
    decoded_arr = reshape(decoded_arr, shape)
    decoded = Image.fromarray(decoded_arr)
    return decoded.convert("RGB")


if __name__ == "__main__":
    image = Image.open("SECRET.png")
    



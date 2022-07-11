from dataclasses import dataclass
from typing import Callable, Concatenate, ParamSpec, TypeVar
from tqdm import tqdm
from sympy.polys import Poly
from sympy import symbols, gcd
from PIL import Image
from numpy import asarray, ndarray, real, reshape
from numpy.typing import ArrayLike
from numpy.fft import fft, ifft
from datetime import datetime

SCALE = 2 ** 0
SIZE = 512 // SCALE
INDICES = 32 // SCALE


P = ParamSpec("P")
R = TypeVar("R")
def convert_image(func: Callable[Concatenate[ArrayLike, int, P], ArrayLike]) -> Callable[Concatenate[Image.Image, P], Image.Image]:
    def ret(image: Image.Image, *args: P.args, **kwargs: P.kwargs) -> Image.Image:
        image_arr = asarray(image)
        shape = image_arr.shape

        decoded_arr = func(image_arr.flatten(), image.size[0], *args, **kwargs)
        decoded_arr = reshape(decoded_arr, shape)

        return Image.fromarray(decoded_arr)
    return ret

@dataclass
class timed:
    name: str

    def __enter__(self):
        self.start = datetime.now()
    
    def __exit__(self, _t, _v, _tr):
        print(f"{self.name}: {(datetime.now() - self.start).total_seconds()}")


def enc_coords(size: int, indices: int) -> list[tuple[int, int]]:
    out = []
    di, dj = 1337, 42
    for _ in range(indices):
        di, dj = (di * di + dj) % size, (dj * dj + di) % size
        x, y = (di % size, (dj + di//size) % size)
        out.append((x, y))
    return out


def circulant_vec(size: int, indices: int) -> list[float]:
    sequence = [0.] * size * size
    for x, y in enc_coords(size, indices):
        sequence[x + y * size] = 1 / INDICES

    sequence.append(sequence[0])
    sequence.pop(0)
    return sequence[::-1]


def invertible(matrix: list[float]) -> bool:
    x = symbols("x")
    f = Poly(matrix[::-1], x)
    cg = [0] * (SIZE * SIZE + 1)
    cg[0] = 1
    cg[-1] = -1
    g = Poly(cg, x)
    
    res: Poly = gcd(f, g)
    return res.degree()


def circ_inv_mul(matrix: ArrayLike, vec: ArrayLike) -> ArrayLike:
    num = fft(vec)
    denom = fft(matrix)
    frac = [x/y for x, y in zip(num, denom)]
    res = ifft(frac)
    return real(res)


def encode(image: Image.Image) -> Image.Image:
    width, height = image.size
    image_arr = asarray(image)
    result = ndarray((width, height))
    offsets = enc_coords(width, INDICES)
    for y in tqdm(range(width)):
        for x in range(height):
            value = 0
            for dx, dy in offsets:
                value += image_arr[((y + dy + (x + dx)//width) % height), (x + dx) % width]
            result[y, x] = value / INDICES
    return Image.fromarray(result)


@convert_image
def encode_fast(image: ArrayLike, size: int) -> ArrayLike:
    matrix = circulant_vec(size, INDICES)
    matrix = fft(matrix)
    image = fft(image)
    ret = ifft(matrix * image)
    return real(ret)


@convert_image
def decode(image: ArrayLike, size: int) -> ArrayLike:
    matrix = circulant_vec(size, INDICES)
    return circ_inv_mul(matrix, image)


if __name__ == "__main__":
    image = Image.open("SECRET.png").crop((0, 0, SIZE, SIZE)).convert("F")
    image.convert("RGB").save("cropped.png")

    #with timed("encode"):
    #    encoded = encode(image)
    #encoded.convert("RGB").save("encoded.png")

    with timed("encode fast"):
        encoded_fast = encode_fast(image)
    encoded_fast.convert("RGB").save("encoded_fast.png")

    with timed("decode"):
        decoded = decode(encoded_fast)
    decoded.convert("RGB").save("decoded.png")
    



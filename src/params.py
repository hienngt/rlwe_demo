from pathlib import Path
from dataclasses import dataclass

from src.rlwe_concept import RingLWECrypto
from src.image_processor import ImageProcessor

BASE_DIR = Path(__file__).resolve().parent


@dataclass
class Params:
    n: int = 128
    q: int = 7681
    sigma: float = 0.2

    mode = 'RGB'

    input_image_path = BASE_DIR / "data/test3.jpg"
    encrypted_output_path = BASE_DIR / "data/encrypted_output.jpg"
    decrypted_output_path = BASE_DIR / "data/decrypted_output.jpg"


params = Params()

# Initialize the Ring-LWE crypto system
crypto = RingLWECrypto(n=params.n, q=params.q, sigma=params.sigma)
image = ImageProcessor(path=params.input_image_path,
                       new_width=params.n, mode=params.mode)

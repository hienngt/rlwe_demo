from matplotlib import pyplot as plt

from src.params import Params
from src.rlwe_concept import RingLWECrypto
from src.image_processor import ImageProcessor
from test import check_sspi
import pandas as pd


result = []
for n in range(100, 1000, 10):
    params = Params(n=n)
    # Initialize the Ring-LWE crypto system
    crypto = RingLWECrypto(n=params.n, q=params.q, sigma=params.sigma)
    image = ImageProcessor(path=params.input_image_path,
                           new_width=params.n, mode=params.mode)
    result_tmp = check_sspi(crypto, image, params)
    result_tmp.append(n)
    result.append(result_tmp)

df = pd.DataFrame(result, columns=['OrgVsEnc', 'OrgVsDec', 'DecVsEnc', 'n'])

plt.figure(figsize=(10, 6))

# Plot each SSIM comparison line
plt.plot(df['n'], df['OrgVsEnc'], label='OrgVsEnc')
plt.plot(df['n'], df['OrgVsDec'], label='OrgVsDec')
plt.plot(df['n'], df['DecVsEnc'], label='DecVsEnc')

plt.title('SSIM Comparison')
plt.xlabel('n')
plt.ylabel('SSIM Value')
plt.legend()
plt.grid(True)
plt.ylim(0, 1)  # Adjust based on your data range
plt.show()
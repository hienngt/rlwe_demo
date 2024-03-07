from matplotlib import pyplot as plt

from src.params import Params
from src.rlwe_concept import RingLWECrypto
from src.image_processor import ImageProcessor
from test import check_sspi
import pandas as pd


result = []
for sigma in range(0, 500, 20):
    sigma = sigma/10.0
    params = Params(sigma=sigma)
    # Initialize the Ring-LWE crypto system
    crypto = RingLWECrypto(n=params.n, q=params.q, sigma=params.sigma)
    image = ImageProcessor(path=params.input_image_path,
                           new_width=params.n, mode=params.mode)
    result_tmp = check_sspi(crypto, image, params)
    result_tmp.append(sigma)
    result.append(result_tmp)

df = pd.DataFrame(result, columns=['OrgVsEnc', 'OrgVsDec', 'DecVsEnc', 'sigma'])

plt.figure(figsize=(10, 6))

# Plot each SSIM comparison line
plt.plot(df['sigma'], df['OrgVsEnc'], label='OrgVsEnc')
plt.plot(df['sigma'], df['OrgVsDec'], label='OrgVsDec')
plt.plot(df['sigma'], df['DecVsEnc'], label='DecVsEnc')

plt.title('SSIM Comparison')
plt.xlabel('sigma')
plt.ylabel('SSIM Value')
plt.legend()
plt.grid(True)
plt.ylim(0, 1)  # Adjust based on your data range
plt.show()
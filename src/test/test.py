import time

import numpy as np
from PIL import Image

from src.image_processor import calculate_similarity
from src.params import crypto, image, params


def _encrypt(crypto=crypto, image=image, params=params):
    a_values = []  # Store 'a' values for each row's encryption

    encrypted_image = np.zeros_like(image.arr)  # Initialize encrypted image array
    for channel in range(image.arr.shape[-1]):
        for i, row in enumerate(image.arr[:, :, channel]):
            a, encrypted_row = crypto.encrypt(row)
            encrypted_image[i, :, channel] = encrypted_row
            a_values.append(a)
    image.save(params.encrypted_output_path, encrypted_image)
    return encrypted_image, a_values


def _decrypt(encrypted_image, a_values, crypto=crypto, image=image, params=params):
    decrypted_image = np.zeros_like(encrypted_image)  # Initialize decrypted image array
    for channel in range(encrypted_image.shape[-1]):
        a_index = channel * encrypted_image.shape[0]  # Calculate the starting index for 'a_values' for this channel
        for i, row in enumerate(encrypted_image[:, :, channel]):
            decrypted_row = crypto.decrypt(a_values[a_index + i], row)
            decrypted_image[i, :, channel] = decrypted_row

    # Resize the decrypted image back to the original size
    decrypted_image = np.array(Image.fromarray(decrypted_image)
                               .resize([image.new_width, image.new_height], Image.LANCZOS))
    image.save(params.decrypted_output_path, decrypted_image)
    return decrypted_image


def check_sspi(crypto=crypto, image=image, params=params):
    result = []
    encrypted_image, a_values = _encrypt(crypto=crypto, image=image, params=params)
    decrypted_image = _decrypt(encrypted_image, a_values, crypto=crypto, image=image, params=params)

    sspi = calculate_similarity(image.arr, encrypted_image)
    print(f'sspi between original image vs encrypted image: {sspi}')
    result.append(sspi)
    sspi = calculate_similarity(image.arr, decrypted_image)
    print(f'sspi between original image vs decrypted image: {sspi}')
    result.append(sspi)
    sspi = calculate_similarity(decrypted_image, encrypted_image)
    print(f'sspi between decrypted image vs encrypted image: {sspi}')
    result.append(sspi)
    return result


def check_timespend(crypto=crypto, image=image, params=params):
    start_enc_time = time.time()
    encrypted_image, a_values = _encrypt(crypto=crypto, image=image, params=params)
    end_enc_time = time.time()
    print(f"Encrypted: {end_enc_time - start_enc_time}")

    start_dec_time = time.time()
    decrypted_image = _decrypt(encrypted_image, a_values, crypto=crypto, image=image, params=params)
    end_dec_time = time.time()
    print(f"Decrypted: {end_dec_time - start_dec_time}")


check_sspi()

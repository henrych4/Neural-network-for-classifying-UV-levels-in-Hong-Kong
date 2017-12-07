#Reference: https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url

import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":
    file1_id = '1XNVMW2q6y-25B2iKFgUvlfIUcvJkK_uv'
    destination1 = 'data2_norm.npz'
    
    file2_id = '17V7wapW2MvEC799UTr-sK60fvdUhLYQH'
    destination2 = 'model.pkl'

    file3_id = '1Rz4p7XHep5G-t4kEVL8nc2V_G8kL8o-3'
    destination3 = 'v2_preprocessed.csv'

    download_file_from_google_drive(file1_id, destination1)
    download_file_from_google_drive(file2_id, destination2)
    download_file_from_google_drive(file3_id, destination3)



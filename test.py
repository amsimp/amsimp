# Import dependices.
import amsimp
import requests
import iris
from tqdm import tqdm

# Download zip file from Google Drive.
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id':id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id' : id, 'confirm':token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    total_size = 420300000
    t = tqdm(
        total=total_size, 
        unit='iB', 
        unit_scale=True,
        desc='Downloading models'
    )
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                t.update(len(chunk))

    t.close()

# Download file.
file_id = '1ta1pf0l1bbyG5aJkxVDJ5b0QqNHMZUv'
destination = 'example.nc'
download_file_from_google_drive(file_id, destination)

# Load file.
data = iris.load('example.nc')

# Define atmospheric state.
state = amsimp.Weather(historical_data=data)

# Generate forecast.
fct = state.generate_forecast()

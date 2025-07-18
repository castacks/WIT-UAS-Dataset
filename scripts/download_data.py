from minio import Minio

def download_from_airlab_server(client, bucket_name, source_name, target_name):
    """
    Downloads a file using Minio.

    Args:

    Returns:
    """
    print(f"Downloading {source_name} from {bucket_name}...")
    client.fget_object(bucket_name, source_name, target_name)
    print(f"Successfully downloaded {source_name} to {target_name}!")

    return True

file_1 = 'WIT-UAV-Dataset.zip' 
file_2 = 'WIT-UAV-Dataset_split.zip' 
bucket_name = "wit-uas-dataset"
endpoint_url = "airlab-share-02.andrew.cmu.edu:9000"

client = Minio(endpoint_url,
                access_key="y5zkhCOAl5d0OMxwtx4H",
                secret_key="W5qsLEyLYNofJhupZ0S4ofarEIAHDm9wkwV5UHFs",
                secure=True 
                )
res = download_from_airlab_server(client,
                                bucket_name,
                                source_name=file_1, # file on the server
                                target_name=file_1)
res = download_from_airlab_server(client,
                                bucket_name,
                                source_name=file_2, # file on the server
                                target_name=file_2)
print("Done.")
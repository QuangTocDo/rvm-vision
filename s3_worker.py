import glob
import boto3
from uuid import uuid4 as get_mac
import zlib
import shutil
import time
import os
import re
from datetime import datetime, timedelta
from threading import Thread

import cv2

S3_BUCKET_NAME = 'rvm-storage'
S3_ENDPOINT = "https://s3.altacloud.biz:443"
AWS_PROFILE = 'rvm-profile'
S3_access_key_id = 'rvm-publish'
S3_secret_access_key = "&Dfhcj$vRVbaUU2c"
CACHES = []
DATA_FILES_LOCATION = "./temp/"
config = boto3.session.Config(connect_timeout=5, signature_version='s3v4' , retries={'max_attempts': 5})

# rvm-publish/&Dfhcj$vRVbaUU2c

s3 = boto3.client('s3',
                  endpoint_url=S3_ENDPOINT,
                  config=config,
                  aws_access_key_id=S3_access_key_id,
                  aws_secret_access_key=S3_secret_access_key,
                  aws_session_token=None
                  )

try:
    result = s3.get_bucket_acl(Bucket=S3_BUCKET_NAME)
except Exception as ex:
    try:
        s3.create_bucket(Bucket=S3_BUCKET_NAME)
    except Exception as ex2:
        print(ex2)

client = boto3.resource("s3",
                        endpoint_url=S3_ENDPOINT,
                        config=config,
                        aws_access_key_id=S3_access_key_id,
                        aws_secret_access_key=S3_secret_access_key,
                        aws_session_token=None)


def s3_dirs(path: str, bucket: str = None):
    try:
        objects = client.Bucket(
            bucket if bucket is not None else S3_BUCKET_NAME).objects.all()
        all = []
        for obj in objects.filter(Prefix=path):
            object_name = obj.key
            all.append(object_name)
        return all
    except Exception as ex:
        return []


def get_all():
    try:
        all = []
        resp = s3.list_objects(Bucket=S3_BUCKET_NAME, Prefix="", Delimiter='/')
        for obj in resp['CommonPrefixes']:
            all.append(str(obj['Prefix']).rstrip("/"))
        return all
    except Exception as ex:
        return []


def create_presigned_url(bucket_key, expiration=3600, Bucket=None):
    return s3.generate_presigned_url('get_object',
                                     Params={'Bucket': Bucket if Bucket is not None else S3_BUCKET_NAME,
                                             'Key': bucket_key},
                                     ExpiresIn=expiration)


def random():
    __mac = str(
        hex(zlib.crc32(bytes(str(get_mac()), 'ascii'))).replace("0x", "r-"))
    return __mac


def unix():
    ignores = get_all()
    mac = random()
    while mac in ignores:
        mac = random()
    try:
        s3.put_object(Bucket=S3_BUCKET_NAME, Key=str(mac) + "/")
    except Exception as ex:
        print(ex)
    return mac


def move_s3_object(src, dest) -> None:
    old_bucket = src['Bucket'] if 'Bucket' in src else S3_BUCKET_NAME
    old_key = src['Key']
    client.Bucket(dest['Bucket'] if 'Bucket' in dest else S3_BUCKET_NAME).copy(
        {'Bucket': old_bucket, 'Key': old_key}, dest['Key'] if 'Key' in dest else old_key)
    s3.delete_object(Bucket=old_bucket, Key=old_key)


def delete_object(Key: str, bucket: str = None):
    s3.delete_object(
        Bucket=bucket if bucket is not None else S3_BUCKET_NAME, Key=Key)


def getDirs(root_dir, regex=".*\.npz"):
    # files = glob.glob(regex,root_dir=root_dir,recursive=True)
    files = os.listdir(root_dir)
    if files == None or len(files) == 0:
        return []
    file_list = []
    for file in files:
        if re.match(regex, regex):
            file_list.append(file)
    return file_list


def upload_file(path: str, key: str = None, bucket: str = None):
    s3.upload_file(path, bucket if bucket is not None else S3_BUCKET_NAME,
                   key if key is not None else path)


def sync_dir(path, S3_FOLDER_NAME):
    files = getDirs(path)
    if files == None or len(files) == 0:
        print("CLEAN", S3_FOLDER_NAME)
        shutil.rmtree(path, True)
        return
    print("sync_dir", len(files), path)
    try:
        for file in files:
            full_path = path + "/" + file
            s3_file = f"{S3_FOLDER_NAME}/{file}"
            s3.upload_file(full_path, S3_BUCKET_NAME, s3_file)
            # upload_file(full_path, s3_file)
            # socketio.sleep(0.05)
            os.unlink(full_path)
            # time.sleep(0.01)
            # socketio.sleep(0.05)
            print("sync_dir", full_path)
    except Exception as ex:
        print("SC Exception", ex)
        pass


def dhash_path(path, hashSize=8):
    img = cv2.imread(path)
    return dhash(img, hashSize), img


def dhash(image, hashSize=8):
    # resize the input image, adding a single column (width) so we
    # can compute the horizontal gradient
    resized = cv2.resize(image, (hashSize + 1, hashSize))
    # compute the (relative) horizontal gradient between adjacent
    # column pixels
    diff = resized[:, 1:] > resized[:, :-1]
    # convert the difference image to a hash
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


def sync(S3_FOLDER_NAME, date):
    root_dirs = getDirs(DATA_FILES_LOCATION, ".*")
    print("SYNC BEGIN", S3_FOLDER_NAME, date, root_dirs)
    for d in root_dirs:
        if d <= date:
            th = Thread(target=sync_dir, args=(
                f"{DATA_FILES_LOCATION}{d}", f"{S3_FOLDER_NAME}/{d}"))
            th.daemon = True
            th.start()
            # socketio.sleep(0.05)
    print("SYNC END", S3_FOLDER_NAME)


if __name__ == "__main__":
    print(s3_dirs("r-e34c74b8/2023-07-28"))

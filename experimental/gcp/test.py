from google.cloud import storage

client = storage.Client()

bucket = client.get_bucket('first-ml-project-222122-mlengine')
# for blob in bucket.list_blobs(delimiter="/"):
# for blob in bucket.list_blobs(prefix='sample-data/train/real', delimiter='/'):
	# print(blob.name)

iterator = bucket.list_blobs(prefix='sample-data/train/real/', delimiter='/')
for page in iterator.pages:
    for prefix in page.prefixes:
    	print("PREFIX: ", prefix)
    	iter2 = bucket.list_blobs(prefix=prefix)
    	for file in iter2:
    		print(file.name)
    	print()

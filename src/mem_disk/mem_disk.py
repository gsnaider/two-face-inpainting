import fs
from fs.zipfs import ZipFS
from fs_gcsfs import GCSFS

# Already fixed:
# data/train/real
# data/train/masked
# data/train/reference
# data/validation

with fs.open_fs('mem://') as mem_fs:
  # with fs.open_fs('gs://two-face-inpainting-mlengine/sample-data?strict=False') as gcsfs:
  with GCSFS(bucket_name="two-face-inpainting-mlengine", root_path='data/validation') as gcsfs:
    # with gcsfs.open('sample.zip', 'rb') as zip_file:
    #   with ZipFS(zip_file) as zip_fs:
    #     fs.copy.copy_dir(zip_fs, '.', mem_fs, '.')
    gcsfs.fix_storage()
    # gcsfs.tree()
  # mem_fs.tree()
  # walker = Walker()
  # for path in walker.files(mem_fs):
  #   print(path)
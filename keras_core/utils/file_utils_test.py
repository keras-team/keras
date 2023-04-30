import os
import tarfile
import urllib
import zipfile

from keras_core.testing import test_case
from keras_core.utils import file_utils


class TestGetFile(test_case.TestCase):
    def test_get_file_and_validate_it(self):
        """Tests get_file from a url, plus extraction and validation."""
        dest_dir = self.get_temp_dir()
        orig_dir = self.get_temp_dir()

        text_file_path = os.path.join(orig_dir, "test.txt")
        zip_file_path = os.path.join(orig_dir, "test.zip")
        tar_file_path = os.path.join(orig_dir, "test.tar.gz")

        with open(text_file_path, "w") as text_file:
            text_file.write("Float like a butterfly, sting like a bee.")

        with tarfile.open(tar_file_path, "w:gz") as tar_file:
            tar_file.add(text_file_path)

        with zipfile.ZipFile(zip_file_path, "w") as zip_file:
            zip_file.write(text_file_path)

        origin = urllib.parse.urljoin(
            "file://",
            urllib.request.pathname2url(os.path.abspath(tar_file_path)),
        )

        path = file_utils.get_file(
            "test.txt", origin, untar=True, cache_subdir=dest_dir
        )
        filepath = path + ".tar.gz"
        hashval_sha256 = file_utils.hash_file(filepath)
        hashval_md5 = file_utils.hash_file(filepath, algorithm="md5")
        path = file_utils.get_file(
            "test.txt",
            origin,
            md5_hash=hashval_md5,
            untar=True,
            cache_subdir=dest_dir,
        )
        path = file_utils.get_file(
            filepath,
            origin,
            file_hash=hashval_sha256,
            extract=True,
            cache_subdir=dest_dir,
        )
        self.assertTrue(os.path.exists(filepath))
        self.assertTrue(file_utils.validate_file(filepath, hashval_sha256))
        self.assertTrue(file_utils.validate_file(filepath, hashval_md5))
        os.remove(filepath)

        origin = urllib.parse.urljoin(
            "file://",
            urllib.request.pathname2url(os.path.abspath(zip_file_path)),
        )

        hashval_sha256 = file_utils.hash_file(zip_file_path)
        hashval_md5 = file_utils.hash_file(zip_file_path, algorithm="md5")
        path = file_utils.get_file(
            "test",
            origin,
            md5_hash=hashval_md5,
            extract=True,
            cache_subdir=dest_dir,
        )
        path = file_utils.get_file(
            "test",
            origin,
            file_hash=hashval_sha256,
            extract=True,
            cache_subdir=dest_dir,
        )
        self.assertTrue(os.path.exists(path))
        self.assertTrue(file_utils.validate_file(path, hashval_sha256))
        self.assertTrue(file_utils.validate_file(path, hashval_md5))
        os.remove(path)

        for file_path, extract in [
            (text_file_path, False),
            (tar_file_path, True),
            (zip_file_path, True),
        ]:
            origin = urllib.parse.urljoin(
                "file://",
                urllib.request.pathname2url(os.path.abspath(file_path)),
            )
            hashval_sha256 = file_utils.hash_file(file_path)
            path = file_utils.get_file(
                origin=origin,
                file_hash=hashval_sha256,
                extract=extract,
                cache_subdir=dest_dir,
            )
            self.assertTrue(os.path.exists(path))
            self.assertTrue(file_utils.validate_file(path, hashval_sha256))
            os.remove(path)

        with self.assertRaisesRegexp(
            ValueError, 'Please specify the "origin".*'
        ):
            _ = file_utils.get_file()

    def test_get_file_with_tgz_extension(self):
        """Tests get_file from a url, plus extraction and validation."""
        dest_dir = self.get_temp_dir()
        orig_dir = dest_dir

        text_file_path = os.path.join(orig_dir, "test.txt")
        tar_file_path = os.path.join(orig_dir, "test.tar.gz")

        with open(text_file_path, "w") as text_file:
            text_file.write("Float like a butterfly, sting like a bee.")

        with tarfile.open(tar_file_path, "w:gz") as tar_file:
            tar_file.add(text_file_path)

        origin = urllib.parse.urljoin(
            "file://",
            urllib.request.pathname2url(os.path.abspath(tar_file_path)),
        )

        path = file_utils.get_file(
            "test.txt.tar.gz", origin, untar=True, cache_subdir=dest_dir
        )
        self.assertTrue(path.endswith(".txt"))
        self.assertTrue(os.path.exists(path))

    def test_get_file_with_integrity_check(self):
        """Tests get_file with validation before download."""
        orig_dir = self.get_temp_dir()
        file_path = os.path.join(orig_dir, "test.txt")

        with open(file_path, "w") as text_file:
            text_file.write("Float like a butterfly, sting like a bee.")

        hashval = file_utils.hash_file(file_path)

        origin = urllib.parse.urljoin(
            "file://", urllib.request.pathname2url(os.path.abspath(file_path))
        )

        path = file_utils.get_file("test.txt", origin, file_hash=hashval)
        self.assertTrue(os.path.exists(path))

    def test_get_file_with_failed_integrity_check(self):
        """Tests get_file with validation before download."""
        orig_dir = self.get_temp_dir()
        file_path = os.path.join(orig_dir, "test.txt")

        with open(file_path, "w") as text_file:
            text_file.write("Float like a butterfly, sting like a bee.")

        hashval = "0" * 64

        origin = urllib.parse.urljoin(
            "file://", urllib.request.pathname2url(os.path.abspath(file_path))
        )

        with self.assertRaisesRegex(
            ValueError, "Incomplete or corrupted file.*"
        ):
            _ = file_utils.get_file("test.txt", origin, file_hash=hashval)

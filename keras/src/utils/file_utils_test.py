import hashlib
import os
import shutil
import tarfile
import tempfile
import urllib
import urllib.parse
import urllib.request
import zipfile
from unittest.mock import patch

from keras.src.testing import test_case
from keras.src.utils import file_utils


class PathToStringTest(test_case.TestCase):
    def test_path_to_string_with_string_path(self):
        path = os.path.join(os.path.sep, "path", "to", "file.txt")
        string_path = file_utils.path_to_string(path)
        self.assertEqual(string_path, path)

    def test_path_to_string_with_PathLike_object(self):
        path = os.path.join(os.path.sep, "path", "to", "file.txt")
        string_path = file_utils.path_to_string(path)
        self.assertEqual(string_path, str(path))

    def test_path_to_string_with_non_string_typed_path_object(self):
        class NonStringTypedPathObject:
            def __fspath__(self):
                return os.path.join(os.path.sep, "path", "to", "file.txt")

        path = NonStringTypedPathObject()
        string_path = file_utils.path_to_string(path)
        self.assertEqual(
            string_path, os.path.join(os.path.sep, "path", "to", "file.txt")
        )

    def test_path_to_string_with_none_path(self):
        string_path = file_utils.path_to_string(None)
        self.assertEqual(string_path, None)


class ResolvePathTest(test_case.TestCase):
    def test_resolve_path_with_absolute_path(self):
        path = os.path.join(os.path.sep, "path", "to", "file.txt")
        resolved_path = file_utils.resolve_path(path)
        self.assertEqual(resolved_path, os.path.realpath(os.path.abspath(path)))

    def test_resolve_path_with_relative_path(self):
        path = os.path.join(".", "file.txt")
        resolved_path = file_utils.resolve_path(path)
        self.assertEqual(resolved_path, os.path.realpath(os.path.abspath(path)))


class IsPathInDirTest(test_case.TestCase):
    def test_is_path_in_dir_with_absolute_paths(self):
        base_dir = os.path.join(os.path.sep, "path", "to", "base_dir")
        path = os.path.join(base_dir, "file.txt")
        self.assertTrue(file_utils.is_path_in_dir(path, base_dir))


class IsLinkInDirTest(test_case.TestCase):
    def setUp(self):
        self._cleanup(os.path.join("test_path", "to", "base_dir"))
        self._cleanup(os.path.join(".", "base_dir"))

    def _cleanup(self, base_dir):
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)

    def test_is_link_in_dir_with_absolute_paths(self):
        base_dir = os.path.join("test_path", "to", "base_dir")
        link_path = os.path.join(base_dir, "symlink")
        target_path = os.path.join(base_dir, "file.txt")

        # Create the base_dir directory if it does not exist.
        os.makedirs(base_dir, exist_ok=True)

        # Create the file.txt file.
        with open(target_path, "w") as f:
            f.write("Hello, world!")

        os.symlink(target_path, link_path)

        # Creating a stat_result-like object with a name attribute
        info = os.lstat(link_path)
        info = type(
            "stat_with_name",
            (object,),
            {
                "name": os.path.basename(link_path),
                "linkname": os.readlink(link_path),
            },
        )

        self.assertTrue(file_utils.is_link_in_dir(info, base_dir))

    def test_is_link_in_dir_with_relative_paths(self):
        base_dir = os.path.join(".", "base_dir")
        link_path = os.path.join(base_dir, "symlink")
        target_path = os.path.join(base_dir, "file.txt")

        # Create the base_dir directory if it does not exist.
        os.makedirs(base_dir, exist_ok=True)

        # Create the file.txt file.
        with open(target_path, "w") as f:
            f.write("Hello, world!")

        os.symlink(target_path, link_path)

        # Creating a stat_result-like object with a name attribute
        info = os.lstat(link_path)
        info = type(
            "stat_with_name",
            (object,),
            {
                "name": os.path.basename(link_path),
                "linkname": os.readlink(link_path),
            },
        )

        self.assertTrue(file_utils.is_link_in_dir(info, base_dir))

    def tearDown(self):
        self._cleanup(os.path.join("test_path", "to", "base_dir"))
        self._cleanup(os.path.join(".", "base_dir"))


class FilterSafePathsTest(test_case.TestCase):
    def setUp(self):
        self.base_dir = os.path.join(os.getcwd(), "temp_dir")
        os.makedirs(self.base_dir, exist_ok=True)
        self.tar_path = os.path.join(self.base_dir, "test.tar")

    def tearDown(self):
        os.remove(self.tar_path)
        shutil.rmtree(self.base_dir)

    def test_member_within_base_dir(self):
        """Test a member within the base directory."""
        with tarfile.open(self.tar_path, "w") as tar:
            tar.add(__file__, arcname="safe_path.txt")
        with tarfile.open(self.tar_path, "r") as tar:
            members = list(file_utils.filter_safe_tarinfos(tar.getmembers()))
            self.assertEqual(len(members), 1)
            self.assertEqual(members[0].name, "safe_path.txt")

    def test_symlink_within_base_dir(self):
        """Test a symlink pointing within the base directory."""
        symlink_path = os.path.join(self.base_dir, "symlink.txt")
        target_path = os.path.join(self.base_dir, "target.txt")
        with open(target_path, "w") as f:
            f.write("target")
        os.symlink(target_path, symlink_path)
        with tarfile.open(self.tar_path, "w") as tar:
            tar.add(symlink_path, arcname="symlink.txt")
        with tarfile.open(self.tar_path, "r") as tar:
            members = list(file_utils.filter_safe_tarinfos(tar.getmembers()))
            self.assertEqual(len(members), 1)
            self.assertEqual(members[0].name, "symlink.txt")
        os.remove(symlink_path)
        os.remove(target_path)

    def test_invalid_path_warning(self):
        """Test warning for an invalid path during archive extraction."""
        invalid_path = os.path.join(os.getcwd(), "invalid.txt")
        with open(invalid_path, "w") as f:
            f.write("invalid")
        with tarfile.open(self.tar_path, "w") as tar:
            tar.add(
                invalid_path, arcname="../../invalid.txt"
            )  # Path intended to be outside of base dir
        with tarfile.open(self.tar_path, "r") as tar:
            with patch("warnings.warn") as mock_warn:
                _ = list(file_utils.filter_safe_tarinfos(tar.getmembers()))
                warning_msg = (
                    "Skipping invalid path during archive extraction: "
                    "'../../invalid.txt'."
                )
                mock_warn.assert_called_with(warning_msg, stacklevel=2)
        os.remove(invalid_path)

    def test_symbolic_link_in_base_dir(self):
        """symbolic link within the base directory is correctly processed."""
        symlink_path = os.path.join(self.base_dir, "symlink.txt")
        target_path = os.path.join(self.base_dir, "target.txt")

        # Create a target file and then a symbolic link pointing to it.
        with open(target_path, "w") as f:
            f.write("target")
        os.symlink(target_path, symlink_path)

        # Add the symbolic link to the tar archive.
        with tarfile.open(self.tar_path, "w") as tar:
            tar.add(symlink_path, arcname="symlink.txt")

        with tarfile.open(self.tar_path, "r") as tar:
            members = list(file_utils.filter_safe_tarinfos(tar.getmembers()))
            self.assertEqual(len(members), 1)
            self.assertEqual(members[0].name, "symlink.txt")
            self.assertTrue(
                members[0].issym()
            )  # Explicitly assert it's a symbolic link.

        os.remove(symlink_path)
        os.remove(target_path)


class ExtractArchiveTest(test_case.TestCase):
    def setUp(self):
        """Create temporary directories and files for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.file_content = "Hello, world!"

        # Create sample files to be archived
        with open(os.path.join(self.temp_dir, "sample.txt"), "w") as f:
            f.write(self.file_content)

    def tearDown(self):
        """Clean up temporary directories."""
        shutil.rmtree(self.temp_dir)

    def create_tar(self):
        archive_path = os.path.join(self.temp_dir, "sample.tar")
        with tarfile.open(archive_path, "w") as archive:
            archive.add(
                os.path.join(self.temp_dir, "sample.txt"), arcname="sample.txt"
            )
        return archive_path

    def create_zip(self):
        archive_path = os.path.join(self.temp_dir, "sample.zip")
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.write(
                os.path.join(self.temp_dir, "sample.txt"), arcname="sample.txt"
            )
        return archive_path

    def test_extract_tar(self):
        archive_path = self.create_tar()
        extract_path = os.path.join(self.temp_dir, "extract_tar")
        result = file_utils.extract_archive(archive_path, extract_path, "tar")
        self.assertTrue(result)
        with open(os.path.join(extract_path, "sample.txt"), "r") as f:
            self.assertEqual(f.read(), self.file_content)

    def test_extract_zip(self):
        archive_path = self.create_zip()
        extract_path = os.path.join(self.temp_dir, "extract_zip")
        result = file_utils.extract_archive(archive_path, extract_path, "zip")
        self.assertTrue(result)
        with open(os.path.join(extract_path, "sample.txt"), "r") as f:
            self.assertEqual(f.read(), self.file_content)

    def test_extract_auto(self):
        # This will test the 'auto' functionality
        tar_archive_path = self.create_tar()
        zip_archive_path = self.create_zip()

        extract_tar_path = os.path.join(self.temp_dir, "extract_auto_tar")
        extract_zip_path = os.path.join(self.temp_dir, "extract_auto_zip")

        self.assertTrue(
            file_utils.extract_archive(tar_archive_path, extract_tar_path)
        )
        self.assertTrue(
            file_utils.extract_archive(zip_archive_path, extract_zip_path)
        )

        with open(os.path.join(extract_tar_path, "sample.txt"), "r") as f:
            self.assertEqual(f.read(), self.file_content)

        with open(os.path.join(extract_zip_path, "sample.txt"), "r") as f:
            self.assertEqual(f.read(), self.file_content)

    def test_non_existent_file(self):
        extract_path = os.path.join(self.temp_dir, "non_existent")
        with self.assertRaises(FileNotFoundError):
            file_utils.extract_archive("non_existent.tar", extract_path)

    def test_archive_format_none(self):
        archive_path = self.create_tar()
        extract_path = os.path.join(self.temp_dir, "none_format")
        result = file_utils.extract_archive(archive_path, extract_path, None)
        self.assertFalse(result)

    def test_runtime_error_during_extraction(self):
        tar_path = self.create_tar()
        extract_path = os.path.join(self.temp_dir, "runtime_error_extraction")

        with patch.object(
            tarfile.TarFile, "extractall", side_effect=RuntimeError
        ):
            with self.assertRaises(RuntimeError):
                file_utils.extract_archive(tar_path, extract_path, "tar")
        self.assertFalse(os.path.exists(extract_path))

    def test_keyboard_interrupt_during_extraction(self):
        tar_path = self.create_tar()
        extract_path = os.path.join(
            self.temp_dir, "keyboard_interrupt_extraction"
        )

        with patch.object(
            tarfile.TarFile, "extractall", side_effect=KeyboardInterrupt
        ):
            with self.assertRaises(KeyboardInterrupt):
                file_utils.extract_archive(tar_path, extract_path, "tar")
        self.assertFalse(os.path.exists(extract_path))


class GetFileTest(test_case.TestCase):
    def setUp(self):
        """Set up temporary directories and sample files."""
        self.temp_dir = self.get_temp_dir()
        self.file_path = os.path.join(self.temp_dir, "sample_file.txt")
        with open(self.file_path, "w") as f:
            f.write("Sample content")

    def test_valid_tar_extraction(self):
        """Test valid tar.gz extraction and hash validation."""
        dest_dir = self.get_temp_dir()
        orig_dir = self.get_temp_dir()
        _, tar_file_path = self._create_tar_file(orig_dir)
        self._test_file_extraction_and_validation(
            dest_dir, tar_file_path, "tar.gz"
        )

    def test_valid_zip_extraction(self):
        """Test valid zip extraction and hash validation."""
        dest_dir = self.get_temp_dir()
        orig_dir = self.get_temp_dir()
        _, zip_file_path = self._create_zip_file(orig_dir)
        self._test_file_extraction_and_validation(
            dest_dir, zip_file_path, "zip"
        )

    def test_valid_text_file_download(self):
        """Test valid text file download and hash validation."""
        dest_dir = self.get_temp_dir()
        orig_dir = self.get_temp_dir()
        text_file_path = os.path.join(orig_dir, "test.txt")
        with open(text_file_path, "w") as text_file:
            text_file.write("Float like a butterfly, sting like a bee.")
        self._test_file_extraction_and_validation(
            dest_dir, text_file_path, None
        )

    def test_get_file_with_tgz_extension(self):
        """Test extraction of file with .tar.gz extension."""
        dest_dir = self.get_temp_dir()
        orig_dir = dest_dir
        _, tar_file_path = self._create_tar_file(orig_dir)

        origin = urllib.parse.urljoin(
            "file://",
            urllib.request.pathname2url(os.path.abspath(tar_file_path)),
        )

        path = file_utils.get_file(
            "test.txt.tar.gz", origin, untar=True, cache_subdir=dest_dir
        )
        self.assertTrue(os.path.exists(path))
        self.assertTrue(os.path.exists(os.path.join(path, "test.txt")))

    def test_get_file_with_integrity_check(self):
        """Test file download with integrity check."""
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

    def test_cache_invalidation(self):
        """Test using a hash to force cache invalidation."""
        cache_dir = self.get_temp_dir()
        src_path = os.path.join(self.get_temp_dir(), "test.txt")
        with open(src_path, "w") as text_file:
            text_file.write("Float like a butterfly, sting like a bee.")
        orig_hash = file_utils.hash_file(src_path)
        origin = urllib.parse.urljoin(
            "file://", urllib.request.pathname2url(os.path.abspath(src_path))
        )
        # Download into the cache.
        dest_path = file_utils.get_file(
            "test.txt", origin, file_hash=orig_hash, cache_dir=cache_dir
        )
        self.assertEqual(orig_hash, file_utils.hash_file(dest_path))

        with open(src_path, "w") as text_file:
            text_file.write("Float like a zeppelin, sting like a jellyfish.")
        new_hash = file_utils.hash_file(src_path)
        # Without a hash, we should get the cached version.
        dest_path = file_utils.get_file("test.txt", origin, cache_dir=cache_dir)
        self.assertEqual(orig_hash, file_utils.hash_file(dest_path))
        # Without the new hash, we should re-download.
        dest_path = file_utils.get_file(
            "test.txt", origin, file_hash=new_hash, cache_dir=cache_dir
        )
        self.assertEqual(new_hash, file_utils.hash_file(dest_path))

    def test_force_download(self):
        """Test using a hash to force cache invalidation."""
        cache_dir = self.get_temp_dir()
        src_path = os.path.join(self.get_temp_dir(), "test.txt")
        with open(src_path, "w") as text_file:
            text_file.write("Float like a butterfly, sting like a bee.")
        orig_hash = file_utils.hash_file(src_path)
        origin = urllib.parse.urljoin(
            "file://", urllib.request.pathname2url(os.path.abspath(src_path))
        )
        # Download into the cache.
        dest_path = file_utils.get_file("test.txt", origin, cache_dir=cache_dir)
        self.assertEqual(orig_hash, file_utils.hash_file(dest_path))

        with open(src_path, "w") as text_file:
            text_file.write("Float like a zeppelin, sting like a jellyfish.")
        new_hash = file_utils.hash_file(src_path)
        # Get cached version.
        dest_path = file_utils.get_file("test.txt", origin, cache_dir=cache_dir)
        self.assertEqual(orig_hash, file_utils.hash_file(dest_path))
        # Force download.
        dest_path = file_utils.get_file(
            "test.txt", origin, force_download=True, cache_dir=cache_dir
        )
        self.assertEqual(new_hash, file_utils.hash_file(dest_path))

    def test_get_file_with_failed_integrity_check(self):
        """Test file download with failed integrity check."""
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

    def _create_tar_file(self, directory):
        """Helper function to create a tar file."""
        text_file_path = os.path.join(directory, "test.txt")
        tar_file_path = os.path.join(directory, "test.tar.gz")
        with open(text_file_path, "w") as text_file:
            text_file.write("Float like a butterfly, sting like a bee.")

        with tarfile.open(tar_file_path, "w:gz") as tar_file:
            tar_file.add(text_file_path, arcname="test.txt")

        return text_file_path, tar_file_path

    def _create_zip_file(self, directory):
        """Helper function to create a zip file."""
        text_file_path = os.path.join(directory, "test.txt")
        zip_file_path = os.path.join(directory, "test.zip")
        with open(text_file_path, "w") as text_file:
            text_file.write("Float like a butterfly, sting like a bee.")

        with zipfile.ZipFile(zip_file_path, "w") as zip_file:
            zip_file.write(text_file_path, arcname="test.txt")

        return text_file_path, zip_file_path

    def _test_file_extraction_and_validation(
        self, dest_dir, file_path, archive_type
    ):
        """Helper function for file extraction and validation."""
        origin = urllib.parse.urljoin(
            "file://",
            urllib.request.pathname2url(os.path.abspath(file_path)),
        )

        hashval_md5 = file_utils.hash_file(file_path, algorithm="md5")

        extract = bool(archive_type)

        path = file_utils.get_file(
            "test",
            origin,
            md5_hash=hashval_md5,
            extract=extract,
            cache_subdir=dest_dir,
        )
        if extract:
            fpath = f"{path}_archive"
        else:
            fpath = path

        self.assertTrue(os.path.exists(path))
        self.assertTrue(file_utils.validate_file(fpath, hashval_md5))
        if extract:
            self.assertTrue(os.path.exists(os.path.join(path, "test.txt")))

    def test_exists(self):
        temp_dir = self.get_temp_dir()
        file_path = os.path.join(temp_dir, "test_exists.txt")

        with open(file_path, "w") as f:
            f.write("test")

        self.assertTrue(file_utils.exists(file_path))
        self.assertFalse(
            file_utils.exists(os.path.join(temp_dir, "non_existent.txt"))
        )

    def test_file_open_read(self):
        temp_dir = self.get_temp_dir()
        file_path = os.path.join(temp_dir, "test_file.txt")
        content = "test content"

        with open(file_path, "w") as f:
            f.write(content)

        with file_utils.File(file_path, "r") as f:
            self.assertEqual(f.read(), content)

    def test_file_open_write(self):
        temp_dir = self.get_temp_dir()
        file_path = os.path.join(temp_dir, "test_file_write.txt")
        content = "test write content"

        with file_utils.File(file_path, "w") as f:
            f.write(content)

        with open(file_path, "r") as f:
            self.assertEqual(f.read(), content)

    def test_isdir(self):
        temp_dir = self.get_temp_dir()
        self.assertTrue(file_utils.isdir(temp_dir))

        file_path = os.path.join(temp_dir, "test_isdir.txt")
        with open(file_path, "w") as f:
            f.write("test")
        self.assertFalse(file_utils.isdir(file_path))

    def test_join_simple(self):
        self.assertEqual(file_utils.join("/path", "to", "dir"), "/path/to/dir")

    def test_join_single_directory(self):
        self.assertEqual(file_utils.join("/path"), "/path")

    def test_listdir(self):
        content = file_utils.listdir(self.temp_dir)
        self.assertIn("sample_file.txt", content)

    def test_makedirs_and_rmtree(self):
        new_dir = os.path.join(self.temp_dir, "new_directory")
        file_utils.makedirs(new_dir)
        self.assertTrue(os.path.isdir(new_dir))
        file_utils.rmtree(new_dir)
        self.assertFalse(os.path.exists(new_dir))

    def test_copy(self):
        dest_path = os.path.join(self.temp_dir, "copy_sample_file.txt")
        file_utils.copy(self.file_path, dest_path)
        self.assertTrue(os.path.exists(dest_path))
        with open(dest_path, "r") as f:
            content = f.read()
        self.assertEqual(content, "Sample content")

    def test_remove_sub_directory(self):
        parent_dir = os.path.join(self.get_temp_dir(), "parent_directory")
        child_dir = os.path.join(parent_dir, "child_directory")
        file_utils.makedirs(child_dir)
        file_utils.rmtree(parent_dir)
        self.assertFalse(os.path.exists(parent_dir))
        self.assertFalse(os.path.exists(child_dir))

    def test_remove_files_inside_directory(self):
        dir_path = os.path.join(self.get_temp_dir(), "test_directory")
        file_path = os.path.join(dir_path, "test.txt")
        file_utils.makedirs(dir_path)
        with open(file_path, "w") as f:
            f.write("Test content")
        file_utils.rmtree(dir_path)
        self.assertFalse(os.path.exists(dir_path))
        self.assertFalse(os.path.exists(file_path))

    def test_handle_complex_paths(self):
        complex_dir = os.path.join(self.get_temp_dir(), "complex dir@#%&!")
        file_utils.makedirs(complex_dir)
        file_utils.rmtree(complex_dir)
        self.assertFalse(os.path.exists(complex_dir))


class HashFileTest(test_case.TestCase):
    def setUp(self):
        self.test_content = b"Hello, World!"
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file.write(self.test_content)
        self.temp_file.close()

    def tearDown(self):
        os.remove(self.temp_file.name)

    def test_hash_file_sha256(self):
        """Test SHA256 hashing of a file."""
        expected_sha256 = (
            "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"
        )
        calculated_sha256 = file_utils.hash_file(
            self.temp_file.name, algorithm="sha256"
        )
        self.assertEqual(expected_sha256, calculated_sha256)

    def test_hash_file_md5(self):
        """Test MD5 hashing of a file."""
        expected_md5 = "65a8e27d8879283831b664bd8b7f0ad4"
        calculated_md5 = file_utils.hash_file(
            self.temp_file.name, algorithm="md5"
        )
        self.assertEqual(expected_md5, calculated_md5)


class TestValidateFile(test_case.TestCase):
    def setUp(self):
        self.tmp_file = tempfile.NamedTemporaryFile(delete=False)
        self.tmp_file.write(b"Hello, World!")
        self.tmp_file.close()

        self.sha256_hash = (
            "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"
        )
        self.md5_hash = "65a8e27d8879283831b664bd8b7f0ad4"

    def test_validate_file_sha256(self):
        """Validate SHA256 hash of a file."""
        self.assertTrue(
            file_utils.validate_file(
                self.tmp_file.name, self.sha256_hash, "sha256"
            )
        )

    def test_validate_file_md5(self):
        """Validate MD5 hash of a file."""
        self.assertTrue(
            file_utils.validate_file(self.tmp_file.name, self.md5_hash, "md5")
        )

    def test_validate_file_auto_sha256(self):
        """Auto-detect and validate SHA256 hash."""
        self.assertTrue(
            file_utils.validate_file(
                self.tmp_file.name, self.sha256_hash, "auto"
            )
        )

    def test_validate_file_auto_md5(self):
        """Auto-detect and validate MD5 hash."""
        self.assertTrue(
            file_utils.validate_file(self.tmp_file.name, self.md5_hash, "auto")
        )

    def test_validate_file_wrong_hash(self):
        """Test validation with incorrect hash."""
        wrong_hash = "deadbeef" * 8
        self.assertFalse(
            file_utils.validate_file(self.tmp_file.name, wrong_hash, "sha256")
        )

    def tearDown(self):
        os.remove(self.tmp_file.name)


class ResolveHasherTest(test_case.TestCase):
    def test_resolve_hasher_sha256(self):
        """Test resolving hasher for sha256 algorithm."""
        hasher = file_utils.resolve_hasher("sha256")
        self.assertIsInstance(hasher, type(hashlib.sha256()))

    def test_resolve_hasher_auto_sha256(self):
        """Auto-detect and resolve hasher for sha256."""
        hasher = file_utils.resolve_hasher("auto", file_hash="a" * 64)
        self.assertIsInstance(hasher, type(hashlib.sha256()))

    def test_resolve_hasher_auto_md5(self):
        """Auto-detect and resolve hasher for md5."""
        hasher = file_utils.resolve_hasher("auto", file_hash="a" * 32)
        self.assertIsInstance(hasher, type(hashlib.md5()))

    def test_resolve_hasher_default(self):
        """Resolve hasher with a random algorithm value."""
        hasher = file_utils.resolve_hasher("random_value")
        self.assertIsInstance(hasher, type(hashlib.md5()))


class IsRemotePathTest(test_case.TestCase):
    def test_gcs_remote_path(self):
        self.assertTrue(file_utils.is_remote_path("/gcs/some/path/to/file.txt"))
        self.assertTrue(file_utils.is_remote_path("/gcs/another/directory/"))
        self.assertTrue(file_utils.is_remote_path("gcs://bucket/some/file.txt"))

    def test_hdfs_remote_path(self):
        self.assertTrue(file_utils.is_remote_path("hdfs://some/path/on/hdfs"))
        self.assertTrue(file_utils.is_remote_path("/hdfs/some/local/path"))

    def test_cns_remote_path(self):
        self.assertTrue(file_utils.is_remote_path("/cns/some/path"))

    def test_placer_remote_path(self):
        self.assertTrue(
            file_utils.is_remote_path("/placer/prod/home/some/path")
        )
        self.assertTrue(
            file_utils.is_remote_path("/placer/test/home/some/path")
        )
        self.assertTrue(
            file_utils.is_remote_path("/placer/prod/scratch/home/some/path")
        )

    def test_tfhub_remote_path(self):
        self.assertTrue(file_utils.is_remote_path("/tfhub/some/path"))

    def test_cfs_remote_path(self):
        self.assertTrue(file_utils.is_remote_path("/cfs/some/path"))

    def test_readahead_remote_path(self):
        self.assertTrue(file_utils.is_remote_path("/readahead/some/path"))

    def test_non_remote_paths(self):
        self.assertFalse(file_utils.is_remote_path("/local/path/to/file.txt"))
        self.assertFalse(
            file_utils.is_remote_path("C:\\local\\path\\on\\windows\\file.txt")
        )
        self.assertFalse(file_utils.is_remote_path("~/relative/path/"))
        self.assertFalse(file_utils.is_remote_path("./another/relative/path"))
        self.assertFalse(file_utils.is_remote_path("/local/path"))
        self.assertFalse(file_utils.is_remote_path("./relative/path"))
        self.assertFalse(file_utils.is_remote_path("~/relative/path"))


class TestRaiseIfNoGFile(test_case.TestCase):
    def test_raise_if_no_gfile_raises_correct_message(self):
        path = "gs://bucket/some/file.txt"
        expected_error_msg = (
            "Handling remote paths requires installing TensorFlow "
            f".*Received path: {path}"
        )
        with self.assertRaisesRegex(ValueError, expected_error_msg):
            file_utils._raise_if_no_gfile(path)

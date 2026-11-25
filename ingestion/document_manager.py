
import hashlib
import json

from pathlib import Path


class DocumentManager:

    def __init__(self, upload_directory="data/uploads",
                 manifest_path="data/processed/manifest.json"):

        self.upload_directory = Path(upload_directory)
        self.manifest_path = Path(manifest_path)

    def calculate_checksum(self, file_path):
        """
        Calculate SHA-256 checksum of a PDF.
        """

        sha256 = hashlib.sha256()

        with open(file_path, "rb") as file:

            for block in iter(lambda: file.read(4096), b""):
                sha256.update(block)

        return sha256.hexdigest()

    def get_current_manifest(self):
        """
        Generate metadata for all uploaded PDFs.
        """

        manifest = {}

        pdf_files = sorted(
            self.upload_directory.glob("*.pdf")
        )

        for pdf_file in pdf_files:

            manifest[pdf_file.name] = {
                "checksum": self.calculate_checksum(pdf_file),
                "size": pdf_file.stat().st_size
            }

        return manifest

    def load_saved_manifest(self):
        """
        Load previously saved document metadata.
        """

        if not self.manifest_path.exists():
            return None

        with open(self.manifest_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def save_manifest(self, manifest):
        """
        Save current document metadata.
        """

        self.manifest_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(self.manifest_path, "w", encoding="utf-8") as file:
            json.dump(
                manifest,
                file,
                indent=2
            )

    def documents_have_changed(self):
        """
        Compare current PDFs with the saved manifest.
        """

        current_manifest = self.get_current_manifest()
        saved_manifest = self.load_saved_manifest()

        if saved_manifest is None:
            return True

        return current_manifest != saved_manifest


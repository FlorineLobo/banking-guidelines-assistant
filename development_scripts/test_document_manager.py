
from ingestion.document_manager import DocumentManager


manager = DocumentManager()

print("\nSTEP 1: CHECKING CURRENT DOCUMENTS")

current_manifest = manager.get_current_manifest()

for filename, details in current_manifest.items():

    print("\nDocument:", filename)
    print("Size:", details["size"], "bytes")
    print("SHA-256:", details["checksum"])


print("\nSTEP 2: CHECKING WHETHER DOCUMENTS CHANGED")

changed = manager.documents_have_changed()

print("Documents changed:", changed)


assert len(current_manifest) == 3

print("\nDOCUMENT MANAGER TEST PASSED!")

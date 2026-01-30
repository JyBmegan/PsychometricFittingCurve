# Data Transfer Incident Log

## 1. General Information

*Date: 2026-01-30*

Storage Device: Lenovo 1TB External Hard Drive (Verified Authentic)

File Type: .pth (PyTorch Model Weights) & .zip Archives

Workflow: Linux (Source) → macOS (Intermediary) → Windows (Target)

## 2. Problem Description
Files copied from the Linux environment were invisible or appeared as 0-byte placeholders when accessed on Windows/macOS.

The folder structure and metadata (folder names) were present, but actual data blocks were missing.

Automatic macOS system folders (e.g., .fseventsd, .Spotlight-V100) were generated on the drive, confirming the hardware was recognized, but user data remained inaccessible.

## 3. Root Cause Analysis

1. The Linux kernel utilizes an asynchronous write buffer. While the File Manager UI indicated completion, the physical data had not been committed to the disk platters/flash cells.

2. Even though the *Eject* button was used, the OS likely unmounted the drive before the write buffer was fully flushed, or the file system index remained in a dirty state.

3. The drive name (ZX2 1TB) contains a space, which caused navigation errors in the Linux Terminal when not properly escaped or quoted.

## 4. Resolution & Actions Taken

I renamed the file/archive within the Linux environment after the initial copy. This action forced the file system to update the Inode/Metadata table and flush the cache. After renaming, the file became visible and accessible on the Windows machine.

The archive was successfully extracted on the target machine. Since ZIP/TAR formats use CRC/MD5 checksums, a successful extraction confirms 100% data integrity.

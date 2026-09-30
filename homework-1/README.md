# Homework 1 - CLEF RAG Setup

## Download the CLEF corpus with Git LFS

The CLEF corpus archive, `homework-1/clef-rag/data.zip`, is stored with Git LFS.
Install [Git LFS](https://git-lfs.com/) before downloading the archive.

For a new clone, replace `<repository-url>` with this repository's clone URL:

```bash
git lfs install
git clone <repository-url> fall-2026-interest-group-projects
cd fall-2026-interest-group-projects
git lfs pull
```

For an existing checkout, run from the repository root:

```bash
git lfs install
git pull
git lfs pull
```

`git lfs pull` downloads the large files and replaces their pointer files with
the actual contents. To download only the CLEF archive, use:

```bash
git lfs pull --include="homework-1/clef-rag/data.zip"
```

See the [CLEF RAG README](homework-1/clef-rag/README.md) for setup and instructions
on asking questions about the papers.

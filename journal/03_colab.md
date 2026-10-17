# 03 · Running things on Google Colab

## Option A: open the notebook in Colab (what we do now)

1. Click the **Open in Colab** badge at the top of a notebook on GitHub. Or in Colab: *File → Open notebook → GitHub* and paste the repo URL.
2. *Runtime → Change runtime type → T4 GPU* (for notebook 01).
3. Run cells top to bottom.
4. To save changes back to GitHub: *File → Save a copy in GitHub* and pick our repo + `main`. Write a short commit message.
   - ⚠️ If two of us edit the same notebook at the same time, the second save overwrites the first. Tell the group chat before you save a notebook to GitHub.

Private repo? Colab asks you to authorise GitHub once. The notebook's clone step also needs a `GITHUB_TOKEN` secret (see the notebook).

## Option B: Google's Colab CLI (from the terminal / Claude Code)

Researched 2026-10-17. **Not set up yet.**

- Released June 2026, open source: https://github.com/googlecolab/google-colab-cli
- Install: `uv tool install google-colab-cli`. Linux/macOS only.
- Gets a Colab GPU from the terminal and runs code on it:
  - `colab run --gpu T4 script.py`: start a VM, run, download outputs, shut down
  - `colab new -s NAME --gpu T4`, then `colab exec -s NAME -f notebook.ipynb`, `colab upload`, `colab download`
  - `colab usage` shows compute units
- Needs a one-time Google login (OAuth). *Unclear:* how much a **free** account can use GPUs through it. Check `colab usage` after login.
- Useful later for: running the batch test over all scans without clicking through the notebook.

## Option C: Colab MCP server

Released March 2026: https://github.com/googlecolab/colab-mcp

- Lets an AI agent (e.g. Claude Code) add and run cells *inside a notebook you have open in the browser*.
- Nice for pair-programming in a notebook. Not needed for now.

# Uploading the source to GitHub

1. Extract the ZIP into a new local folder. The archive contains repository files at its root.
2. Open the intended repository on GitHub and choose **Add file → Upload files**.
3. Upload the extracted files and directories, preserving their relative paths. Upload the contents, not the enclosing extraction folder and not the ZIP itself. GitHub stores an uploaded ZIP as a single file; it does not expand it into source files.
4. Include `.gitignore`; enable hidden-file display in your file picker if necessary.
5. Review the upload list and proposed changes. Replace the existing README and LICENSE only after checking that the changes match your intent. Commit the source.

This bundle contains no `.git` history or credentials. It does not need access to an account to be extracted. No GitHub workflow is included, so uploading these files does not itself launch a build or model evaluation.

After publishing, validate the source in an execution environment before describing it as tested or creating a release. See VALIDATION.md and DEVELOPMENT_PLAN.md.

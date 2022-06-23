# GRAIMatter
Repository for GRAIMatter project: Guidelines and Resource for AI Model Access from Trusted Research Environments

## Contents

- WP1/notes/datasets.MD includes a list of potential datasets

quick link to docs page: https://uwe-ai-group.github.io/GRAIMATTER-TEST/


## Git contribution guidelines

Please try, where possible, to adhere to the following guidelines when working with this project repository. Note that the nature of the project is such that it may not always be practical / possible to follow these completely. If you're unsure about any of this, just ask!:

1. Code in the main branch should be fairly complete, documented, and ready for others to use.
1. Direct modifications to the main branch are ok for updates to notes, meeting agendas etc (although doing these through new branches is ultimately safer).
1. In general, data should not be added to the repository [need to find an alternative], nor should large model files. There will almost certainly be exceptions to this where, e.g. training takes forever, but the default should be to leave large files out.
1. If code makes use of an open dataset, please ensure the code has clear comments describing how other users can download the data.
1. All pull requests to the main that include code should be reviewed by at least one other project member.
1. "Projects" have been setup for WP1 and WP2. When new issues are created, they can be added to the project when created (they don't get added automatically). They can then be viewed on the kanban board for the project they've been added.
1. Try and use issues so keep work visible to all team members.
1. Try and make use of the github discussion threads -- it's a more permanent and easily accessible discussion forum than email / teams.
1. Try and add plenty of detail to issues.
1. Give branches a useful (but short) name. E.g. if a branch is created to deal with issue 5, which is building a wrapper for a certain method, then a name like 5-wrapper-rf would be good.

## Pre-commit checks

[Pre-commit](https://pre-commit.com) is a tool for automatically checking and fixing contributions before they are committed to the repository.

A `.pre-commit-config.yaml` file is located in the project root directory and contains hooks to repositories for:
* automatically removing trailing whitespace;
* formatting Python code with [black](https://github.com/psf/black);
* checking for spelling mistakes with [codespell](https://github.com/codespell-project/codespell);
* automatically upgrading Python code to the newest syntax with [pyupgrade](https://github.com/asottile/pyupgrade);

It can be installed with:

```
$ pip install pre-commit
```

And run on all files with:
(Note: often it's useful to re-run a second time if changes are automatically made.)

```
$ pre-commit run --all-files
```

It can also be installed as a git-hook, automatically running on every `git commit`:

```
$ pre-commit install
```

Note that if installed as a git-hook, it will prevent the `git commit` from executing if any checks fail; checks can be disabled by adding `--no-verify` to the `git commit`.

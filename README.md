<!--
<p align="center">
  <img src="docs/source/logo.png" height="150">
</p>
-->

<h1 align="center">
  Multi-STonKGs
</h1>

<p align="center">
    <a href="https://github.com/multi-stonkgs/multi-stonkgs/actions?query=workflow%3ATests">
        <img alt="Tests" src="https://github.com/multi-stonkgs/multi-stonkgs/workflows/Tests/badge.svg" />
    </a>
    <a href="https://pypi.org/project/multi_stonkgs">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/multi_stonkgs" />
    </a>
    <a href="https://pypi.org/project/multi_stonkgs">
        <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/multi_stonkgs" />
    </a>
    <a href="https://github.com/multi-stonkgs/multi-stonkgs/blob/main/LICENSE">
        <img alt="PyPI - License" src="https://img.shields.io/pypi/l/multi_stonkgs" />
    </a>
    <a href='https://multi_stonkgs.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/multi_stonkgs/badge/?version=latest' alt='Documentation Status' />
    </a>
</p>

todo

## 💪 Getting Started

> TODO show in a very small amount of space the **MOST** useful thing your package can do.
Make it as short as possible! You have an entire set of docs for later.

### Command Line Interface

The multi_stonkgs command line tool is automatically installed. It can
be used from the shell with the `--help` flag to show all subcommands:

```shell
$ multi_stonkgs --help
```

> TODO show the most useful thing the CLI does! The CLI will have document auto-generated
by sphinx.

## ⬇️ Installation

The most recent release can be installed from
[PyPI](https://pypi.org/project/multi_stonkgs/) with:

```bash
$ pip install multi_stonkgs
```

The most recent code and data can be installed directly from GitHub with:

```bash
$ pip install git+https://github.com/multi-stonkgs/multi-stonkgs.git
```

To install in development mode, use the following:

```bash
$ git clone git+https://github.com/multi-stonkgs/multi-stonkgs.git
$ cd multi-stonkgs
$ pip install -e .
```

## ⚖️ License

The code in this package is licensed under the MIT License.

## 🛠️ Development

The final section of the README is for if you want to get involved by making a code contribution.

### ❓ Testing

After cloning the repository and installing `tox` with `pip install tox`, the unit tests in the `tests/` folder can be
run reproducibly with:

```shell
$ tox
```

Additionally, these tests are automatically re-run with each commit in a [GitHub Action](https://github.com/multi-stonkgs/multi-stonkgs/actions?query=workflow%3ATests).

### 📦 Making a Release

After installing the package in development mode and installing
`tox` with `pip install tox`, the commands for making a new release are contained within the `finish` environment
in `tox.ini`. Run the following from the shell:

```shell
$ tox -e finish
```

This script does the following:

1. Uses BumpVersion to switch the version number in the `setup.cfg` and
   `src/multi_stonkgs/version.py` to not have the `-dev` suffix
2. Packages the code in both a tar archive and a wheel
3. Uploads to PyPI using `twine`. Be sure to have a `.pypirc` file configured to avoid the need for manual input at this
   step
4. Push to GitHub. You'll need to make a release going with the commit where the version was bumped.
5. Bump the version to the next patch. If you made big changes and want to bump the version by minor, you can
   use `tox -e bumpversion minor` after.

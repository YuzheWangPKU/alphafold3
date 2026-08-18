# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run AF3 source from this checkout with image-built binary data."""

from __future__ import annotations

import os
import pathlib
import runpy
import subprocess
import sys


_STATIC_PACKAGE_ROOT_ENV = 'AF3_STATIC_PACKAGE_ROOT'
_DEFAULT_STATIC_PACKAGE_ROOT = pathlib.Path(
    '/alphafold3_venv/lib/python3.12/site-packages/alphafold3'
)


def _relative_to(path: pathlib.Path, root: pathlib.Path) -> bool:
  try:
    path.relative_to(root)
  except ValueError:
    return False
  return True


def _git_commit(checkout: pathlib.Path) -> str:
  try:
    result = subprocess.run(
        ['git', '-C', os.fspath(checkout), 'rev-parse', 'HEAD'],
        check=False,
        capture_output=True,
        text=True,
    )
  except OSError as exc:
    return f'unavailable ({exc})'
  if result.returncode:
    return f'unavailable ({result.stderr.strip()})'
  return result.stdout.strip()


def _prepare_imports() -> pathlib.Path:
  checkout = pathlib.Path(__file__).resolve().parent
  source_root = (checkout / 'src').resolve()
  static_package_root = pathlib.Path(
      os.environ.get(
          _STATIC_PACKAGE_ROOT_ENV, os.fspath(_DEFAULT_STATIC_PACKAGE_ROOT)
      )
  ).resolve()
  required_static_paths = (
      static_package_root / 'cpp.cpython-312-x86_64-linux-gnu.so',
      static_package_root / 'constants/converters/ccd.pickle',
      static_package_root
      / 'constants/converters/chemical_component_sets.pickle',
  )
  missing = [path for path in required_static_paths if not path.is_file()]
  if missing:
    raise RuntimeError(f'AF3 image-built runtime assets are missing: {missing}')

  sys.path.insert(0, os.fspath(source_root))
  import alphafold3  # pylint: disable=g-import-not-at-top

  imported_source = pathlib.Path(alphafold3.__file__).resolve()
  if not _relative_to(imported_source, source_root):
    raise RuntimeError(
        'Imported alphafold3 outside the cyclic-topology checkout:'
        f' {imported_source} (expected below {source_root}).'
    )
  alphafold3.__path__.append(os.fspath(static_package_root))

  from alphafold3 import cpp  # pylint: disable=g-import-not-at-top
  from alphafold3.common import resources  # pylint: disable=g-import-not-at-top

  imported_cpp = pathlib.Path(cpp.__file__).resolve()
  if not _relative_to(imported_cpp, static_package_root):
    raise RuntimeError(
        'Did not import the image-built AF3 extension:'
        f' {imported_cpp} (expected below {static_package_root}).'
    )

  # The pure-Python resource loader comes from the checkout, whose source tree
  # intentionally does not contain the generated 539 MB CCD pickle. Point only
  # its immutable data root at the ABI-matched static installation in the SIF.
  resources._DATA_ROOT = static_package_root  # pylint: disable=protected-access
  resources.ROOT = static_package_root

  print(f'CYCLIC_AF3_CHECKOUT={checkout}', flush=True)
  print(f'CYCLIC_AF3_COMMIT={_git_commit(checkout)}', flush=True)
  print(f'CYCLIC_AF3_PYTHON_SOURCE={imported_source}', flush=True)
  print(f'CYCLIC_AF3_CPP_SOURCE={imported_cpp}', flush=True)
  print(f'CYCLIC_AF3_DATA_ROOT={resources.ROOT}', flush=True)
  return checkout


def main() -> None:
  checkout = _prepare_imports()
  runpy.run_path(os.fspath(checkout / 'run_alphafold.py'), run_name='__main__')


if __name__ == '__main__':
  main()

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

"""Peptide cyclic-topology chemistry and representation features."""

from __future__ import annotations

from collections import Counter
from collections import deque
from collections.abc import Sequence
import dataclasses

from alphafold3 import structure
from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.constants import mmcif_names
from alphafold3.constants import residue_names
from alphafold3.model.atom_layout import atom_layout
import numpy as np


MAX_TOPOLOGY_TOKENS = 64
MAX_TOPOLOGY_EDGES = 16

_ACID_ATOMS = {
    ('ASP', 'CG'): ('OD2', 'OD1'),
    ('GLU', 'CD'): ('OE2', 'OE1'),
}

AtomId = tuple[str, int, str]


@dataclasses.dataclass(frozen=True, order=True)
class TopologyEdge:
  """One validated intrapeptide topology edge in input atom identity."""

  atom1: AtomId
  atom2: AtomId
  kind: str

  def __post_init__(self) -> None:
    if self.kind not in {
        'head_to_tail',
        'disulfide',
        'isopeptide',
        'head_lariat',
        'tail_lariat',
    }:
      raise ValueError(f'Unsupported topology edge kind: {self.kind!r}.')
    if self.atom1 == self.atom2:
      raise ValueError('A topology edge must connect two distinct atoms.')


@dataclasses.dataclass(frozen=True)
class _AtomRecord:
  res_name: str
  chain_type: str


@dataclasses.dataclass(frozen=True)
class _LeavingAtomSpec:
  leaving_atom: AtomId
  centre_atom: AtomId
  retained_atom: AtomId | None


def _atom_records(struc: structure.Structure) -> dict[AtomId, _AtomRecord]:
  records = {}
  for chain_id, res_id, atom_name, res_name, chain_type in zip(
      struc.chain_id,
      struc.res_id,
      struc.atom_name,
      struc.res_name,
      struc.chain_type,
      strict=True,
  ):
    atom_id = (str(chain_id), int(res_id), str(atom_name))
    if atom_id in records:
      raise ValueError(f'Atom identity is not unique: {atom_id}.')
    records[atom_id] = _AtomRecord(
        res_name=str(res_name),
        chain_type=str(chain_type),
    )
  return records


def _normalised_pair(atom1: AtomId, atom2: AtomId) -> tuple[AtomId, AtomId]:
  return tuple(sorted((atom1, atom2)))  # pyrefly: ignore[bad-return-type]


def classify_declared_topology_edges(
    struc: structure.Structure,
    bonded_atom_pairs: Sequence[tuple[AtomId, AtomId]] | None,
) -> tuple[TopologyEdge, ...]:
  """Validate and classify supported declared polymer-polymer bonds."""
  if not bonded_atom_pairs:
    return ()

  records = _atom_records(struc)
  chain_residue_ids = {
      str(chain_id): {int(res_id) for _, res_id in residues}
      for chain_id, residues in struc.all_residues.items()
  }

  seen_pairs: set[tuple[AtomId, AtomId]] = set()
  declared_endpoint_uses: Counter[AtomId] = Counter()
  h2t_chains: set[str] = set()
  topology_edges = []
  polymer_types = set(mmcif_names.POLYMER_CHAIN_TYPES)
  peptide_types = set(mmcif_names.PEPTIDE_CHAIN_TYPES)

  for raw_atom1, raw_atom2 in bonded_atom_pairs:
    atom1 = (str(raw_atom1[0]), int(raw_atom1[1]), str(raw_atom1[2]))
    atom2 = (str(raw_atom2[0]), int(raw_atom2[1]), str(raw_atom2[2]))
    if atom1 == atom2:
      raise ValueError(f'Declared bond has the same endpoint twice: {atom1}.')
    if atom1 not in records or atom2 not in records:
      missing = [atom for atom in (atom1, atom2) if atom not in records]
      raise ValueError(f'Declared bond endpoint(s) are missing: {missing}.')
    pair = _normalised_pair(atom1, atom2)
    if pair in seen_pairs:
      raise ValueError(
          'Duplicate or reversed duplicate declared bond: '
          f'{atom1} - {atom2}.'
      )
    seen_pairs.add(pair)
    declared_endpoint_uses.update((atom1, atom2))

    record1 = records[atom1]
    record2 = records[atom2]
    if not (
        record1.chain_type in polymer_types
        and record2.chain_type in polymer_types
    ):
      continue
    if atom1[0] != atom2[0]:
      raise ValueError(
          'Cyclic-topology support is limited to intrapeptide topology'
          f' edges only, got {atom1} - {atom2}.'
      )
    if not (
        record1.chain_type in peptide_types
        and record2.chain_type in peptide_types
    ):
      raise ValueError(
          'Cyclic-topology support is limited to peptide polymer bonds,'
          f' got {record1.chain_type!r} and {record2.chain_type!r}.'
      )
    if atom1[1] == atom2[1]:
      raise ValueError(
          'A peptide topology edge must connect two distinct residues, got'
          f' {atom1} - {atom2}.'
      )

    chain_id = atom1[0]
    atom_names = {atom1[2], atom2[2]}
    if atom_names == {'N', 'C'}:
      n_atom = atom1 if atom1[2] == 'N' else atom2
      c_atom = atom1 if atom1[2] == 'C' else atom2
      if (
          records[n_atom].res_name
          not in residue_names.PROTEIN_TYPES_WITH_UNKNOWN
          or records[c_atom].res_name
          not in residue_names.PROTEIN_TYPES_WITH_UNKNOWN
      ):
        raise ValueError(
            'A head-to-tail edge requires protein or UNK backbone endpoints,'
            ' got'
            f' {records[n_atom].res_name} {n_atom} and'
            f' {records[c_atom].res_name} {c_atom}.'
        )
      first_residue = min(chain_residue_ids[chain_id])
      last_residue = max(chain_residue_ids[chain_id])
      if n_atom[1] != first_residue or c_atom[1] != last_residue:
        raise ValueError(
            'A head-to-tail edge must connect the first-residue N to the'
            f' last-residue C, got {n_atom} - {c_atom}.'
        )
      if chain_id in h2t_chains:
        raise ValueError(
            f'Peptide chain {chain_id!r} has multiple head-to-tail edges.'
        )
      h2t_chains.add(chain_id)
      topology_edges.append(
          TopologyEdge(atom1=n_atom, atom2=c_atom, kind='head_to_tail')
      )
      continue

    if atom1[2] == atom2[2] == 'SG':
      if record1.res_name != 'CYS' or record2.res_name != 'CYS':
        raise ValueError(
            'A disulfide edge requires CYS SG endpoints, got'
            f' {record1.res_name} {atom1} and {record2.res_name} {atom2}.'
        )
      ss_atom1, ss_atom2 = _normalised_pair(atom1, atom2)
      topology_edges.append(
          TopologyEdge(
              atom1=ss_atom1, atom2=ss_atom2, kind='disulfide'
          )
      )
      continue

    endpoints = ((atom1, record1), (atom2, record2))
    acid_endpoints = [
        atom
        for atom, record in endpoints
        if (record.res_name, atom[2]) in _ACID_ATOMS
    ]
    lys_endpoints = [
        atom
        for atom, record in endpoints
        if (record.res_name, atom[2]) == ('LYS', 'NZ')
    ]
    if len(acid_endpoints) == 1 and len(lys_endpoints) == 1:
      topology_edges.append(
          TopologyEdge(
              atom1=acid_endpoints[0],
              atom2=lys_endpoints[0],
              kind='isopeptide',
          )
      )
      continue

    terminal_n_endpoints = [
        atom
        for atom, record in endpoints
        if atom[2] == 'N'
        and atom[1] == min(chain_residue_ids[chain_id])
        and record.res_name in residue_names.PROTEIN_TYPES
    ]
    if len(acid_endpoints) == 1 and len(terminal_n_endpoints) == 1:
      n_atom = terminal_n_endpoints[0]
      if records[n_atom].res_name == 'PRO':
        raise ValueError(
            'A head-to-side-chain lactam cannot use N-terminal PRO.'
        )
      topology_edges.append(
          TopologyEdge(
              atom1=acid_endpoints[0],
              atom2=n_atom,
              kind='head_lariat',
          )
      )
      continue

    terminal_c_endpoints = [
        atom
        for atom, record in endpoints
        if atom[2] == 'C'
        and atom[1] == max(chain_residue_ids[chain_id])
        and record.res_name in residue_names.PROTEIN_TYPES
    ]
    if len(lys_endpoints) == 1 and len(terminal_c_endpoints) == 1:
      topology_edges.append(
          TopologyEdge(
              atom1=lys_endpoints[0],
              atom2=terminal_c_endpoints[0],
              kind='tail_lariat',
          )
      )
      continue

    raise ValueError(
        'Unsupported declared peptide-polymer topology edge:'
        f' {atom1} - {atom2}.'
    )

  if not topology_edges:
    return ()
  topology_edges = tuple(sorted(topology_edges))
  reused_endpoints = {
      endpoint: declared_endpoint_uses[endpoint]
      for edge in topology_edges
      for endpoint in (edge.atom1, edge.atom2)
      if declared_endpoint_uses[endpoint] != 1
  }
  if reused_endpoints:
    raise ValueError(
        'A topology endpoint is reused by another declared bond:'
        f' {reused_endpoints}.'
    )
  reused_leaving_atoms = {
      spec.leaving_atom: declared_endpoint_uses[spec.leaving_atom]
      for spec in _topology_leaving_atom_specs(topology_edges)
      if declared_endpoint_uses[spec.leaving_atom]
  }
  if reused_leaving_atoms:
    raise ValueError(
        'A topology leaving atom is reused by another declared bond:'
        f' {reused_leaving_atoms}.'
    )
  return topology_edges


def topology_bond_layout(
    reference_layout: atom_layout.AtomLayout,
    topology_edges: Sequence[TopologyEdge],
) -> atom_layout.AtomLayout:
  """Build an endpoint-exact atom layout for validated topology edges."""
  uid_to_index = {}
  for index, uid in enumerate(
      zip(
          reference_layout.chain_id.ravel(),
          reference_layout.res_id.ravel(),
          reference_layout.atom_name.ravel(),
          strict=True,
      )
  ):
    normalised_uid = (str(uid[0]), int(uid[1]), str(uid[2]))
    if normalised_uid in uid_to_index:
      raise ValueError(
          f'Reference atom layout has duplicate atom identity {normalised_uid}.'
      )
    uid_to_index[normalised_uid] = index

  endpoint_indices = []
  for edge in topology_edges:
    try:
      endpoint_indices.append(
          [uid_to_index[edge.atom1], uid_to_index[edge.atom2]]
      )
    except KeyError as e:
      raise ValueError(f'Topology endpoint is absent from atom layout: {e}.') from e
  return reference_layout[np.asarray(endpoint_indices, dtype=np.int64)]


def restore_topology_bonds(
    cleaned_struc: structure.Structure,
    topology_edges: Sequence[TopologyEdge],
) -> structure.Structure:
  """Restore topology edges to AF3's cleaned minimal structure."""
  existing_pairs = {
      _normalised_pair(
          (
              str(bond.from_atom['chain_id']),
              int(bond.from_atom['res_id']),
              str(bond.from_atom['atom_name']),
          ),
          (
              str(bond.dest_atom['chain_id']),
              int(bond.dest_atom['res_id']),
              str(bond.dest_atom['atom_name']),
          ),
      )
      for bond in cleaned_struc.iter_bonds()
  }
  topology_pairs = {
      _normalised_pair(edge.atom1, edge.atom2) for edge in topology_edges
  }
  duplicates = existing_pairs & topology_pairs
  if duplicates:
    raise ValueError(
        f'Cleaned structure unexpectedly retained topology bonds: {duplicates}.'
    )
  restored = cleaned_struc.add_bonds(
      [(edge.atom1, edge.atom2) for edge in topology_edges],
      bond_type=mmcif_names.COVALENT_BOND,
  )
  validate_topology_bonds(restored, topology_edges)
  return restored


def _topology_leaving_atom_specs(
    topology_edges: Sequence[TopologyEdge],
) -> tuple[_LeavingAtomSpec, ...]:
  specs = []
  for edge in topology_edges:
    if edge.kind in {'head_to_tail', 'tail_lariat'}:
      carbon_atoms = [
          atom for atom in (edge.atom1, edge.atom2) if atom[2] == 'C'
      ]
      if len(carbon_atoms) != 1:
        raise ValueError(f'{edge.kind} edge lacks one terminal C: {edge}.')
      carbon = carbon_atoms[0]
      specs.append(
          _LeavingAtomSpec(
              leaving_atom=(carbon[0], carbon[1], 'OXT'),
              centre_atom=carbon,
              retained_atom=None,
          )
      )
    elif edge.kind in {'isopeptide', 'head_lariat'}:
      acid_atoms = [
          atom for atom in (edge.atom1, edge.atom2) if atom[2] in {'CG', 'CD'}
      ]
      if len(acid_atoms) != 1:
        raise ValueError(f'Isopeptide edge lacks one acid carbon: {edge}.')
      acid = acid_atoms[0]
      removed_name, retained_name = {
          'CG': ('OD2', 'OD1'),
          'CD': ('OE2', 'OE1'),
      }[acid[2]]
      specs.append(
          _LeavingAtomSpec(
              leaving_atom=(acid[0], acid[1], removed_name),
              centre_atom=acid,
              retained_atom=(acid[0], acid[1], retained_name),
          )
      )
  duplicate_leaving_atoms = {
      atom_id: count
      for atom_id, count in Counter(
          spec.leaving_atom for spec in specs
      ).items()
      if count != 1
  }
  if duplicate_leaving_atoms:
    raise ValueError(
        'Multiple topology edges consume the same leaving atom:'
        f' {duplicate_leaving_atoms}.'
    )
  return tuple(specs)


def _ccd_component(
    ccd: chemical_components.Ccd, res_name: str
) -> dict[str, Sequence[str]]:
  component = ccd.get(res_name)
  if component is None:
    raise ValueError(f'CCD component {res_name} is unavailable.')
  return dict(component)


def _ccd_atom_values(
    component: dict[str, Sequence[str]],
    res_name: str,
    atom_name: str,
) -> tuple[str, str]:
  atom_names = list(component.get('_chem_comp_atom.atom_id', ()))
  indices = [index for index, name in enumerate(atom_names) if name == atom_name]
  if len(indices) != 1:
    raise ValueError(
        f'CCD component {res_name} must contain exactly one {atom_name} atom.'
    )
  index = indices[0]
  elements = list(component.get('_chem_comp_atom.type_symbol', ()))
  leaving_flags = list(
      component.get('_chem_comp_atom.pdbx_leaving_atom_flag', ())
  )
  if index >= len(elements) or index >= len(leaving_flags):
    raise ValueError(f'CCD component {res_name} has incomplete atom metadata.')
  return str(elements[index]).upper(), str(leaving_flags[index]).upper()


def _ccd_heavy_bonds(
    component: dict[str, Sequence[str]],
    res_name: str,
    atom_name: str,
) -> list[tuple[str, str]]:
  atom_names = list(component.get('_chem_comp_atom.atom_id', ()))
  elements = list(component.get('_chem_comp_atom.type_symbol', ()))
  if len(atom_names) != len(elements):
    raise ValueError(f'CCD component {res_name} has incomplete atom metadata.')
  element_by_atom = dict(zip(atom_names, elements, strict=True))
  if len(element_by_atom) != len(atom_names):
    raise ValueError(f'CCD component {res_name} has duplicate atom names.')

  atom1 = list(component.get('_chem_comp_bond.atom_id_1', ()))
  atom2 = list(component.get('_chem_comp_bond.atom_id_2', ()))
  orders = list(component.get('_chem_comp_bond.value_order', ()))
  if not (len(atom1) == len(atom2) == len(orders)):
    raise ValueError(f'CCD component {res_name} has incomplete bond metadata.')
  heavy_bonds = []
  for left, right, order in zip(atom1, atom2, orders, strict=True):
    if left == atom_name:
      neighbour = right
    elif right == atom_name:
      neighbour = left
    else:
      continue
    if neighbour not in element_by_atom:
      raise ValueError(
          f'CCD component {res_name} bond references unknown atom {neighbour}.'
      )
    if str(element_by_atom[neighbour]).upper() not in {'H', 'D'}:
      heavy_bonds.append((str(neighbour), str(order).upper()))
  return heavy_bonds


def _assert_acid_ccd_pattern(
    ccd: chemical_components.Ccd,
    res_name: str,
    centre_name: str,
    removed_name: str,
    retained_name: str,
) -> None:
  component = _ccd_component(ccd, res_name)
  centre_element, _ = _ccd_atom_values(component, res_name, centre_name)
  removed_element, removed_flag = _ccd_atom_values(
      component, res_name, removed_name
  )
  retained_element, retained_flag = _ccd_atom_values(
      component, res_name, retained_name
  )
  if (
      centre_element != 'C'
      or removed_element != 'O'
      or retained_element != 'O'
      or removed_flag != 'N'
      or retained_flag != 'N'
      or _ccd_heavy_bonds(component, res_name, removed_name)
      != [(centre_name, 'SING')]
      or _ccd_heavy_bonds(component, res_name, retained_name)
      != [(centre_name, 'DOUB')]
  ):
    raise ValueError(
        f'CCD component {res_name} does not have the required unique'
        f' {centre_name}-{removed_name} single /'
        f' {centre_name}-{retained_name} double pattern.'
    )


def _assert_terminal_oxt_ccd_pattern(
    ccd: chemical_components.Ccd, res_name: str
) -> None:
  component = _ccd_component(ccd, res_name)
  carbon_element, _ = _ccd_atom_values(component, res_name, 'C')
  oxt_element, oxt_flag = _ccd_atom_values(component, res_name, 'OXT')
  if (
      carbon_element != 'C'
      or oxt_element != 'O'
      or oxt_flag != 'Y'
      or _ccd_heavy_bonds(component, res_name, 'OXT') != [('C', 'SING')]
  ):
    raise ValueError(
        f'CCD component {res_name} does not have the required terminal'
        ' C-OXT single-bond leaving-group pattern.'
    )


def filter_topology_leaving_atoms(
    stock_flat_layout: atom_layout.AtomLayout,
    topology_edges: Sequence[TopologyEdge],
    ccd: chemical_components.Ccd,
) -> atom_layout.AtomLayout:
  """Remove exactly the heavy atoms consumed by declared amide closures."""
  specs = _topology_leaving_atom_specs(topology_edges)
  if not specs:
    return stock_flat_layout

  flat_atom_ids = [
      (str(chain_id), int(res_id), str(atom_name))
      for chain_id, res_id, atom_name in zip(
          stock_flat_layout.chain_id,
          stock_flat_layout.res_id,
          stock_flat_layout.atom_name,
          strict=True,
      )
  ]
  counts = Counter(flat_atom_ids)
  res_name_by_atom = {
      atom_id: str(res_name)
      for atom_id, res_name in zip(
          flat_atom_ids, stock_flat_layout.res_name, strict=True
      )
  }
  required_atoms = {
      atom_id
      for spec in specs
      for atom_id in (
          spec.leaving_atom,
          spec.centre_atom,
          *((spec.retained_atom,) if spec.retained_atom is not None else ()),
      )
  }
  invalid_counts = {
      atom_id: counts[atom_id]
      for atom_id in required_atoms
      if counts[atom_id] != 1
  }
  if invalid_counts:
    raise ValueError(
        'Topology amide chemistry requires exactly one of each center,'
        f' leaving, and retained atom, got {invalid_counts}.'
    )

  checked_patterns = set()
  for spec in specs:
    res_name = res_name_by_atom[spec.centre_atom]
    if spec.retained_atom is None:
      signature = ('terminal_oxt', res_name)
      if signature not in checked_patterns:
        _assert_terminal_oxt_ccd_pattern(ccd, res_name)
        checked_patterns.add(signature)
    else:
      signature = ('acid', res_name, spec.centre_atom[2])
      if signature not in checked_patterns:
        expected_res_name = {'CG': 'ASP', 'CD': 'GLU'}[spec.centre_atom[2]]
        if res_name != expected_res_name:
          raise ValueError(
              'Isopeptide acid endpoint residue disagrees with its atom name:'
              f' {res_name} {spec.centre_atom}.'
          )
        _assert_acid_ccd_pattern(
            ccd,
            res_name,
            spec.centre_atom[2],
            spec.leaving_atom[2],
            spec.retained_atom[2],
        )
        checked_patterns.add(signature)

  leaving_atom_ids = {spec.leaving_atom for spec in specs}
  keep_mask = np.array(
      [atom_id not in leaving_atom_ids for atom_id in flat_atom_ids],
      dtype=bool,
  )
  filtered = stock_flat_layout[keep_mask]
  filtered_atom_ids = Counter(
      (str(chain_id), int(res_id), str(atom_name))
      for chain_id, res_id, atom_name in zip(
          filtered.chain_id,
          filtered.res_id,
          filtered.atom_name,
          strict=True,
      )
  )
  removed = Counter(flat_atom_ids) - filtered_atom_ids
  expected_removed = Counter(leaving_atom_ids)
  if removed != expected_removed:
    raise ValueError(
        'Topology atom inventory changed by more than the exact leaving-atom'
        f' set: removed={removed}, expected={expected_removed}.'
    )
  return filtered


def validate_topology_bonds(
    struc: structure.Structure,
    topology_edges: Sequence[TopologyEdge],
) -> None:
  """Require every declared topology edge exactly once in a structure graph."""
  expected_pairs = Counter(
      _normalised_pair(edge.atom1, edge.atom2) for edge in topology_edges
  )
  observed_pairs = Counter()
  for bond in struc.iter_bonds():
    pair = _normalised_pair(
        (
            str(bond.from_atom['chain_id']),
            int(bond.from_atom['res_id']),
            str(bond.from_atom['atom_name']),
        ),
        (
            str(bond.dest_atom['chain_id']),
            int(bond.dest_atom['res_id']),
            str(bond.dest_atom['atom_name']),
        ),
    )
    if pair in expected_pairs:
      observed_pairs[pair] += 1
  if observed_pairs != expected_pairs:
    raise ValueError(
        'Structure topology disagrees with declared input graph:'
        f' observed={observed_pairs}, expected={expected_pairs}.'
    )

  atom_ids = Counter(
      (str(chain_id), int(res_id), str(atom_name))
      for chain_id, res_id, atom_name in zip(
          struc.chain_id, struc.res_id, struc.atom_name, strict=True
      )
  )
  for edge in topology_edges:
    if atom_ids[edge.atom1] != 1 or atom_ids[edge.atom2] != 1:
      raise ValueError(
          'Topology endpoint atom inventory is invalid for edge'
          f' {edge}: counts=({atom_ids[edge.atom1]}, {atom_ids[edge.atom2]}).'
      )


def validate_topology_output(
    struc: structure.Structure,
    topology_edges: Sequence[TopologyEdge],
) -> None:
  """Validate the declared graph and final topology-specific atom inventory."""
  validate_topology_bonds(struc, topology_edges)
  atom_ids = Counter(
      (str(chain_id), int(res_id), str(atom_name))
      for chain_id, res_id, atom_name in zip(
          struc.chain_id, struc.res_id, struc.atom_name, strict=True
      )
  )
  leaving_specs = _topology_leaving_atom_specs(topology_edges)
  retained_leaving_atoms = {
      spec.leaving_atom
      for spec in leaving_specs
      if atom_ids[spec.leaving_atom]
  }
  if retained_leaving_atoms:
    raise ValueError(
        'Topology output retained amide leaving atoms'
        f' {retained_leaving_atoms}.'
    )
  missing_retained_atoms = {
      spec.retained_atom: atom_ids[spec.retained_atom]
      for spec in leaving_specs
      if spec.retained_atom is not None
      and atom_ids[spec.retained_atom] != 1
  }
  if missing_retained_atoms:
    raise ValueError(
        'Topology output has invalid retained acid-oxygen inventory:'
        f' {missing_retained_atoms}.'
    )


def _structure_atom_inventory(
    struc: structure.Structure,
) -> tuple[tuple[str, int, str, str, str], ...]:
  return tuple(
      (
          str(chain_id),
          int(res_id),
          str(res_name),
          str(atom_name),
          str(element),
      )
      for chain_id, res_id, res_name, atom_name, element in zip(
          struc.chain_id,
          struc.res_id,
          struc.res_name,
          struc.atom_name,
          struc.atom_element,
          strict=True,
      )
  )


def _structure_bond_inventory(
    struc: structure.Structure,
) -> Counter[tuple[AtomId, AtomId, str]]:
  inventory = Counter()
  for bond in struc.iter_bonds():
    atom1, atom2 = _normalised_pair(
        (
            str(bond.from_atom['chain_id']),
            int(bond.from_atom['res_id']),
            str(bond.from_atom['atom_name']),
        ),
        (
            str(bond.dest_atom['chain_id']),
            int(bond.dest_atom['res_id']),
            str(bond.dest_atom['atom_name']),
        ),
    )
    inventory[(atom1, atom2, str(bond.bond_info['type']))] += 1
  return inventory


def validate_structure_identity(
    actual: structure.Structure,
    expected: structure.Structure,
) -> None:
  """Validate canonical atom order/inventory and complete chemical graph."""
  actual_atoms = _structure_atom_inventory(actual)
  expected_atoms = _structure_atom_inventory(expected)
  if actual_atoms != expected_atoms:
    raise ValueError('Output atom inventory or serialization order changed.')
  actual_bonds = _structure_bond_inventory(actual)
  expected_bonds = _structure_bond_inventory(expected)
  if actual_bonds != expected_bonds:
    raise ValueError(
        'Output chemical graph changed:'
        f' observed={actual_bonds}, expected={expected_bonds}.'
    )


def validate_serialized_structure(
    expected: structure.Structure, serialized_mmcif: str | bytes
) -> None:
  """Parse a serialized prediction and require an exact graph/inventory roundtrip."""
  parsed = structure.from_mmcif(
      serialized_mmcif,
      include_water=True,
      include_other=True,
      include_bonds=True,
  )
  validate_structure_identity(parsed, expected)


def _residue_positions_for_chain(
    layout: atom_layout.AtomLayout, chain_id: str
) -> tuple[int, ...]:
  positions = tuple(
      sorted(
          {
              int(res_id)
              for res_id in layout.res_id[layout.chain_id == chain_id]
          }
      )
  )
  if not positions:
    raise ValueError(f'Topology chain {chain_id!r} is absent from layout.')
  if positions != tuple(range(positions[0], positions[-1] + 1)):
    raise ValueError(
        f'Topology chain {chain_id!r} has non-contiguous residue IDs:'
        f' {positions}.'
    )
  return positions


def _residue_graph_edges(
    positions: tuple[int, ...], topology_edges: Sequence[TopologyEdge]
) -> set[tuple[int, int]]:
  graph_edges = {
      (left, right) for left, right in zip(positions[:-1], positions[1:])
  }
  for edge in topology_edges:
    graph_edges.add(tuple(sorted((edge.atom1[1], edge.atom2[1]))))
  return graph_edges


def _shortest_path_distances(
    positions: tuple[int, ...], graph_edges: set[tuple[int, int]]
) -> dict[tuple[int, int], int]:
  adjacency = {position: set() for position in positions}
  for left, right in graph_edges:
    adjacency[left].add(right)
    adjacency[right].add(left)
  distances = {}
  for source in positions:
    source_distances = {source: 0}
    queue = deque([source])
    while queue:
      current = queue.popleft()
      for neighbour in sorted(adjacency[current]):
        if neighbour not in source_distances:
          source_distances[neighbour] = source_distances[current] + 1
          queue.append(neighbour)
    if len(source_distances) != len(positions):
      raise ValueError('The topology residue graph must be connected.')
    for target, distance in source_distances.items():
      distances[(source, target)] = distance
  return distances


def build_rpe_override(
    all_tokens: atom_layout.AtomLayout,
    topology_edges: Sequence[TopologyEdge],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Build automatic H2T signed shortest-path residue offsets."""
  selected_edges = tuple(
      edge for edge in topology_edges if edge.kind == 'head_to_tail'
  )
  token_indices = np.zeros(MAX_TOPOLOGY_TOKENS, dtype=np.int32)
  token_mask = np.zeros(MAX_TOPOLOGY_TOKENS, dtype=bool)
  pair_offsets = np.zeros(
      (MAX_TOPOLOGY_TOKENS, MAX_TOPOLOGY_TOKENS), dtype=np.int32
  )
  if not selected_edges:
    return token_indices, token_mask, pair_offsets

  chain_to_edges: dict[str, list[TopologyEdge]] = {}
  for edge in selected_edges:
    if edge.atom1[0] != edge.atom2[0]:
      raise ValueError('Topology RPE supports intrachain edges only.')
    chain_to_edges.setdefault(edge.atom1[0], []).append(edge)

  selected_token_indices = np.flatnonzero(
      np.isin(all_tokens.chain_id, sorted(chain_to_edges))
  )
  if selected_token_indices.size > MAX_TOPOLOGY_TOKENS:
    raise ValueError(
        'Topology RPE exceeds the compact-token limit:'
        f' {selected_token_indices.size} > {MAX_TOPOLOGY_TOKENS}.'
    )
  token_indices[: selected_token_indices.size] = selected_token_indices
  token_mask[: selected_token_indices.size] = True

  selected_residue_ids = all_tokens.res_id[selected_token_indices].astype(
      np.int64
  )
  pair_offsets[
      : selected_token_indices.size, : selected_token_indices.size
  ] = (
      selected_residue_ids[:, None] - selected_residue_ids[None, :]
  ).astype(np.int32)
  compact_index = {
      int(token_index): index
      for index, token_index in enumerate(selected_token_indices)
  }

  for chain_id, chain_edges in sorted(chain_to_edges.items()):
    chain_token_indices = np.flatnonzero(all_tokens.chain_id == chain_id)
    chain_residue_ids = all_tokens.res_id[chain_token_indices].astype(np.int64)
    if len(set(chain_residue_ids.tolist())) != len(chain_residue_ids):
      raise ValueError(
          'Topology RPE requires exactly one token per peptide residue in'
          f' chain {chain_id!r}.'
      )
    positions = _residue_positions_for_chain(all_tokens, chain_id)
    graph_edges = _residue_graph_edges(positions, chain_edges)
    distances = _shortest_path_distances(positions, graph_edges)
    period = len(positions)
    half_period = period // 2
    for left_token, left_position in zip(
        chain_token_indices, chain_residue_ids, strict=True
    ):
      for right_token, right_position in zip(
        chain_token_indices, chain_residue_ids, strict=True
      ):
        baseline = int(left_position - right_position)
        if baseline > half_period:
          baseline -= period
        elif baseline < -half_period:
          baseline += period
        magnitude = distances[(int(left_position), int(right_position))]
        offset = 0 if baseline == 0 else int(np.sign(baseline)) * magnitude
        pair_offsets[
            compact_index[int(left_token)], compact_index[int(right_token)]
        ] = offset
  return token_indices, token_mask, pair_offsets


def validate_token_bond_pairs(
    topology_edges: Sequence[TopologyEdge],
    token_bond_pairs: Sequence[folding_input.TokenBondPair] | None,
) -> tuple[folding_input.TokenBondPair, ...]:
  """Require every token hint to match declared non-H2T atom edges."""
  if not token_bond_pairs:
    return ()

  edges_by_token_pair: dict[
      tuple[folding_input.TokenId, folding_input.TokenId], list[TopologyEdge]
  ] = {}
  for edge in topology_edges:
    pair = tuple(
        sorted(
            (
                (edge.atom1[0], edge.atom1[1]),
                (edge.atom2[0], edge.atom2[1]),
            )
        )
    )
    edges_by_token_pair.setdefault(pair, []).append(edge)  # pyrefly: ignore[bad-argument-type]

  normalised_pairs = []
  seen_pairs = set()
  for raw_pair in token_bond_pairs:
    if len(raw_pair) != 2:
      raise ValueError(
          f'Token-bond pair {raw_pair} must contain exactly two tokens.'
      )
    token1, token2 = raw_pair
    if (
        len(token1) != 2
        or len(token2) != 2
        or not isinstance(token1[0], str)
        or not isinstance(token2[0], str)
        or not isinstance(token1[1], int)
        or not isinstance(token2[1], int)
    ):
      raise ValueError(
          f'Invalid token-bond pair {raw_pair}; tokens must be'
          ' (chain_id: str, res_id: int).'
      )
    pair = tuple(sorted((tuple(token1), tuple(token2))))
    if pair in seen_pairs:
      raise ValueError(
          f'Duplicate or reversed duplicate token-bond pair: {raw_pair}.'
      )
    seen_pairs.add(pair)
    matching_edges = edges_by_token_pair.get(pair, [])
    if not matching_edges:
      raise ValueError(
          'Each token-bond pair must match at least one declared topology atom'
          f' edge, got none for {raw_pair}.'
      )
    if any(edge.kind == 'head_to_tail' for edge in matching_edges):
      raise ValueError(
          'Head-to-tail closure uses automatic RPE and cannot receive a'
          f' token-bond hint: {raw_pair}.'
      )
    normalised_pairs.append(pair)
  return tuple(sorted(normalised_pairs))  # pyrefly: ignore[bad-return-type]


def build_token_bond_indices(
    all_tokens: atom_layout.AtomLayout,
    token_bond_pairs: Sequence[folding_input.TokenBondPair],
) -> tuple[np.ndarray, np.ndarray]:
  """Build compact symmetric-hint endpoint token pairs."""
  edge_indices = np.zeros((MAX_TOPOLOGY_EDGES, 2), dtype=np.int32)
  edge_mask = np.zeros(MAX_TOPOLOGY_EDGES, dtype=bool)
  if len(token_bond_pairs) > MAX_TOPOLOGY_EDGES:
    raise ValueError(
        'Token-bond hints exceed the compact edge limit:'
        f' {len(token_bond_pairs)} > {MAX_TOPOLOGY_EDGES}.'
    )

  residue_to_token: dict[tuple[str, int], list[int]] = {}
  for token_index, (chain_id, res_id) in enumerate(
      zip(all_tokens.chain_id, all_tokens.res_id, strict=True)
  ):
    residue_to_token.setdefault((str(chain_id), int(res_id)), []).append(
        token_index
    )
  for edge_index, pair in enumerate(token_bond_pairs):
    endpoints = []
    for token_id in pair:
      token_matches = residue_to_token.get(token_id, [])
      if len(token_matches) != 1:
        raise ValueError(
            'Token-bond hints require one endpoint token per residue, got'
            f' {token_matches} for {token_id}.'
        )
      endpoints.append(token_matches[0])
    if endpoints[0] == endpoints[1]:
      raise ValueError(f'Token-bond hint collapsed to one token: {pair}.')
    edge_indices[edge_index] = endpoints
    edge_mask[edge_index] = True
  return edge_indices, edge_mask


def input_topology_edges(
    fold_input: folding_input.Input,
    struc: structure.Structure,
) -> tuple[TopologyEdge, ...]:
  """Return every validated supported polymer-polymer topology edge."""
  return classify_declared_topology_edges(struc, fold_input.bonded_atom_pairs)

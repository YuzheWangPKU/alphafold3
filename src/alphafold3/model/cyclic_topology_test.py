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

"""Behavior tests for peptide cyclic-topology support."""

from __future__ import annotations

from collections import Counter
import json
import types

from absl.testing import absltest
from alphafold3 import structure
from alphafold3.common import folding_input
from alphafold3.constants import mmcif_names
from alphafold3.model import cyclic_topology
from alphafold3.model import features
from alphafold3.model.atom_layout import atom_layout
from alphafold3.model.components import utils
from alphafold3.model.network import diffusion_head
from alphafold3.model.network import evoformer
from alphafold3.model.network import featurization
from alphafold3.model.pipeline import structure_cleaning
import jax.numpy as jnp
import numpy as np


_H2T = cyclic_topology.TopologyEdge(
    atom1=('P', 1, 'N'), atom2=('P', 5, 'C'), kind='head_to_tail'
)
_SS = cyclic_topology.TopologyEdge(
    atom1=('P', 2, 'SG'), atom2=('P', 4, 'SG'), kind='disulfide'
)
_ISO_ASP = cyclic_topology.TopologyEdge(
    atom1=('P', 3, 'CG'), atom2=('P', 2, 'NZ'), kind='isopeptide'
)
_ISO_GLU = cyclic_topology.TopologyEdge(
    atom1=('P', 4, 'CD'), atom2=('P', 2, 'NZ'), kind='isopeptide'
)
_HEAD = cyclic_topology.TopologyEdge(
    atom1=('P', 3, 'CG'), atom2=('P', 1, 'N'), kind='head_lariat'
)
_TAIL = cyclic_topology.TopologyEdge(
    atom1=('P', 2, 'NZ'), atom2=('P', 5, 'C'), kind='tail_lariat'
)


def _test_ccd():
  def component(atoms, elements, flags, bonds):
    return {
        '_chem_comp_atom.atom_id': list(atoms),
        '_chem_comp_atom.type_symbol': list(elements),
        '_chem_comp_atom.pdbx_leaving_atom_flag': list(flags),
        '_chem_comp_bond.atom_id_1': [bond[0] for bond in bonds],
        '_chem_comp_bond.atom_id_2': [bond[1] for bond in bonds],
        '_chem_comp_bond.value_order': [bond[2] for bond in bonds],
    }

  return {
      'GLY': component(
          ('C', 'OXT'),
          ('C', 'O'),
          ('N', 'Y'),
          (('C', 'OXT', 'SING'),),
      ),
      'ASP': component(
          ('CG', 'OD1', 'OD2'),
          ('C', 'O', 'O'),
          ('N', 'N', 'N'),
          (('CG', 'OD1', 'DOUB'), ('CG', 'OD2', 'SING')),
      ),
      'GLU': component(
          ('CD', 'OE1', 'OE2'),
          ('C', 'O', 'O'),
          ('N', 'N', 'N'),
          (('CD', 'OE1', 'DOUB'), ('CD', 'OE2', 'SING')),
      ),
  }


def _peptide_structure(*, bonds: tuple[cyclic_topology.TopologyEdge, ...]):
  residue_names = ('GLY', 'CYS', 'ALA', 'CYS', 'GLY')
  atom_names = []
  atom_elements = []
  res_ids = []
  res_names = []
  for res_id, res_name in enumerate(residue_names, start=1):
    names = ['N', 'CA', 'C', 'O']
    if res_name == 'CYS':
      names.extend(('CB', 'SG'))
    if res_id == len(residue_names):
      names.append('OXT')
    atom_names.extend(names)
    atom_elements.extend(name[0] for name in names)
    res_ids.extend([res_id] * len(names))
    res_names.extend([res_name] * len(names))

  atom_names_array = np.asarray(atom_names, dtype=object)
  res_ids_array = np.asarray(res_ids, dtype=np.int32)
  atom_keys = np.arange(len(atom_names), dtype=np.int64)
  key_by_id = {
      ('P', int(res_id), str(atom_name)): int(atom_key)
      for atom_key, res_id, atom_name in zip(
          atom_keys, res_ids_array, atom_names_array, strict=True
      )
  }
  bond_table = structure.Bonds(
      key=np.arange(len(bonds), dtype=np.int64),
      from_atom_key=np.asarray(
          [key_by_id[edge.atom1] for edge in bonds], dtype=np.int64
      ),
      dest_atom_key=np.asarray(
          [key_by_id[edge.atom2] for edge in bonds], dtype=np.int64
      ),
      type=np.asarray(
          [mmcif_names.COVALENT_BOND] * len(bonds), dtype=object
      ),
      role=np.asarray(['.'] * len(bonds), dtype=object),
  )
  return structure.from_atom_arrays(
      name='cyclic_topology_test',
      res_id=res_ids_array,
      all_residues={
          'P': tuple(
              (res_name, res_id)
              for res_id, res_name in enumerate(residue_names, start=1)
          )
      },
      bond_table=bond_table,
      chain_id=np.asarray(['P'] * len(atom_names), dtype=object),
      chain_type=np.asarray(
          [mmcif_names.PROTEIN_CHAIN] * len(atom_names), dtype=object
      ),
      res_name=np.asarray(res_names, dtype=object),
      atom_key=atom_keys,
      atom_name=atom_names_array,
      atom_element=np.asarray(atom_elements, dtype=object),
  )


def _amide_peptide_structure(
    *, bonds: tuple[cyclic_topology.TopologyEdge, ...]
):
  residue_atoms = (
      ('GLY', ('N', 'CA', 'C', 'O')),
      ('LYS', ('N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ')),
      ('ASP', ('N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2')),
      ('GLU', ('N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2')),
      ('GLY', ('N', 'CA', 'C', 'O', 'OXT')),
  )
  atom_names = []
  atom_elements = []
  res_ids = []
  res_names = []
  for res_id, (res_name, names) in enumerate(residue_atoms, start=1):
    atom_names.extend(names)
    atom_elements.extend(name[0] for name in names)
    res_ids.extend([res_id] * len(names))
    res_names.extend([res_name] * len(names))

  atom_names_array = np.asarray(atom_names, dtype=object)
  res_ids_array = np.asarray(res_ids, dtype=np.int32)
  atom_keys = np.arange(len(atom_names), dtype=np.int64)
  key_by_id = {
      ('P', int(res_id), str(atom_name)): int(atom_key)
      for atom_key, res_id, atom_name in zip(
          atom_keys, res_ids_array, atom_names_array, strict=True
      )
  }
  bond_table = structure.Bonds(
      key=np.arange(len(bonds), dtype=np.int64),
      from_atom_key=np.asarray(
          [key_by_id[edge.atom1] for edge in bonds], dtype=np.int64
      ),
      dest_atom_key=np.asarray(
          [key_by_id[edge.atom2] for edge in bonds], dtype=np.int64
      ),
      type=np.asarray(
          [mmcif_names.COVALENT_BOND] * len(bonds), dtype=object
      ),
      role=np.asarray(['.'] * len(bonds), dtype=object),
  )
  return structure.from_atom_arrays(
      name='cyclic_amide_test',
      res_id=res_ids_array,
      all_residues={
          'P': tuple(
              (res_name, res_id)
              for res_id, (res_name, _) in enumerate(
                  residue_atoms, start=1
              )
          )
      },
      bond_table=bond_table,
      chain_id=np.asarray(['P'] * len(atom_names), dtype=object),
      chain_type=np.asarray(
          [mmcif_names.PROTEIN_CHAIN] * len(atom_names), dtype=object
      ),
      res_name=np.asarray(res_names, dtype=object),
      atom_key=atom_keys,
      atom_name=atom_names_array,
      atom_element=np.asarray(atom_elements, dtype=object),
  )


def _stock_clean(struc: structure.Structure) -> structure.Structure:
  return structure_cleaning.clean_structure(
      struc,
      ccd={},  # Unused because standard-atom and leaving-atom filters are off.
      drop_missing_sequence=False,
      drop_non_standard_atoms=False,
      filter_waters=False,
      filter_hydrogens=False,
      filter_leaving_atoms=False,
      only_glycan_ligands_for_leaving_atoms=True,
      covalent_bonds_only=True,
      remove_polymer_polymer_bonds=True,
      remove_bad_bonds=False,
      fix_standalone_glycans=False,
  )


def _token_layout(num_peptide_residues: int = 8) -> atom_layout.AtomLayout:
  target_residues = 2
  num_tokens = target_residues + num_peptide_residues
  return atom_layout.AtomLayout(
      atom_name=np.asarray(['CA'] * num_tokens, dtype=object),
      res_id=np.asarray(
          [1, 2, *range(1, num_peptide_residues + 1)], dtype=np.int32
      ),
      chain_id=np.asarray(
          ['A'] * target_residues + ['P'] * num_peptide_residues, dtype=object
      ),
      atom_element=np.asarray(['C'] * num_tokens, dtype=object),
      res_name=np.asarray(['ALA'] * num_tokens, dtype=object),
      chain_type=np.asarray(
          [mmcif_names.PROTEIN_CHAIN] * num_tokens, dtype=object
      ),
  )


class CyclicTopologyTest(absltest.TestCase):

  def test_inputs_without_topology_are_feature_exact(self):
    tokens = _token_layout()
    padding = features.PaddingShapes(
        num_tokens=tokens.shape[0],
        msa_size=1,
        num_chains=2,
        num_templates=1,
        num_atoms=128,
    )
    token_features = features.TokenFeatures.compute_features(tokens, padding)
    topology_features = features.CyclicTopologyFeatures.compute_features(
        tokens, (), ()
    )
    self.assertFalse(topology_features.graph_preserving)
    official = featurization.create_relative_encoding(token_features, 32, 2)
    stock = featurization.create_relative_encoding(
        token_features, 32, 2, topology_features
    )
    np.testing.assert_array_equal(np.asarray(stock), np.asarray(official))

    contact_matrix = jnp.eye(tokens.shape[0], dtype=jnp.float32)
    np.testing.assert_array_equal(
        np.asarray(
            evoformer.add_topology_bond_hints(
                contact_matrix, topology_features
            )
        ),
        np.asarray(contact_matrix),
    )

  def test_topology_features_survive_model_input_dtype_filter(self):
    topology_features = features.CyclicTopologyFeatures.compute_features(
        _token_layout(),
        (_H2T, _SS),
        ((('P', 2), ('P', 4)),),
    )
    feature_dict = topology_features.as_data_dict()

    filtered = utils.remove_invalidly_typed_feats(feature_dict)

    self.assertEqual(filtered.keys(), feature_dict.keys())

  def test_stock_cleaning_loses_graph_and_topology_path_restores_inventory(self):
    input_struc = _peptide_structure(bonds=(_H2T, _SS))
    declared = ((_H2T.atom1, _H2T.atom2), (_SS.atom1, _SS.atom2))
    edges = cyclic_topology.classify_declared_topology_edges(
        input_struc, declared
    )
    cleaned = _stock_clean(input_struc)
    self.assertEmpty(tuple(cleaned.iter_bonds()))
    self.assertIn('OXT', cleaned.atom_name)

    minimal_cleaned = cleaned.filter_out(cleaned.atom_name == 'OXT')
    minimal_restored = cyclic_topology.restore_topology_bonds(
        minimal_cleaned, edges
    )
    cyclic_topology.validate_topology_bonds(minimal_restored, edges)
    self.assertNotIn('OXT', minimal_restored.atom_name)

    restored = cyclic_topology.restore_topology_bonds(cleaned, edges)
    cyclic_topology.validate_topology_bonds(restored, edges)
    self.assertIn('OXT', restored.atom_name)
    with self.assertRaisesRegex(ValueError, 'retained amide leaving atoms'):
      cyclic_topology.validate_topology_output(restored, edges)
    self.assertLen(tuple(restored.iter_bonds()), 2)
    restored_layout = atom_layout.atom_layout_from_structure(restored)
    filtered_layout = cyclic_topology.filter_topology_leaving_atoms(
        restored_layout,
        edges,
        _test_ccd(),  # pyrefly: ignore[bad-argument-type]
    )
    self.assertNotIn('OXT', filtered_layout.atom_name)
    output_struc = restored.filter_out(restored.atom_name == 'OXT')
    cyclic_topology.validate_topology_output(output_struc, edges)
    expected_atoms = [
        (str(chain), int(residue), str(atom))
        for chain, residue, atom in zip(
            input_struc.chain_id,
            input_struc.res_id,
            input_struc.atom_name,
            strict=True,
        )
        if atom != 'OXT'
    ]
    observed_atoms = [
        (str(chain), int(residue), str(atom))
        for chain, residue, atom in zip(
            output_struc.chain_id,
            output_struc.res_id,
            output_struc.atom_name,
            strict=True,
        )
    ]
    self.assertEqual(observed_atoms, expected_atoms)
    cyclic_topology.validate_serialized_structure(
        output_struc, output_struc.to_mmcif()
    )

  def test_disulfide_inventory_removes_no_heavy_atom(self):
    input_struc = _peptide_structure(bonds=(_SS,))
    edges = cyclic_topology.classify_declared_topology_edges(
        input_struc, ((_SS.atom1, _SS.atom2),)
    )
    restored = cyclic_topology.restore_topology_bonds(
        _stock_clean(input_struc), edges
    )
    np.testing.assert_array_equal(restored.atom_name, input_struc.atom_name)
    cyclic_topology.validate_structure_identity(restored, input_struc)

  def test_isopeptide_and_tail_lariat_remove_exact_heavy_atoms(self):
    ccd = _test_ccd()
    cases = (
        (_ISO_ASP, 'isopeptide', ('P', 3, 'OD2'), ('P', 3, 'OD1')),
        (_ISO_GLU, 'isopeptide', ('P', 4, 'OE2'), ('P', 4, 'OE1')),
        (_HEAD, 'head_lariat', ('P', 3, 'OD2'), ('P', 3, 'OD1')),
        (_TAIL, 'tail_lariat', ('P', 5, 'OXT'), None),
    )
    for edge, expected_kind, removed_atom, retained_atom in cases:
      with self.subTest(kind=expected_kind, removed_atom=removed_atom):
        input_struc = _amide_peptide_structure(bonds=(edge,))
        edges = cyclic_topology.classify_declared_topology_edges(
            input_struc, ((edge.atom2, edge.atom1),)
        )
        self.assertEqual(edges[0].kind, expected_kind)
        restored = cyclic_topology.restore_topology_bonds(
            _stock_clean(input_struc), edges
        )
        before_layout = atom_layout.atom_layout_from_structure(restored)
        filtered_layout = cyclic_topology.filter_topology_leaving_atoms(
            before_layout,
            edges,
            ccd,  # pyrefly: ignore[bad-argument-type]
        )
        before_ids = Counter(
            zip(
                before_layout.chain_id,
                before_layout.res_id,
                before_layout.atom_name,
                strict=True,
            )
        )
        after_ids = Counter(
            zip(
                filtered_layout.chain_id,
                filtered_layout.res_id,
                filtered_layout.atom_name,
                strict=True,
            )
        )
        self.assertEqual(before_ids - after_ids, Counter({removed_atom: 1}))
        if retained_atom is not None:
          self.assertEqual(after_ids[retained_atom], 1)

        remove_mask = np.array(
            [
                (str(chain), int(residue), str(atom)) == removed_atom
                for chain, residue, atom in zip(
                    restored.chain_id,
                    restored.res_id,
                    restored.atom_name,
                    strict=True,
                )
            ],
            dtype=bool,
        )
        output_struc = restored.filter_out(remove_mask)
        cyclic_topology.validate_topology_output(output_struc, edges)
        cyclic_topology.validate_serialized_structure(
            output_struc, output_struc.to_mmcif()
        )

  def test_isopeptide_requires_validated_ccd_bond_orders(self):
    input_struc = _amide_peptide_structure(bonds=(_ISO_ASP,))
    edges = cyclic_topology.classify_declared_topology_edges(
        input_struc, ((_ISO_ASP.atom1, _ISO_ASP.atom2),)
    )
    component = _test_ccd()['ASP']
    atom1 = component['_chem_comp_bond.atom_id_1']
    atom2 = component['_chem_comp_bond.atom_id_2']
    orders = component['_chem_comp_bond.value_order']
    acid_bond = next(
        index
        for index, pair in enumerate(zip(atom1, atom2, strict=True))
        if set(pair) == {'CG', 'OD2'}
    )
    orders[acid_bond] = 'DOUB'
    with self.assertRaisesRegex(ValueError, 'required unique'):
      cyclic_topology.filter_topology_leaving_atoms(
          atom_layout.atom_layout_from_structure(input_struc),
          edges,
          {'ASP': component},  # pyrefly: ignore[bad-argument-type]
      )

  def test_topology_endpoints_cannot_be_reused(self):
    input_struc = _amide_peptide_structure(bonds=(_ISO_ASP, _TAIL))
    with self.assertRaisesRegex(ValueError, 'endpoint is reused'):
      cyclic_topology.classify_declared_topology_edges(
          input_struc,
          (
              (_ISO_ASP.atom1, _ISO_ASP.atom2),
              (_TAIL.atom1, _TAIL.atom2),
          ),
      )

  def test_token_bond_pairs_json_roundtrip(self):
    payload = {
        'dialect': 'alphafold3',
        'version': folding_input.JSON_VERSION,
        'name': 'token_bond_pair_test',
        'modelSeeds': [1],
        'sequences': [{
            'protein': {
                'id': 'A',
                'sequence': 'ACDACG',
                'unpairedMsa': '',
                'pairedMsa': '',
                'templates': [],
            }
        }],
        'bondedAtomPairs': [[['A', 2, 'SG'], ['A', 5, 'SG']]],
        'tokenBondPairs': [[['A', 2], ['A', 5]]],
    }

    fold_input = folding_input.Input.from_json(json.dumps(payload))

    self.assertEqual(
        fold_input.token_bond_pairs, ((('A', 2), ('A', 5)),)
    )
    self.assertEqual(
        json.loads(fold_input.to_json())['tokenBondPairs'],
        [[['A', 2], ['A', 5]]],
    )
    payload_without_hints = {**payload, 'tokenBondPairs': []}
    stock_input = folding_input.Input.from_json(
        json.dumps(payload_without_hints)
    )
    self.assertNotIn('tokenBondPairs', json.loads(stock_input.to_json()))
    payload['tokenBondPairs'].append([['A', 5], ['A', 2]])
    with self.assertRaisesRegex(ValueError, 'reversed duplicate'):
      folding_input.Input.from_json(json.dumps(payload))

  def test_internal_backbone_bond_is_not_misclassified_as_terminal(self):
    internal = cyclic_topology.TopologyEdge(
        atom1=('P', 2, 'N'),
        atom2=('P', 4, 'C'),
        kind='head_to_tail',
    )
    full_struc = _peptide_structure(bonds=(internal,))
    input_struc = full_struc.filter(np.isin(full_struc.res_id, (2, 4)))
    self.assertEqual(set(input_struc.res_id), {2, 4})
    self.assertEqual(
        {int(res_id) for _, res_id in input_struc.all_residues['P']},
        {1, 2, 3, 4, 5},
    )
    with self.assertRaisesRegex(ValueError, 'first-residue N'):
      cyclic_topology.classify_declared_topology_edges(
          input_struc, ((internal.atom1, internal.atom2),)
      )

  def test_token_bond_pairs_require_non_h2t_declared_edges(self):
    h2t = cyclic_topology.TopologyEdge(
        atom1=('P', 1, 'N'),
        atom2=('P', 8, 'C'),
        kind='head_to_tail',
    )
    disulfide = cyclic_topology.TopologyEdge(
        atom1=('P', 2, 'SG'),
        atom2=('P', 7, 'SG'),
        kind='disulfide',
    )
    self.assertEqual(
        cyclic_topology.validate_token_bond_pairs(
            (h2t, disulfide), ((('P', 7), ('P', 2)),)
        ),
        ((('P', 2), ('P', 7)),),
    )
    with self.assertRaisesRegex(ValueError, 'cannot receive'):
      cyclic_topology.validate_token_bond_pairs(
          (h2t, disulfide), ((('P', 1), ('P', 8)),)
      )
    with self.assertRaisesRegex(ValueError, 'at least one'):
      cyclic_topology.validate_token_bond_pairs(
          (h2t, disulfide), ((('P', 3), ('P', 6)),)
      )

  def test_token_bond_pair_collapses_multiple_non_h2t_edges(self):
    head_lariat = cyclic_topology.TopologyEdge(
        atom1=('P', 5, 'CD'),
        atom2=('P', 1, 'N'),
        kind='head_lariat',
    )
    tail_lariat = cyclic_topology.TopologyEdge(
        atom1=('P', 1, 'NZ'),
        atom2=('P', 5, 'C'),
        kind='tail_lariat',
    )
    token_pair = ((('P', 5), ('P', 1)),)
    self.assertEqual(
        cyclic_topology.validate_token_bond_pairs(
            (head_lariat, tail_lariat), token_pair
        ),
        ((('P', 1), ('P', 5)),),
    )

    h2t = cyclic_topology.TopologyEdge(
        atom1=('P', 1, 'N'), atom2=('P', 5, 'C'), kind='head_to_tail'
    )
    with self.assertRaisesRegex(ValueError, 'cannot receive'):
      cyclic_topology.validate_token_bond_pairs(
          (h2t, tail_lariat), token_pair
      )

  def test_isopeptide_hint_is_endpoint_exact_without_crosslink_rpe(self):
    tokens = _token_layout()
    topology_features = features.CyclicTopologyFeatures.compute_features(
        tokens,
        (_ISO_ASP,),
        ((('P', 3), ('P', 2)),),
    )
    self.assertFalse(np.any(topology_features.rpe_token_mask))
    self.assertFalse(np.any(topology_features.rpe_pair_offsets))
    np.testing.assert_array_equal(
        topology_features.token_bond_indices[0], np.asarray([4, 3])
    )
    self.assertTrue(topology_features.token_bond_mask[0])
    self.assertFalse(np.any(topology_features.token_bond_mask[1:]))

  def test_h2t_only_rpe_changes_both_call_sites_without_width_change(self):
    tokens = _token_layout()
    h2t = cyclic_topology.TopologyEdge(
        atom1=('P', 1, 'N'),
        atom2=('P', 8, 'C'),
        kind='head_to_tail',
    )
    disulfide = cyclic_topology.TopologyEdge(
        atom1=('P', 2, 'SG'),
        atom2=('P', 7, 'SG'),
        kind='disulfide',
    )
    edges = (h2t, disulfide)
    padding = features.PaddingShapes(
        num_tokens=tokens.shape[0],
        msa_size=1,
        num_chains=2,
        num_templates=1,
        num_atoms=128,
    )
    token_features = features.TokenFeatures.compute_features(tokens, padding)
    native = featurization.create_relative_encoding(token_features, 32, 2)
    topology_features = features.CyclicTopologyFeatures.compute_features(
        tokens, edges, ()
    )
    h2t_only_features = features.CyclicTopologyFeatures.compute_features(
        tokens, (h2t,), ()
    )
    np.testing.assert_array_equal(
        topology_features.rpe_pair_offsets,
        h2t_only_features.rpe_pair_offsets,
    )
    modified = featurization.create_relative_encoding(
        token_features, 32, 2, topology_features
    )
    self.assertEqual(modified.shape, native.shape)
    rel_pos_width = 2 * 32 + 2
    np.testing.assert_array_equal(
        np.asarray(modified[..., rel_pos_width:]),
        np.asarray(native[..., rel_pos_width:]),
    )
    # Compact indices 1 and 6 are peptide positions 2 and 7. H2T periodicity
    # wraps their signed offset from -5 to +3; the S-S edge is not used by RPE.
    self.assertEqual(int(topology_features.rpe_pair_offsets[1, 6]), 3)
    self.assertFalse(
        np.array_equal(np.asarray(modified), np.asarray(native))
    )
    batch = types.SimpleNamespace(
        token_features=token_features, cyclic_topology=topology_features
    )
    evoformer_rpe = evoformer.topology_relative_encoding(
        batch, max_relative_idx=32, max_relative_chain=2
    )
    diffusion_rpe = diffusion_head.topology_relative_encoding(batch)
    self.assertEqual(evoformer_rpe.shape, native.shape)
    self.assertEqual(diffusion_rpe.shape, native.shape)
    np.testing.assert_array_equal(evoformer_rpe, modified)
    np.testing.assert_array_equal(diffusion_rpe, modified)

  def test_token_bond_hints_preserve_native_edges_and_are_endpoint_exact(self):
    tokens = _token_layout()
    h2t = cyclic_topology.TopologyEdge(
        atom1=('P', 1, 'N'),
        atom2=('P', 8, 'C'),
        kind='head_to_tail',
    )
    disulfide = cyclic_topology.TopologyEdge(
        atom1=('P', 2, 'SG'),
        atom2=('P', 7, 'SG'),
        kind='disulfide',
    )
    topology_features = features.CyclicTopologyFeatures.compute_features(
        tokens,
        (h2t, disulfide),
        ((('P', 2), ('P', 7)),),
    )
    baseline = jnp.zeros(
        (tokens.shape[0], tokens.shape[0]), dtype=jnp.float32
    )
    baseline = baseline.at[0, 1].set(1.0)
    baseline = baseline.at[1, 2].set(1.0)
    hinted = np.asarray(
        evoformer.add_topology_bond_hints(baseline, topology_features)
    )
    expected = np.asarray(baseline).copy()
    expected[3, 8] = expected[8, 3] = 1.0
    np.testing.assert_array_equal(hinted, expected)


if __name__ == '__main__':
  absltest.main()

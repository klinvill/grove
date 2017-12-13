import matplotlib
matplotlib.use('Agg')
import pytest
import os
import numpy as np
from mock import patch
from mock import MagicMock, Mock
import qutip as qt
import json

from pyquil.api import QVMConnection

from grove.tomography.tomography import (MAX_QUBITS_PROCESS_TOMO,
                                         default_channel_ops)
from grove.tomography.process_tomography import (DEFAULT_PROCESS_TOMO_SETTINGS,
                                                 process_tomography_programs,
                                                 do_process_tomography, ProcessTomography,
                                                 COMPLETELY_POSITIVE)
from grove.tomography.process_tomography import (TRACE_PRESERVING)
from grove.tomography.utils import (POVM_PI_BASIS, make_histogram,
                                    sample_bad_readout, basis_state_preps,
                                    estimate_assignment_probs, BAD_2Q_READOUT, SEED,
                                    EPS, CNOT_PROGRAM)
from grove.tomography.operator_utils import make_diagonal_povm


SHOTS_PATH = os.path.join(os.path.dirname(__file__), 'process_shots.json')
RESULTS_PATH = os.path.join(os.path.dirname(__file__), 'process_results.json')
sample_bad_readout = MagicMock(sample_bad_readout)
sample_bad_readout.side_effect = [np.array(shots) for shots in json.load(open(SHOTS_PATH, 'r'))]

cxn = MagicMock(QVMConnection)
cxn.run_and_measure.side_effect = json.load(open(RESULTS_PATH, 'r'))


def test_process_tomography():
    num_qubits = len(CNOT_PROGRAM.get_qubits())
    dimension = 2 ** num_qubits

    tomo_seq = list(process_tomography_programs(CNOT_PROGRAM))
    nsamples = 3000

    np.random.seed(SEED)
    # We need more samples on the readout to ensure convergence.
    state_prep_hists = [make_histogram(sample_bad_readout(p, 2 * nsamples, BAD_2Q_READOUT, cxn),
                                       dimension) for p in basis_state_preps(*range(num_qubits))]
    assignment_probs = estimate_assignment_probs(state_prep_hists)

    histograms = np.zeros((len(tomo_seq), dimension))

    for jj, p in enumerate(tomo_seq):
        histograms[jj] = make_histogram(sample_bad_readout(p, nsamples, BAD_2Q_READOUT, cxn),
                                        dimension)

    channel_ops = list(default_channel_ops(num_qubits))
    histograms = histograms.reshape((len(channel_ops), len(channel_ops), dimension))

    povm = make_diagonal_povm(POVM_PI_BASIS ** num_qubits, assignment_probs)
    cnot_ideal = qt.cnot()
    for settings in [
        DEFAULT_PROCESS_TOMO_SETTINGS,
        DEFAULT_PROCESS_TOMO_SETTINGS._replace(constraints={TRACE_PRESERVING}),
        DEFAULT_PROCESS_TOMO_SETTINGS._replace(constraints={TRACE_PRESERVING, COMPLETELY_POSITIVE}),
    ]:

        process_tomo = ProcessTomography.estimate_from_ssr(histograms, povm, channel_ops,
                                                           channel_ops,
                                                           settings)

        assert abs(1 - process_tomo.avg_gate_fidelity(cnot_ideal)) < EPS

        transfer_matrix = process_tomo.pauli_basis.transfer_matrix(qt.to_super(cnot_ideal))
        assert abs(1 - process_tomo.avg_gate_fidelity(transfer_matrix)) < EPS
        chi_rep = process_tomo.to_chi().data.toarray()
        # When comparing to the identity, the chi representation is quadratically larger than the
        # Hilbert space representation, so we take a square root.
        probabilty_scale = np.sqrt(chi_rep.shape[0])
        super_op_from_chi = np.zeros(process_tomo.pauli_basis.ops[0].shape, dtype=np.complex128)
        for i, si in enumerate(process_tomo.pauli_basis.ops):
            for j, sj in enumerate(process_tomo.pauli_basis.ops):
                contribution = chi_rep[i][j] * si.data.toarray().conj().T.dot(sj.data.toarray())
                super_op_from_chi += contribution / probabilty_scale
        assert np.isclose(np.eye(process_tomo.pauli_basis.ops[0].shape[0]), super_op_from_chi,
                          atol=EPS).all()
        choi_rep = process_tomo.to_choi()

        # Choi matrix should be a valid density matrix, scaled by the dimension of the system.
        assert np.isclose(np.trace(choi_rep.data.toarray()) / probabilty_scale, 1, atol=EPS)

        super_op = process_tomo.to_super()
        # The map should be trace preserving.
        assert np.isclose(np.sum(super_op[0]), 1, atol=EPS)

    assert abs(1 - process_tomo.avg_gate_fidelity(qt.to_super(cnot_ideal))) < EPS

    with patch("grove.tomography.utils.plot_pauli_transfer_matrix"), \
         patch("grove.tomography.process_tomography.plt") as mplt:
        mplt.subplots.return_value = Mock(), Mock()
        process_tomo.plot()


def test_do_process_tomography():
    nsamples = 3000
    qubits = list(range(MAX_QUBITS_PROCESS_TOMO + 1))
    # Test with too many qubits.
    with pytest.raises(ValueError):
        _ = do_process_tomography(CNOT_PROGRAM, nsamples,
                                  cxn, qubits)
    process_tomo, assignment_probs, histograms = do_process_tomography(CNOT_PROGRAM, nsamples, cxn)
    cnot_ideal = qt.cnot()
    assert abs(1 - process_tomo.avg_gate_fidelity(cnot_ideal)) < EPS
    for histogram_collection in histograms:
        for histogram in histogram_collection:
            assert np.sum(histogram) == nsamples
    num_qubits = len(CNOT_PROGRAM.get_qubits())
    assert np.isclose(assignment_probs, np.eye(2 ** num_qubits), atol=EPS).all()

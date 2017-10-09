##############################################################################
# Copyright 2016-2017 Rigetti Computing
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
##############################################################################

"""
The Jordan-Wigner Transform
"""
from pyquil.paulis import PauliTerm, PauliSum, ID, ZERO
import itertools
from operator import mul


def qubit_create(index):
    return 1 / 2. * (PauliTerm('X', index, 1.0) + PauliTerm('Y', index, -1.j))


def qubit_kill(index):
    return 1 / 2. * (PauliTerm('X', index, 1.0) + PauliTerm('Y', index, 1.j))


class JWTransform(object):
    """
    Jordan-Wigner object the appropriate Pauli operators
    """
    def create(self, index):
        """
        Fermion creation operator at orbital 'n'

        :param Int index: creation index
        """
        return self._operator_generator(index, -1.0)

    def kill(self, index):
        """
        Fermion annihilation operator at orbital 'n'

        :param Int index: annihilation index
        """
        return self._operator_generator(index, 1.0)

    def product_ops(self, indices, conjugate):
        """
        Convert a list of site indices and coefficients to a Pauli Operators
        list with the Jordan-Wigner (JW) transformation

        :param List indices: list of ints specifying the site the fermionic operator acts on,
                             e.g. [0,2,4,6]
        :param List conjugate: List of -1, 1 specifying which of the indices are
                               creation operators (-1) and which are annihilation
                               operators (1).  e.g. [-1,-1,1,1]
        """
        pterm = PauliTerm('I', 0, 1.0)
        for conj, index in zip(conjugate, indices):
            pterm = pterm * self._operator_generator(index, conj)

        pterm = pterm.simplify()
        return pterm

    def one_body_term(self, i, j):
        """
        Gives the equivalent PauliSum for the fermionic terms a_i^{\dagger}a_j + h.c.

        :param i: The first index in the term.
        :param j: The second index in the term.
        :return: The equivalent PauliSum for the desired one body term.
        :rtype: PauliSum
        """
        lower_index = min(i, j)
        upper_index = max(i, j)
        z_chain = 1
        ps = 0
        if lower_index != upper_index:
            for index in range(lower_index + 1, upper_index):
                z_chain *= PauliTerm('Z', index)

            for pauli in ['X', 'Y']:
                ps += (PauliTerm(pauli, lower_index)
                       * z_chain
                       * PauliTerm(pauli, upper_index))
            ps *= 0.5
        else:
            return PauliTerm('I', lower_index) + PauliTerm('Z', lower_index, -1.0)
        return ps

    def two_body_term(self, i, j, k, l):
        """
        Gives the equivalent PauliSum for the fermionic terms a_i^{\dagger}a_j^{\dagger}a_ka_l
         + h.c.

        :param i: The first index in the term.
        :param j: The second index in the term.
        :param k: The third index in the term.
        :param l: The fourth index in the term.
        :return: The equivalent PauliSum for the desired two body term.
        :rtype: PauliSum
        """
        if i == j or k == l:
            return PauliSum([ZERO])
        ps = PauliSum([ID])

        if len({i, j, k, l}) == 4:
            for paulis in itertools.product(['X', 'Y'], repeat=3):
                if paulis.count('X') % 2:
                    majority = 'X'
                else:
                    majority = 'Y'
                op1, op2, op3 = paulis
                sorted_operators = sorted([(i, op1), (j, op2),
                                           (k, op3), (l, majority)],
                                          key=lambda pair: pair[0])
                (a, operator_a), (b, operator_b), (c, operator_c), (d, operator_d) = sorted_operators

                operator = PauliTerm(operator_a, a)
                z_chain = 1
                for index in range(a + 1, b):
                    z_chain *= PauliTerm('Z', index)
                operator *= z_chain
                operator *= PauliTerm(operator_b, b)
                operator *= PauliTerm(operator_c, c)
                z_chain = 1
                for index in range(c + 1, d):
                    z_chain *= PauliTerm('Z', index)
                operator *= z_chain
                operator *= PauliTerm(operator_d, d)

                coefficient = .125
                parity_condition = bool(paulis[0] != paulis[1] or
                                        paulis[0] == paulis[2])
                if (i > j) ^ (k > l):
                    if not parity_condition:
                        coefficient *= -1.
                elif parity_condition:
                    coefficient *= -1.
                ps += coefficient * operator

        elif len({i, j, k, l}) == 3:
            if i == k:
                a, b = sorted([j, l])
                c = i
            elif i == l:
                a, b = sorted([j, k])
                c = i
            elif j == k:
                a, b = sorted([i, l])
                c = j
            elif j == l:
                a, b = sorted([i, k])
                c = j

            z_chain = 1
            for index in range(a + 1, b):
                z_chain *= PauliTerm('Z', index)

            pauli_z = PauliTerm('Z', c)
            for operator in ['X', 'Y']:
                operators = PauliTerm(operator, a) * z_chain * PauliTerm(operator, b)

                if (i == l) or (j == k):
                    coefficient = .25
                else:
                    coefficient = -.25

                hopping_term = coefficient * operators
                ps -= pauli_z * hopping_term
                ps += hopping_term

        elif len(set([i, j, k, l])) == 2:

            # Get coefficient.
            if i == l:
                coefficient = -0.5
            else:
                coefficient = 0.5

            ps -= coefficient
            ps += PauliTerm('Z', i, coefficient)
            ps += PauliTerm('Z', j, coefficient)
            ps -= (coefficient
                                     * PauliTerm('Z', min(i, j))
                                     * PauliTerm('Z', max(i, j)))

        return ps

    @staticmethod
    def _operator_generator(index, conj):
        """
        Internal method to generate the appropriate operator
        """
        pterm = PauliTerm('I', 0, 1.0)
        Zstring = PauliTerm('I', 0, 1.0)
        for j in range(index):
            Zstring = Zstring*PauliTerm('Z', j, 1.0)

        pterm1 = Zstring*PauliTerm('X', index, 0.5)
        scalar = 0.5 * conj * 1.0j
        pterm2 = Zstring*PauliTerm('Y', index, scalar)
        pterm = pterm * (pterm1 + pterm2)

        pterm = pterm.simplify()
        return pterm


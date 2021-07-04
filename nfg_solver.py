""" normal-form game (NFG) solver

(1) Take a normal form game as input, find minimax regret optimal mixture.
(2) Take a normal form game as input, find an MNE.

Author info removed

"""

import numpy as np
import sys
import nashpy as nash # Nash eq solver

def solve_game(payoffs):
    """ given payoff matrix for a zero-sum normal-form game,
    return first mixed equilibrium (may be multiple)
    returns a tuple of numpy arrays """
    # .vertex_enumeration()
    # .lemke_howson(initial_dropped_label=0) - does not return *all* equilibrium
    game = nash.Game(payoffs)
    equilibria = game.lemke_howson_enumeration()
    # equilibria = game.support_enumeration() # non_degenerate=False, tol=10e-16
    equilibrium = next(equilibria, None)

    # Lemke-Howson couldn't find equilibrium OR
    # Lemke-Howson return error - game may be degenerate. try other approaches
    print(equilibrium)
    print(equilibrium[0])
    print(equilibrium[1])
    if equilibrium is None or np.isnan(equilibrium[0]).any() or np.isnan(equilibrium[1]).any() or (equilibrium[0].shape != (payoffs.shape[0],) or equilibrium[1].shape != (payoffs.shape[1],)):
        # try other
        print('\n\n\n\n\nuh oh! degenerate solution')
        print('payoffs are\n', payoffs)
        equilibria = game.vertex_enumeration()
        equilibrium = next(equilibria, None)
        if equilibrium is None:
            print('\n\n\n\n\nuh oh x2! degenerate solution again!!')
            print('payoffs are\n', payoffs)
            equilibria = game.support_enumeration() # non_degenerate=False, tol=10e-16
            equilibrium = next(equilibria, None)

    assert equilibrium is not None
    return equilibrium


def solve_minimax_regret(payoffs):
    """ given payoff matrix for a zero-sum normal-form game,
    return a minimax regret optimal mixture


    subtract the max from each column.
    The mixed Nash equilibrium for this game
    appears to be the same as the minimax regret optimal mixed strategy

    TODO: need to confirm"""
    # degenerate case: if num rows == 1, regret will be 0
    if len(payoffs) == 1:
        eq1 = [1.]
        eq2 = np.zeros(len(payoffs[0]))
        eq2[0] = 1.
        return eq1, eq2
    # subtract max from each column
    mod_payoffs = np.array(payoffs)
    print('payoffs before')
    print(np.round(mod_payoffs, 2))
    mod_payoffs = mod_payoffs - mod_payoffs.max(axis=0)
    print('payoffs after')
    print(np.round(mod_payoffs, 2))
    return solve_game(mod_payoffs)

def solve_minimax_regret_with_regret_array(regret_array):
    """ given payoff matrix for a zero-sum normal-form game,
    return a minimax regret optimal mixture


    subtract the max from each column.
    The mixed Nash equilibrium for this game
    appears to be the same as the minimax regret optimal mixed strategy

    TODO: need to confirm"""
    # degenerate case: if num rows == 1, regret will be 0
    payoffs = regret_array
    if len(payoffs) == 1:
        eq1 = [1.]
        eq2 = np.zeros(len(payoffs[0]))
        eq2[0] = 1.
        return eq1, eq2
    # subtract max from each column
    mod_payoffs = np.array(regret_array)
    # print('payoffs before')
    # print(np.round(mod_payoffs, 2))
    # mod_payoffs = mod_payoffs - mod_payoffs.max(axis=0)
    print('Regrets:')
    print(np.round(mod_payoffs, 2))
    return solve_game(mod_payoffs)

def get_payoff(payoffs, agent_eq, nature_eq):
    """ given player mixed strategies, return expected payoff """

    game = nash.Game(payoffs)
    print('  payoff shape', game.payoff_matrices[0].shape)
    print('  def eq', np.round(agent_eq, 2))
    print('  nature eq', np.round(nature_eq, 2))
    expected_utility = game[agent_eq, nature_eq]
    print('  expected_utility', expected_utility)
    return expected_utility[0]


if __name__ == '__main__':
    payoffs = np.random.randint(0, 10, (2, 2))

    print(payoffs)

    equilibria = solve_game(payoffs)
    for eq in equilibria:
        print(eq)

    print('-----')
    A = np.array([[2, 3], [1, 4]])
    equilibria = solve_game(A)
    for eq in equilibria:
        print(eq)


    print('-----')
    B = np.array([[3/2, 3], [1, 4]])
    equilibria = solve_game(B)
    for eq in equilibria:
        print(eq)

    print('-----')
    C = np.array([[-2, 3], [3, -4]])
    equilibria = solve_game(C)
    for eq in equilibria:
        print(eq)

    sys.exit(0)



    # from https://code.activestate.com/recipes/496825-game-theory-payoff-matrix-solver/

    from operator import add, neg

    def solve(payoff_matrix, iterations=100):
        'Return the oddments (mixed strategy ratios) for a given payoff matrix'
        transpose = list(zip(*payoff_matrix))
        print(transpose)
        numrows = len(payoff_matrix)
        numcols = len(payoff_matrix[0])
        row_cum_payoff = [0] * numrows
        col_cum_payoff = [0] * numcols
        colpos = range(numcols)
        rowpos = map(neg, range(numrows))
        colcnt = [0] * numcols
        rowcnt = [0] * numrows
        active = 0
        for i in range(iterations):
            rowcnt[active] += 1
            col_cum_payoff = map(add, payoff_matrix[active], col_cum_payoff)
            print(col_cum_payoff, colpos)
            active = min(list(zip(col_cum_payoff, colpos)))[1]
            colcnt[active] += 1
            row_cum_payoff = map(add, transpose[active], row_cum_payoff)
            active = -max(list(zip(row_cum_payoff, rowpos)))[1]
        value_of_game = (max(row_cum_payoff) + min(col_cum_payoff)) / 2.0 / iterations
        return rowcnt, colcnt, value_of_game

    ###########################################
    # Example solutions to two pay-off matrices

    print(solve([[2,3,1,4], [1,2,5,4], [2,3,4,1], [4,2,2,2]]))   # Example on page 185
    print(solve([[4,0,2], [6,7,1]]))                             # Exercise 2 number 3

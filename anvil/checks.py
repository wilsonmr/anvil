# SPDX-License-Identifier: GPL-3.0-or-later
# Copywrite © 2021 anvil Michael Wilson, Joe Marsh Rossney, Luigi Del Debbio
"""
checks.py

Module containing checks. Checks are performed at "compile time" and ensure
correct configuration is used before executing any actions

"""

from reportengine.checks import make_argcheck, CheckError

@make_argcheck
def check_trained_with_free_theory(training_target_dist):
    """Check that supplied model is a free theory model which in the case of
    phi^4 means that lambda = 0
    """
    if training_target_dist.c_quartic != 0:
        raise CheckError(
            f"Theory parameters do not correspond to free theory. Quartic term "
            f"should be 0 and is instead = {training_target_dist.c_quartic}."
        )

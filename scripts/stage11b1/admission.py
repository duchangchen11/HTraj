"""Fail closed: a timing pass cannot override a failed evaluator gate."""
def training_permitted(data_passed, unit_tests_passed, precheck, budget):
    return bool(data_passed and unit_tests_passed
                and precheck.get('formal_training_gate_passed',False)
                and budget.get('thirty_epochs_plus_seven_validations_fit',False))

def test_legacy_exports():
    from duonlabs.legacy import DuonLabs, Forecast, ListofListsofNumbers
    assert DuonLabs is not None
    assert Forecast is not None
    assert ListofListsofNumbers is not None


def test_legacy_forecast_signature():
    """The legacy Forecast still takes the old (context, scenarios) shape."""
    from duonlabs.legacy import Forecast
    fc = Forecast(
        context=[[1700000000, 1.0, 2.0, 0.5, 1.5, 10.0]],
        scenarios=[[[1700000060, 1.0, 2.0, 0.5, 1.5, 10.0]]],
    )
    assert fc.cutoff_close == 1.5
    assert fc.n_scenarios == 1


def test_legacy_and_v2_coexist():
    """Both surfaces are importable in the same session."""
    import duonlabs
    import duonlabs.legacy
    assert duonlabs.DuonLabs is not duonlabs.legacy.DuonLabs
    assert duonlabs.Forecast is not duonlabs.legacy.Forecast

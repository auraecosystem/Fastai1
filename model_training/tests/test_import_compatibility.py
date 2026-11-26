def test_legacy_mixed_case_import_alias():
    from model_training.Lmlm_training import TrainConfig as LegacyTrainConfig
    from model_training.lmlm_training import TrainConfig as CanonicalTrainConfig

    assert LegacyTrainConfig is CanonicalTrainConfig

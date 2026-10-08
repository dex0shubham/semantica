"""Tests for the HuggingFace extraction helpers.

Mocks for transformers/torch/spacy/instructor are installed **per test** rather
than at module scope. pytest imports every test module during collection, so a
``sys.modules`` assignment here stays in place while later test modules are
imported, and they bind the stand-ins into their own globals — a
``tearDownModule`` cannot undo that, because by then collection has finished.
The ``instructor`` stand-in in particular made ``BaseProvider.generate_typed``
take its instructor branch for the rest of the run, so 31 tests in 12 other
files failed in full-suite runs while passing alone. Same shape as #1336.

The real modules import without any stand-in, and ``load_ner_model`` imports
torch and transformers inside the method, so the mocks only need to exist while
a test runs.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from semantica.semantic_extract.methods import (  # noqa: E402
    extract_entities_huggingface,
    extract_relations_huggingface,
    extract_triplets_huggingface,
)
from semantica.semantic_extract.ner_extractor import Entity  # noqa: E402


@pytest.fixture
def hf_mocks():
    """Stand in for the optional ML stack for the duration of one test."""
    mock_transformers = MagicMock()
    mock_pipeline = MagicMock()
    mock_transformers.pipeline = mock_pipeline

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    mock_config_module = MagicMock()
    mock_config_instance = MagicMock()
    mock_config_instance.get.return_value = {}
    mock_config_instance.get_optimization_config.return_value = {"enable_cache": False}
    mock_config_module.config = mock_config_instance
    mock_config_module.Config = MagicMock(return_value=mock_config_instance)

    with patch.dict(sys.modules, {
        "transformers": mock_transformers,
        "torch": mock_torch,
        "spacy": MagicMock(),
        "semantica.semantic_extract.config": mock_config_module,
    }):
        yield mock_transformers


def test_enhanced_impl(hf_mocks):
    mock_transformers = hf_mocks
    mock_pipeline = mock_transformers.pipeline

    # 1. Test NER with aggregation strategy
    
    # Setup mock pipeline return value
    mock_ner_pipeline = MagicMock()
    mock_ner_pipeline.return_value = [
        {"entity_group": "PERSON", "score": 0.99, "word": "Elon Musk", "start": 0, "end": 9},
    ]
    
    # Configure pipeline side effect
    def pipeline_side_effect(task, **kwargs):
        if task == "ner": return mock_ner_pipeline
        return MagicMock()
        
    mock_pipeline.side_effect = pipeline_side_effect
    
    # Test calling with aggregation_strategy
    entities = extract_entities_huggingface(
        "Elon Musk founded SpaceX.", 
        model="dslim/bert-base-NER", 
        aggregation_strategy="max"
    )
    
    # Verify aggregation_strategy was passed
    mock_pipeline.assert_any_call(
        "ner", 
        model="dslim/bert-base-NER", 
        device=-1, 
        aggregation_strategy="max",
        tokenizer=None
    )

    # 2. Test Relations with Input Formatting
    e1 = Entity(text="Elon Musk", label="PERSON", start_char=0, end_char=9)
    e2 = Entity(text="SpaceX", label="ORG", start_char=18, end_char=24)
    
    mock_rel_pipeline = MagicMock()
    mock_rel_pipeline.return_value = [{"label": "founded", "score": 0.9}]
    
    # Update pipeline mock to return rel pipeline
    def pipeline_side_effect_rel(task, **kwargs):
        if task == "ner": return mock_ner_pipeline
        if task == "text-classification": return mock_rel_pipeline
        return MagicMock()
        
    mock_pipeline.side_effect = pipeline_side_effect_rel
    
    relations = extract_relations_huggingface(
        "Elon Musk founded SpaceX.", 
        entities=[e1, e2], 
        model="some-relation-model"
    )
    
    # Verify input formatting
    # Check if ANY call contained the correct formatting
    found_match = False
    for call in mock_rel_pipeline.call_args_list:
        args, _ = call
        if "<subj> Elon Musk </subj>" in args[0] and "<obj> SpaceX </obj>" in args[0]:
            found_match = True
            break
    
    assert found_match, (
        "Did not find relation call with Elon Musk as subject; got "
        f"{[c[0] for c in mock_rel_pipeline.call_args_list]}"
    )

    # 3. Test Triplets with REBEL parsing
    
    # Mock Tokenizer and Model
    mock_tokenizer_instance = MagicMock()
    mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer_instance
    mock_tokenizer_instance.encode.return_value = MagicMock()
    # Mock decode to return REBEL format
    mock_tokenizer_instance.decode.return_value = "<s><triplet> Elon Musk <subj> founded <obj> SpaceX <triplet> SpaceX <subj> created <obj> Starship</s>"
    
    mock_model_instance = MagicMock()
    mock_transformers.AutoModelForSeq2SeqLM.from_pretrained.return_value = mock_model_instance
    mock_model_instance.generate.return_value = [MagicMock()]
    
    triplets = extract_triplets_huggingface(
        "Elon Musk founded SpaceX and created Starship.",
        model="Babelscape/rebel-large"
    )
    
    # Verify parsing
    assert len(triplets) == 2
    assert triplets[0].subject == "Elon Musk"
    assert triplets[0].predicate == "founded"
    assert triplets[0].object == "SpaceX"
    assert triplets[1].subject == "SpaceX"
    assert triplets[1].predicate == "created"
    assert triplets[1].object == "Starship"
    
    # Verify skip_special_tokens=False was passed
    mock_tokenizer_instance.decode.assert_called_with(
        mock_model_instance.generate.return_value[0], 
        skip_special_tokens=False
    )

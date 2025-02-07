import pytest
from datetime import datetime
from typing import Dict, Any

from src.cognitive.decision import (
    DecisionOutcome,
    Decision,
    DecisionCriteria,
    DecisionMaker,
    CodeGenerationDecisionMaker
)

def test_decision_creation():
    """Test basic decision creation."""
    decision = Decision(
        id="test_decision",
        context="test context",
        options=["option1", "option2"]
    )
    assert decision.id == "test_decision"
    assert decision.context == "test context"
    assert decision.options == ["option1", "option2"]
    assert decision.outcome == DecisionOutcome.UNCERTAIN
    assert isinstance(decision.timestamp, datetime)

def test_decision_criteria_creation():
    """Test decision criteria creation."""
    def eval_fn(x: Any) -> float:
        return 1.0
    
    criteria = DecisionCriteria(
        name="test_criteria",
        weight=0.5,
        evaluation_fn=eval_fn,
        description="Test criteria"
    )
    assert criteria.name == "test_criteria"
    assert criteria.weight == 0.5
    assert criteria.description == "Test criteria"
    assert criteria.evaluation_fn(None) == 1.0

class TestDecisionMaker:
    """Test suite for DecisionMaker."""
    
    @pytest.fixture
    def decision_maker(self):
        """Create a decision maker instance."""
        return DecisionMaker()
    
    def test_add_criterion(self, decision_maker):
        """Test adding a criterion."""
        def eval_fn(x: Any) -> float:
            return 1.0
        
        decision_maker.add_criterion(
            name="test",
            weight=0.5,
            evaluation_fn=eval_fn
        )
        
        assert "test" in decision_maker.criteria
        assert decision_maker.criteria["test"].weight == 0.5
    
    def test_add_criterion_invalid_weight(self, decision_maker):
        """Test adding a criterion with invalid weight."""
        def eval_fn(x: Any) -> float:
            return 1.0
        
        with pytest.raises(ValueError):
            decision_maker.add_criterion(
                name="test",
                weight=1.5,  # Invalid weight > 1
                evaluation_fn=eval_fn
            )
    
    def test_evaluate_option(self, decision_maker):
        """Test option evaluation."""
        def eval_fn1(x: Dict[str, Any]) -> float:
            return x.get("score1", 0.0)
        
        def eval_fn2(x: Dict[str, Any]) -> float:
            return x.get("score2", 0.0)
        
        decision_maker.add_criterion(
            name="criterion1",
            weight=0.6,
            evaluation_fn=eval_fn1
        )
        decision_maker.add_criterion(
            name="criterion2",
            weight=0.4,
            evaluation_fn=eval_fn2
        )
        
        option = {
            "score1": 0.8,
            "score2": 0.4
        }
        
        # Expected score: (0.8 * 0.6 + 0.4 * 0.4) = 0.64
        score = decision_maker.evaluate_option(option)
        assert score == pytest.approx(0.64)
    
    def test_make_decision(self, decision_maker):
        """Test decision making."""
        def eval_fn(x: Dict[str, Any]) -> float:
            return x.get("score", 0.0)
        
        decision_maker.add_criterion(
            name="test",
            weight=1.0,
            evaluation_fn=eval_fn
        )
        
        options = [
            {"score": 0.9},
            {"score": 0.5}
        ]
        
        decision = decision_maker.make_decision(
            decision_id="test",
            context="test context",
            options=options
        )
        
        assert decision.chosen_option == options[0]
        assert decision.outcome == DecisionOutcome.SUCCESSFUL
        assert decision.confidence == pytest.approx(0.9)
    
    def test_make_decision_uncertain(self, decision_maker):
        """Test decision making with uncertain outcome."""
        def eval_fn(x: Dict[str, Any]) -> float:
            return x.get("score", 0.0)
        
        decision_maker.add_criterion(
            name="test",
            weight=1.0,
            evaluation_fn=eval_fn
        )
        
        options = [
            {"score": 0.65},  # Below confidence threshold
            {"score": 0.60}   # Close to first option
        ]
        
        decision = decision_maker.make_decision(
            decision_id="test",
            context="test context",
            options=options
        )
        
        assert decision.outcome == DecisionOutcome.UNCERTAIN
    
    def test_decision_history(self, decision_maker):
        """Test decision history management."""
        def eval_fn(x: Any) -> float:
            return 1.0
        
        decision_maker.add_criterion(
            name="test",
            weight=1.0,
            evaluation_fn=eval_fn
        )
        
        decision1 = decision_maker.make_decision(
            decision_id="test1",
            context="context1",
            options=[1]
        )
        
        decision2 = decision_maker.make_decision(
            decision_id="test2",
            context="context2",
            options=[2]
        )
        
        history = decision_maker.get_decision_history()
        assert len(history) == 2
        assert decision1 in history
        assert decision2 in history
        
        filtered = decision_maker.get_decision_history("context1")
        assert len(filtered) == 1
        assert filtered[0] == decision1
    
    def test_revise_decision(self, decision_maker):
        """Test decision revision."""
        def eval_fn(x: Any) -> float:
            return 1.0
        
        decision_maker.add_criterion(
            name="test",
            weight=1.0,
            evaluation_fn=eval_fn
        )
        
        decision = decision_maker.make_decision(
            decision_id="test",
            context="test",
            options=[1]
        )
        
        revised = decision_maker.revise_decision(
            decision_id="test",
            new_outcome=DecisionOutcome.FAILED,
            reasoning="Test failed"
        )
        
        assert revised == decision
        assert revised.outcome == DecisionOutcome.FAILED
        assert "Test failed" in revised.reasoning

class TestCodeGenerationDecisionMaker:
    """Test suite for CodeGenerationDecisionMaker."""
    
    @pytest.fixture
    def code_decision_maker(self):
        """Create a code generation decision maker instance."""
        return CodeGenerationDecisionMaker()
    
    def test_default_criteria(self, code_decision_maker):
        """Test default criteria setup."""
        assert "complexity" in code_decision_maker.criteria
        assert "maintainability" in code_decision_maker.criteria
        assert "performance" in code_decision_maker.criteria
        assert "best_practices" in code_decision_maker.criteria
    
    def test_evaluate_code_option(self, code_decision_maker):
        """Test code option evaluation."""
        option = {
            "complexity_score": 3,  # Lower is better
            "readability": 8,
            "modularity": 7,
            "testability": 9,
            "performance_score": 8,
            "best_practices_score": 9
        }
        
        score = code_decision_maker.evaluate_option(option)
        assert 0 <= score <= 1
        
        # Test with poor scores
        poor_option = {
            "complexity_score": 8,
            "readability": 3,
            "modularity": 4,
            "testability": 2,
            "performance_score": 3,
            "best_practices_score": 4
        }
        
        poor_score = code_decision_maker.evaluate_option(poor_option)
        assert poor_score < score
    
    def test_make_code_decision(self, code_decision_maker):
        """Test making a code generation decision."""
        options = [
            {
                "name": "option1",
                "complexity_score": 3,
                "readability": 9,
                "modularity": 8,
                "testability": 9,
                "performance_score": 8,
                "best_practices_score": 9
            },
            {
                "name": "option2",
                "complexity_score": 7,
                "readability": 5,
                "modularity": 4,
                "testability": 6,
                "performance_score": 7,
                "best_practices_score": 5
            }
        ]
        
        decision = code_decision_maker.make_decision(
            decision_id="test",
            context="Generate handler",
            options=options
        )
        
        assert decision.chosen_option == options[0]
        assert decision.outcome == DecisionOutcome.SUCCESSFUL
        assert decision.confidence > 0.7  # Above threshold 
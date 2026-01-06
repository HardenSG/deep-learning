# -*- coding: utf-8 -*-
"""
Test Improved Predictor
"""

import sys
from pathlib import Path
import numpy as np
import torch
import json

sys.path.append(str(Path(__file__).parent))

from src.models.lstm_model import LSTMModel
from src.models.improved_predictor import ImprovedPredictor


def test_improved_predictor():
    """Test the improved predictor with comprehensive output"""
    print("\n" + "="*80)
    print("Testing Improved Predictor")
    print("="*80)

    # Create test data
    np.random.seed(42)
    torch.manual_seed(42)

    # Simulate input data (1 sample, 60 time steps, 29 features)
    X = np.random.randn(1, 60, 29) * 0.5 + 0.5

    # Create a simple LSTM model
    model = LSTMModel(input_size=29, hidden_size=32, num_layers=1, dropout=0.2)

    # Initialize improved predictor
    predictor = ImprovedPredictor(model, device="cpu")

    # Test comprehensive prediction
    stock_code = "600519"
    current_price = 1580.50

    print(f"\n[TEST] Predicting for {stock_code}")
    print(f"[TEST] Current Price: {current_price:.2f}")
    print(f"\n[INFO] Running comprehensive prediction with 50 Monte Carlo simulations...")

    result = predictor.get_comprehensive_prediction(
        X,
        current_price,
        stock_code,
        n_simulations=50
    )

    # Print result in a formatted way
    print("\n" + "="*80)
    print("PREDICTION RESULT")
    print("="*80)

    # Metadata
    print("\n[METADATA]")
    print(f"  Stock Code: {result['metadata']['stock_code']}")
    print(f"  Prediction Time: {result['metadata']['prediction_time']}")
    print(f"  Valid Until: {result['metadata']['prediction_valid_until']}")
    print(f"  N Simulations: {result['metadata']['n_simulations']}")

    # Data Quality
    dq = result['metadata']['data_quality']
    print(f"\n[DATA QUALITY]")
    print(f"  Score: {dq['score']}/100 ({dq['level']})")
    print(f"  Has NaN: {dq['has_nan']}")
    print(f"  Has Inf: {dq['has_inf']}")
    if dq['issues']:
        print(f"  Issues: {dq['issues']}")

    # Price
    print(f"\n[PRICE PREDICTION]")
    print(f"  Current:   {result['price']['current']:.2f}")
    print(f"  Predicted: {result['price']['predicted']:.2f}")
    print(f"  Change:    {result['price']['change_amount']:+.2f} ({result['price']['change_pct']:+.2f}%)")

    # Probability
    print(f"\n[PROBABILITY]")
    print(f"  Direction: {result['probability']['direction']}")
    print(f"  Up:        {result['probability']['up']:.1f}%")
    print(f"  Down:      {result['probability']['down']:.1f}%")
    print(f"  Large Up:  {result['probability']['large_up']:.1f}% (>5%)")
    print(f"  Large Down:{result['probability']['large_down']:.1f}% (<-5%)")

    # Uncertainty
    print(f"\n[UNCERTAINTY]")
    print(f"  Mean Return:   {result['uncertainty']['mean_return_pct']:+.2f}%")
    print(f"  Median Return: {result['uncertainty']['median_return_pct']:+.2f}%")
    print(f"  Std Return:    {result['uncertainty']['std_return_pct']:.2f}%")
    print(f"  Confidence Intervals:")
    ci = result['uncertainty']['confidence_intervals']
    print(f"    5%:  {ci['ci_5_pct']:+.2f}%")
    print(f"    25%: {ci['ci_25_pct']:+.2f}%")
    print(f"    75%: {ci['ci_75_pct']:+.2f}%")
    print(f"    95%: {ci['ci_95_pct']:+.2f}%")

    # Risk Metrics
    print(f"\n[RISK METRICS]")
    vol = result['risk_metrics']['volatility']
    print(f"  Volatility:")
    print(f"    Daily:      {vol['daily']:.4f}")
    print(f"    Annualized: {vol['annualized']:.2%}")
    print(f"    Level:      {vol['level']}")

    var = result['risk_metrics']['value_at_risk']
    print(f"  Value at Risk (95%):")
    print(f"    {var['description']}")
    print(f"    Amount: {var['var_95_amount']:+.2f}")

    ratios = result['risk_metrics']['ratios']
    print(f"  Ratios:")
    print(f"    Sharpe Ratio:       {ratios['sharpe_ratio']:.2f}")
    print(f"    Reward/Risk Ratio:  {ratios['reward_risk_ratio']:.2f}")

    # Trading Signals
    print(f"\n[TRADING SIGNALS]")
    ts = result['trading_signals']
    print(f"  Action: {ts['action']}")
    print(f"  Reason: {ts['reason']}")
    print(f"  Confidence: {ts['confidence']}")

    print(f"\n  Position:")
    print(f"    Suggested: {ts['position']['suggested_pct']}%")

    print(f"\n  Stop Loss:")
    print(f"    Price: {ts['stop_loss']['price']:.2f} ({ts['stop_loss']['pct']:.1f}%)")

    print(f"\n  Take Profit:")
    print(f"    Price: {ts['take_profit']['price']:.2f} ({ts['take_profit']['pct']:.1f}%)")

    risk_assess = ts['risk_assessment']
    print(f"\n  Risk Assessment:")
    print(f"    Overall Risk: {risk_assess['overall_risk']}")
    print(f"    Risk Score:   {risk_assess['risk_score']}/6")
    if risk_assess['warning']:
        print(f"    Warning:      {risk_assess['warning']}")

    # Disclaimer
    print(f"\n[DISCLAIMER]")
    print(f"  {result['disclaimer']['warning']}")
    print(f"  Limitations:")
    for limitation in result['disclaimer']['model_limitations']:
        print(f"    - {limitation}")

    print("\n" + "="*80)
    print("[OK] Improved Predictor Test Passed!")
    print("="*80)

    # Save result to JSON for inspection
    output_file = "test_prediction_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Full result saved to: {output_file}")

    return result


def compare_with_simple_predictor():
    """Compare improved predictor with simple predictor"""
    print("\n" + "="*80)
    print("Comparing Improved vs Simple Predictor")
    print("="*80)

    from src.models.enhanced_predictor import EnhancedPredictor

    # Create test data
    np.random.seed(42)
    torch.manual_seed(42)
    X = np.random.randn(1, 60, 29) * 0.5 + 0.5
    current_price = 1580.50

    # Simple predictor
    model1 = LSTMModel(input_size=29, hidden_size=32, num_layers=1, dropout=0.2)
    simple = EnhancedPredictor(model1, device="cpu")
    simple_result = simple.get_detailed_analysis(X, current_price, use_monte_carlo=False)

    # Improved predictor
    model2 = LSTMModel(input_size=29, hidden_size=32, num_layers=1, dropout=0.2)
    improved = ImprovedPredictor(model2, device="cpu")
    improved_result = improved.get_comprehensive_prediction(X, current_price, "600519", n_simulations=50)

    print("\n[COMPARISON]")
    print(f"\nSimple Predictor:")
    print(f"  Price Change: {simple_result['price_change_pct']:+.2f}%")
    print(f"  Up Probability: {simple_result['up_probability_pct']:.1f}%")
    print(f"  Confidence: {simple_result['confidence_level']}")
    print(f"  Recommendation: {simple_result['recommendation']['action']}")

    print(f"\nImproved Predictor:")
    print(f"  Price Change: {improved_result['price']['change_pct']:+.2f}%")
    print(f"  Up Probability: {improved_result['probability']['up']:.1f}%")
    print(f"  Confidence: {improved_result['trading_signals']['confidence']}")
    print(f"  Recommendation: {improved_result['trading_signals']['action']}")

    print(f"\nImproved Predictor Additional Features:")
    print(f"  + Confidence Intervals: [{improved_result['uncertainty']['confidence_intervals']['ci_5_pct']:.2f}%, {improved_result['uncertainty']['confidence_intervals']['ci_95_pct']:.2f}%]")
    print(f"  + Volatility: {improved_result['risk_metrics']['volatility']['level']}")
    print(f"  + Sharpe Ratio: {improved_result['risk_metrics']['ratios']['sharpe_ratio']:.2f}")
    print(f"  + Stop Loss: {improved_result['trading_signals']['stop_loss']['price']:.2f}")
    print(f"  + Take Profit: {improved_result['trading_signals']['take_profit']['price']:.2f}")
    print(f"  + Data Quality Score: {improved_result['metadata']['data_quality']['score']}/100")

    print("\n[OK] Comparison Complete!")
    print("="*80)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("IMPROVED PREDICTOR TEST SUITE")
    print("="*80)

    try:
        # Test 1: Basic functionality
        test_improved_predictor()

        # Test 2: Comparison
        compare_with_simple_predictor()

        print("\n" + "="*80)
        print("ALL TESTS PASSED!")
        print("="*80)
        print("\nNext Steps:")
        print("1. Integrate improved predictor into API")
        print("2. Update API endpoints in main.py")
        print("3. Test with real stock data")
        print("4. Deploy to production")
        print("="*80)

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

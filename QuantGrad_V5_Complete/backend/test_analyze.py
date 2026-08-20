from server import analyze
try:
    result = analyze('BTCUSDT', 800, 260)
    print('OK: Analysis generated!')
    print(f"Signal: {result['result']['signal_label']}")
    print(f"Confidence: {result['result']['confidence']:.1%}")
except Exception as e:
    print(f'ERROR: {e}')
    import traceback
    traceback.print_exc()
